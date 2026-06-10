"""GPU end-to-end test for the KV event pipeline.

Deploys two real vLLM replicas with KV-cache events enabled, sends prompts
directly to each replica's backend HTTP server, and asserts the engines'
BlockStored/AllBlocksCleared events arrive in the deployment-scoped
``KVRouterActor``'s consumer keyed by the right worker with exact contents.

``KVAwareRouter`` replica selection lands in a later branch, so the actor is
attached via ``deployment_actors`` and the engine KV-events configuration is
applied directly (the builder applies it when KVAwareRouter is selected,
covered by CPU tests); event bridging is router-independent.
"""

import asyncio
import os
import sys
import tempfile

import pytest
import requests

os.environ["RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING"] = "1"
os.environ["RAY_SERVE_ENABLE_DIRECT_INGRESS"] = "1"

import ray  # noqa: E402
from ray import serve  # noqa: E402
from ray._common.test_utils import async_wait_for_condition  # noqa: E402
from ray.llm._internal.serve.core.ingress.builder import (  # noqa: E402
    _build_direct_streaming_llm_deployment,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (  # noqa: E402
    KV_ROUTER_ACTOR_NAME,
    KVRouterActor,
    get_worker_id,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_events import (  # noqa: E402
    configure_kv_events_for_kv_routing,
)
from ray.serve._private.constants import (  # noqa: E402
    SERVE_DEPLOYMENT_ACTOR_PREFIX,
    SERVE_NAMESPACE,
)
from ray.serve.config import DeploymentActorConfig  # noqa: E402
from ray.serve.llm import LLMConfig, ModelLoadingConfig  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
APP_NAME = "kv_events_gpu_test"
NUM_REPLICAS = 2
BLOCK_SIZE = 16
MAX_TOKENS = 50
MESSAGES = [
    {
        "role": "user",
        "content": (
            "Repeat the following sentence five times: the quick brown fox "
            "jumps over the lazy dog while the cat watches from the fence."
        ),
    }
]


def discover_deployment_actor(app_name, deployment_name, actor_name):
    """Resolve a deployment-scoped actor by its registered name.

    The test driver isn't a replica, so ``get_deployment_actor`` is
    unavailable; match the name's stable prefix/suffix instead (the middle
    embeds an opaque code_version).
    """
    prefix = f"{SERVE_DEPLOYMENT_ACTOR_PREFIX}{app_name}::{deployment_name}::"
    suffix = f"::{actor_name}"
    for entry in ray.util.list_named_actors(all_namespaces=True):
        name = entry.get("name") or ""
        if (
            entry.get("namespace") == SERVE_NAMESPACE
            and name.startswith(prefix)
            and name.endswith(suffix)
        ):
            return ray.get_actor(name, namespace=SERVE_NAMESPACE)
    return None


def post_chat(endpoint, max_tokens=MAX_TOKENS):
    host, port = endpoint
    response = requests.post(
        f"http://{host}:{port}/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": MESSAGES,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        timeout=120,
    )
    assert response.status_code == 200, response.text
    return response.json()


def tokenize_prompt(endpoint, messages=MESSAGES):
    """The engine's exact token ids for a chat-templated prompt."""
    host, port = endpoint
    response = requests.post(
        f"http://{host}:{port}/tokenize",
        json={"model": MODEL_ID, "messages": messages, "add_generation_prompt": True},
        timeout=60,
    )
    assert response.status_code == 200, response.text
    return response.json()["tokens"]


class TestKvEventsGPU:
    @pytest.fixture(scope="class")
    def deployed_handle(self):
        """Deploy two direct-streaming LLMServer replicas with KV events on."""
        if not ray.is_initialized():
            # An empty working_dir keeps the runtime-env package tiny; the
            # repo root would exceed the upload size limit.
            ray.init(
                address="auto",
                runtime_env={"working_dir": tempfile.mkdtemp(prefix="kv_ev_wd_")},
            )
        serve.shutdown()  # ensure no prior app is holding GPU memory

        llm_config = LLMConfig(
            model_loading_config=ModelLoadingConfig(
                model_id=MODEL_ID,
                model_source=MODEL_ID,
            ),
            deployment_config=dict(
                autoscaling_config=dict(
                    min_replicas=NUM_REPLICAS, max_replicas=NUM_REPLICAS
                ),
                deployment_actors=[
                    DeploymentActorConfig(
                        name=KV_ROUTER_ACTOR_NAME,
                        actor_class=KVRouterActor,
                        actor_options={"num_cpus": 0},
                    )
                ],
            ),
            engine_kwargs=dict(
                block_size=BLOCK_SIZE,
                enable_prefix_caching=True,
                max_model_len=2048,
                enforce_eager=True,
                gpu_memory_utilization=0.4,
                use_tqdm_on_load=False,
            ),
            experimental_configs={"KV_EVENTS_PORT_BASE": 21557},
            placement_group_config={"bundles": [{"GPU": 1}]},
            # The replica worker process reads these constants at import time.
            runtime_env=dict(
                env_vars={
                    "VLLM_DISABLE_COMPILE_CACHE": "1",
                    "RAY_SERVE_ENABLE_DIRECT_INGRESS": "1",
                    "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING": "1",
                    # /reset_prefix_cache is a vLLM dev-mode endpoint.
                    "VLLM_SERVER_DEV_MODE": "1",
                },
            ),
            log_engine_metrics=False,
        )
        configure_kv_events_for_kv_routing(llm_config)

        app = _build_direct_streaming_llm_deployment(llm_config)
        handle = serve.run(app, name=APP_NAME)
        yield handle
        serve.shutdown()

    async def _discover_replicas(self, handle):
        """Map each replica's worker id to its backend HTTP endpoint."""
        endpoints = {}
        for _ in range(120):
            async with handle.choose_replica() as selection:
                replica = selection._replica
                if replica.backend_http_endpoint is not None:
                    worker_id = get_worker_id(replica.replica_id.unique_id)
                    endpoints[worker_id] = replica.backend_http_endpoint
            if len(endpoints) == NUM_REPLICAS:
                return endpoints
            await asyncio.sleep(0.5)
        raise AssertionError(
            f"Expected {NUM_REPLICAS} replicas with backend endpoints, "
            f"found {len(endpoints)}."
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_kv_events_reach_router_actor(self, deployed_handle):
        actor = discover_deployment_actor(
            APP_NAME, deployed_handle.deployment_name, KV_ROUTER_ACTOR_NAME
        )
        assert actor is not None, "KV router actor was not discoverable"

        endpoints = await self._discover_replicas(deployed_handle)
        worker_ids = sorted(endpoints)

        # The same prompt on each replica caches the same content.
        usages = {}
        for worker_id in worker_ids:
            usages[worker_id] = post_chat(endpoints[worker_id])["usage"]

        async def all_workers_reported():
            return await actor.get_kv_event_worker_ids.remote() == worker_ids

        await async_wait_for_condition(all_workers_reported, timeout=30)

        prompt_token_ids = tokenize_prompt(endpoints[worker_ids[0]])
        prompt_blocks = len(prompt_token_ids) // BLOCK_SIZE
        chunks_by_worker = {}
        for worker_id in worker_ids:
            usage = usages[worker_id]
            assert usage["prompt_tokens"] == len(prompt_token_ids)
            assert usage["completion_tokens"] == MAX_TOKENS

            blocks = await actor.get_kv_cached_blocks.remote(worker_id)
            chunks = list(blocks.values())
            assert all(len(chunk) == BLOCK_SIZE for chunk in chunks)

            # All full blocks of the prompt+completion stream are cached; the
            # block that fills on the final step may not be committed.
            total_tokens = usage["prompt_tokens"] + usage["completion_tokens"]
            full_blocks = total_tokens // BLOCK_SIZE
            assert len(chunks) in (full_blocks, full_blocks - 1)

            # Stored events arrive in chain order: the prompt's full blocks
            # reproduce the tokenized prompt exactly.
            prompt_prefix = [t for chunk in chunks[:prompt_blocks] for t in chunk]
            assert prompt_prefix == prompt_token_ids[: prompt_blocks * BLOCK_SIZE]

            chunks_by_worker[worker_id] = chunks

        # Identical content cached on both workers (engine block hashes are
        # process-seeded, so contents, not hashes, are comparable).
        first, second = (chunks_by_worker[w][:prompt_blocks] for w in worker_ids)
        assert first == second

        # A repeated prompt is a prefix-cache hit: greedy decode rebuilds the
        # identical block chain, so the worker's cached view is unchanged.
        reused_worker = worker_ids[0]
        counts_before = await actor.get_kv_event_counts.remote()
        blocks_before = await actor.get_kv_cached_blocks.remote(reused_worker)
        repeat_usage = post_chat(endpoints[reused_worker])["usage"]
        assert repeat_usage == usages[reused_worker]

        await asyncio.sleep(2)
        assert await actor.get_kv_cached_blocks.remote(reused_worker) == blocks_before
        counts_after = await actor.get_kv_event_counts.remote()
        new_stored = (
            counts_after[reused_worker]["block_stored"]
            - counts_before[reused_worker]["block_stored"]
        )
        # The prompt's cached prefix is not re-stored; only blocks the rerun
        # fills again past the prefix hit (decode region plus at most the
        # previously uncommitted tail) re-emit stored events.
        decode_region_blocks = len(chunks_by_worker[reused_worker]) - prompt_blocks
        assert new_stored <= decode_region_blocks + 1

        # Resetting one replica's prefix cache clears only its worker's view.
        host, port = endpoints[reused_worker]
        response = requests.post(f"http://{host}:{port}/reset_prefix_cache", timeout=60)
        assert response.status_code == 200, response.text
        # The engine drains queued KV events on scheduler steps, so a small
        # follow-up request flushes the AllBlocksCleared event.
        flush_messages = [{"role": "user", "content": "Hi."}]
        flush_response = requests.post(
            f"http://{host}:{port}/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": flush_messages,
                "max_tokens": 2,
            },
            timeout=120,
        )
        assert flush_response.status_code == 200, flush_response.text

        async def cleared():
            counts = await actor.get_kv_event_counts.remote()
            return counts[reused_worker].get("all_blocks_cleared") == 1

        await async_wait_for_condition(cleared, timeout=30)
        # The clear applies before the flush request's own stores: only the
        # flush request's short chain remains, the pre-reset chain is gone.
        blocks = await actor.get_kv_cached_blocks.remote(reused_worker)
        flush_tokens = [t for chunk in blocks.values() for t in chunk]
        flush_prompt_ids = tokenize_prompt(endpoints[reused_worker], flush_messages)
        assert len(flush_tokens) <= len(flush_prompt_ids) + 2
        prompt_full = len(flush_prompt_ids) // BLOCK_SIZE * BLOCK_SIZE
        assert flush_tokens[:prompt_full] == flush_prompt_ids[:prompt_full]
        untouched_blocks = await actor.get_kv_cached_blocks.remote(worker_ids[1])
        assert list(untouched_blocks.values()) == chunks_by_worker[worker_ids[1]]


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(address="auto")
    sys.exit(pytest.main(["-v", "-s", __file__]))
