import contextlib
import itertools
import sys
from typing import List

import msgspec
import pytest
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
)

import ray
from ray._common.test_utils import async_wait_for_condition
from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.core.server.builder import build_llm_deployment
from ray.llm._internal.serve.engines.vllm.kv_transfer.factory import (
    KVConnectorBackendFactory,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (
    KVRouterActor,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_event_publisher import (
    KvEventPublisher,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_events import (
    DYNAMO_KV_CONNECTOR,
    DYNAMO_KV_CONNECTOR_MODULE_PATH,
    assign_replica_kv_events_endpoint,
    configure_kv_events_for_kv_routing,
    resolve_consolidator_endpoints,
    resolve_kv_event_source_endpoint,
)
from ray.serve.llm.request_router import KVAwareRouter

BLOCK_SIZE = 16
WORKER_ID = 7001


def make_llm_config(**kwargs) -> LLMConfig:
    return LLMConfig(
        model_loading_config={
            "model_id": "qwen-0.5b",
            "model_source": "Qwen/Qwen2.5-0.5B-Instruct",
        },
        accelerator_type=None,
        **kwargs,
    )


def make_kv_aware_llm_config(**kwargs) -> LLMConfig:
    return make_llm_config(
        deployment_config={
            "autoscaling_config": {"min_replicas": 1, "max_replicas": 1},
            "request_router_config": {"request_router_class": KVAwareRouter},
        },
        **kwargs,
    )


def stored(block_hashes, token_ids, parent=None, block_size=BLOCK_SIZE):
    return BlockStored(
        block_hashes=block_hashes,
        parent_block_hash=parent,
        token_ids=token_ids,
        block_size=block_size,
        lora_id=None,
        medium="GPU",
        lora_name=None,
    )


class TestConfigureKvEvents:
    def test_build_enables_kv_events(self):
        """Building a KVAwareRouter deployment enables engine KV events."""
        llm_config = make_kv_aware_llm_config()
        build_llm_deployment(llm_config)

        assert llm_config.engine_kwargs["kv_events_config"] == {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": "tcp://*:5557",
        }
        # The Dynamo KVBM connector is not installed in this environment.
        assert "kv_transfer_config" not in llm_config.engine_kwargs

    def test_build_without_kv_aware_router_is_untouched(self):
        llm_config = make_llm_config(
            deployment_config={
                "autoscaling_config": {"min_replicas": 1, "max_replicas": 1}
            },
        )
        build_llm_deployment(llm_config)

        assert "kv_events_config" not in llm_config.engine_kwargs

    def test_user_kv_events_config_is_respected(self):
        llm_config = make_kv_aware_llm_config(
            engine_kwargs={
                "kv_events_config": {
                    "enable_kv_cache_events": True,
                    "publisher": "zmq",
                    "endpoint": "tcp://*:6000",
                    "buffer_steps": 5,
                }
            },
        )
        build_llm_deployment(llm_config)

        assert llm_config.engine_kwargs["kv_events_config"]["endpoint"] == (
            "tcp://*:6000"
        )
        assert llm_config.engine_kwargs["kv_events_config"]["buffer_steps"] == 5

    def test_port_base_override(self):
        llm_config = make_kv_aware_llm_config(
            experimental_configs={"KV_EVENTS_PORT_BASE": 21000},
        )
        configure_kv_events_for_kv_routing(llm_config)

        assert llm_config.engine_kwargs["kv_events_config"]["endpoint"] == (
            "tcp://*:21000"
        )

    def test_dynamo_connector_injected_when_kvbm_installed(self, monkeypatch):
        """With the kvbm package importable, the Dynamo connector is selected."""
        monkeypatch.setattr(
            "ray.llm._internal.serve.routing_policies.kv_aware.kv_events."
            "importlib.util.find_spec",
            lambda name: object() if name == "kvbm" else None,
        )
        llm_config = make_kv_aware_llm_config()
        configure_kv_events_for_kv_routing(llm_config)

        assert llm_config.engine_kwargs["kv_transfer_config"] == {
            "kv_connector": DYNAMO_KV_CONNECTOR,
            "kv_connector_module_path": DYNAMO_KV_CONNECTOR_MODULE_PATH,
            "kv_role": "kv_both",
        }

    def test_user_kv_transfer_config_is_respected(self, monkeypatch):
        monkeypatch.setattr(
            "ray.llm._internal.serve.routing_policies.kv_aware.kv_events."
            "importlib.util.find_spec",
            lambda name: object(),
        )
        user_config = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        llm_config = make_kv_aware_llm_config(
            engine_kwargs={"kv_transfer_config": dict(user_config)},
        )
        configure_kv_events_for_kv_routing(llm_config)

        assert llm_config.engine_kwargs["kv_transfer_config"] == user_config


class TestReplicaEndpoints:
    @pytest.fixture
    def replica_rank(self, monkeypatch):
        def set_rank(rank):
            monkeypatch.setattr(
                "ray.llm._internal.serve.routing_policies.kv_aware.kv_events."
                "_replica_rank",
                lambda: rank,
            )

        return set_rank

    def test_no_kv_events_is_noop(self):
        llm_config = make_llm_config()
        assign_replica_kv_events_endpoint(llm_config)

        assert "kv_events_config" not in llm_config.engine_kwargs
        assert resolve_kv_event_source_endpoint(llm_config) is None

    def test_replica_rank_offsets_port(self, replica_rank):
        """Colocated replicas must bind distinct KV-events ports."""
        replica_rank(2)
        llm_config = make_kv_aware_llm_config()
        configure_kv_events_for_kv_routing(llm_config)
        assign_replica_kv_events_endpoint(llm_config)

        assert llm_config.engine_kwargs["kv_events_config"]["endpoint"] == (
            "tcp://*:5559"
        )
        assert resolve_kv_event_source_endpoint(llm_config) == "tcp://127.0.0.1:5559"

    def test_data_parallel_rank_is_offset_by_vllm(self, replica_rank):
        """With data_parallel_rank, vLLM offsets the bind port internally, so
        the configured endpoint stays at the base and only the subscriber
        endpoint is offset."""
        replica_rank(5)
        llm_config = make_kv_aware_llm_config(
            engine_kwargs={"data_parallel_rank": 3},
        )
        configure_kv_events_for_kv_routing(llm_config)
        assign_replica_kv_events_endpoint(llm_config)

        endpoint = llm_config.engine_kwargs["kv_events_config"]["endpoint"]
        assert endpoint == "tcp://*:5557"
        assert resolve_kv_event_source_endpoint(llm_config) == "tcp://127.0.0.1:5560"
        offset_by_vllm = ZmqEventPublisher.offset_endpoint_port(endpoint, 3)
        assert offset_by_vllm == "tcp://*:5560"

    def test_consolidator_endpoints_with_dynamo_connector(self, replica_rank):
        """The DynamoConnector backend wires per-replica consolidator
        endpoints into additional_config and the publisher consumes the
        consolidated stream."""
        replica_rank(1)
        llm_config = make_kv_aware_llm_config(
            engine_kwargs={
                "kv_transfer_config": {
                    "kv_connector": DYNAMO_KV_CONNECTOR,
                    "kv_connector_module_path": DYNAMO_KV_CONNECTOR_MODULE_PATH,
                    "kv_role": "kv_both",
                }
            },
        )
        configure_kv_events_for_kv_routing(llm_config)
        assign_replica_kv_events_endpoint(llm_config)

        backend = KVConnectorBackendFactory.create_backend(
            DYNAMO_KV_CONNECTOR, llm_config
        )
        backend.setup()

        assert llm_config.engine_kwargs["additional_config"][
            "consolidator_endpoints"
        ] == [
            "tcp://127.0.0.1:5558",
            "tcp://0.0.0.0:57002",
            "tcp://127.0.0.1:57002",
        ]
        assert resolve_kv_event_source_endpoint(llm_config) == "tcp://127.0.0.1:57002"

    def test_user_consolidator_endpoints_are_respected(self, replica_rank):
        replica_rank(0)
        user_endpoints = ["tcp://127.0.0.1:1", "tcp://0.0.0.0:2", "tcp://127.0.0.1:2"]
        llm_config = make_kv_aware_llm_config(
            engine_kwargs={
                "kv_transfer_config": {"kv_connector": DYNAMO_KV_CONNECTOR},
                "additional_config": {"consolidator_endpoints": user_endpoints},
            },
        )
        configure_kv_events_for_kv_routing(llm_config)
        assign_replica_kv_events_endpoint(llm_config)
        KVConnectorBackendFactory.create_backend(
            DYNAMO_KV_CONNECTOR, llm_config
        ).setup()

        assert (
            llm_config.engine_kwargs["additional_config"]["consolidator_endpoints"]
            is user_endpoints
        )

    def test_consolidator_requires_kv_events(self):
        llm_config = make_kv_aware_llm_config()
        with pytest.raises(ValueError, match="kv_events_config"):
            resolve_consolidator_endpoints(llm_config)


@pytest.fixture(scope="module")
def ray_instance():
    if not ray.is_initialized():
        ray.init(address="auto")
    yield


@ray.remote(num_cpus=0)
class LocalKVRouterActor(KVRouterActor.__ray_actor_class__):
    """The real KVRouterActor with replica tracking disabled (no Serve
    controller in these tests)."""

    def _start_replica_tracking(self) -> None:
        pass


_test_ports = itertools.count(21811)


def make_bridge(actor, port, worker_id=WORKER_ID):
    return KvEventPublisher(
        kv_router_actor=actor,
        worker_id=worker_id,
        kv_block_size=BLOCK_SIZE,
        zmq_endpoint=f"tcp://127.0.0.1:{port}",
    )


async def publish_and_wait(actor, engine_publisher, batches: List[KVEventBatch]):
    """Publish batches and wait until the actor has consumed all of them."""
    counts_before = sum(
        sum(c.values()) for c in (await actor.get_kv_event_counts.remote()).values()
    )
    num_events = sum(len(b.events) for b in batches)
    for batch in batches:
        engine_publisher.publish(batch)

    async def consumed():
        counts = await actor.get_kv_event_counts.remote()
        return (
            sum(sum(c.values()) for c in counts.values()) == counts_before + num_events
        )

    await async_wait_for_condition(consumed, timeout=10)


class TestKvEventPipeline:
    """End-to-end: vLLM's production ZMQ publisher -> KvEventPublisher bridge
    -> KVRouterActor's KvEventConsumer."""

    @pytest.mark.asyncio
    async def test_stored_removed_cleared(self, ray_instance):
        actor = LocalKVRouterActor.remote()
        port = next(_test_ports)
        engine_pub = ZmqEventPublisher(
            data_parallel_rank=0, endpoint=f"tcp://*:{port}", topic=""
        )
        bridge = make_bridge(actor, port)
        try:
            # Two chained blocks stored, then one removed.
            await publish_and_wait(
                actor,
                engine_pub,
                [
                    KVEventBatch(
                        ts=1.0,
                        events=[stored([11, 22], list(range(2 * BLOCK_SIZE)))],
                    ),
                    KVEventBatch(
                        ts=2.0, events=[BlockRemoved(block_hashes=[11], medium="GPU")]
                    ),
                ],
            )
            assert await actor.get_kv_event_worker_ids.remote() == [WORKER_ID]
            blocks = await actor.get_kv_cached_blocks.remote(WORKER_ID)
            assert blocks == {22: list(range(BLOCK_SIZE, 2 * BLOCK_SIZE))}

            await publish_and_wait(
                actor, engine_pub, [KVEventBatch(ts=3.0, events=[AllBlocksCleared()])]
            )
            assert await actor.get_kv_cached_blocks.remote(WORKER_ID) == {}
            assert (await actor.get_kv_event_counts.remote())[WORKER_ID] == {
                "block_stored": 1,
                "block_removed": 1,
                "all_blocks_cleared": 1,
            }
        finally:
            await bridge.close()
            engine_pub.shutdown()

    @pytest.mark.asyncio
    async def test_per_worker_isolation_and_removal(self, ray_instance):
        """Two replicas' bridges feed the same actor without cross-talk, and
        worker removal drops the departed worker's state."""
        actor = LocalKVRouterActor.remote()
        ports = (next(_test_ports), next(_test_ports))
        worker_ids = (7001, 7002)
        engine_pubs = [
            ZmqEventPublisher(data_parallel_rank=0, endpoint=f"tcp://*:{p}", topic="")
            for p in ports
        ]
        bridges = [
            make_bridge(actor, port, worker_id)
            for port, worker_id in zip(ports, worker_ids)
        ]
        try:
            for i, engine_pub in enumerate(engine_pubs):
                await publish_and_wait(
                    actor,
                    engine_pub,
                    [
                        KVEventBatch(
                            ts=1.0,
                            events=[stored([100 + i], list(range(BLOCK_SIZE)))],
                        )
                    ],
                )

            assert await actor.get_kv_event_worker_ids.remote() == list(worker_ids)
            assert set(await actor.get_kv_cached_blocks.remote(worker_ids[0])) == {100}
            assert set(await actor.get_kv_cached_blocks.remote(worker_ids[1])) == {101}

            await actor.remove_worker.remote(worker_ids[0])
            assert await actor.get_kv_event_worker_ids.remote() == [worker_ids[1]]
            assert await actor.get_kv_cached_blocks.remote(worker_ids[0]) == {}
        finally:
            for bridge in bridges:
                await bridge.close()
            for engine_pub in engine_pubs:
                engine_pub.shutdown()

    @pytest.mark.asyncio
    async def test_in_order_application_under_flood(self, ray_instance):
        """Store/remove churn on the same hashes lands in publish order: the
        bridge keeps exactly one actor call in flight."""
        actor = LocalKVRouterActor.remote()
        port = next(_test_ports)
        engine_pub = ZmqEventPublisher(
            data_parallel_rank=0, endpoint=f"tcp://*:{port}", topic=""
        )
        bridge = make_bridge(actor, port)
        try:
            batches = []
            for round_idx in range(100):
                batches.append(
                    KVEventBatch(
                        ts=float(round_idx),
                        events=[stored([round_idx], list(range(BLOCK_SIZE)))],
                    )
                )
                # Remove every block except the final round's.
                if round_idx < 99:
                    batches.append(
                        KVEventBatch(
                            ts=float(round_idx),
                            events=[
                                BlockRemoved(block_hashes=[round_idx], medium="GPU")
                            ],
                        )
                    )
            await publish_and_wait(actor, engine_pub, batches)

            blocks = await actor.get_kv_cached_blocks.remote(WORKER_ID)
            assert set(blocks) == {99}
            counts = (await actor.get_kv_event_counts.remote())[WORKER_ID]
            assert counts == {"block_stored": 100, "block_removed": 99}
        finally:
            await bridge.close()
            engine_pub.shutdown()


class TestFrameDecoding:
    """Frame-level behavior of the bridge, exercised without sockets."""

    @staticmethod
    def frames(batch: KVEventBatch, seq: int) -> List[bytes]:
        payload = msgspec.msgpack.Encoder().encode(batch)
        return [b"", seq.to_bytes(8, "big"), payload]

    @contextlib.asynccontextmanager
    async def bridge(self):
        actor = LocalKVRouterActor.remote()
        bridge = make_bridge(actor, next(_test_ports))
        try:
            yield bridge
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_invalid_frames_are_skipped(self, ray_instance):
        async with self.bridge() as bridge:
            assert bridge._decode_frames([b"only-two", b"frames"]) == []
            assert bridge._decode_frames([b"", b"short-seq", b"payload"]) == []

    @pytest.mark.asyncio
    async def test_event_ids_and_dp_rank(self, ray_instance):
        batch = KVEventBatch(
            ts=1.0,
            events=[stored([1], list(range(BLOCK_SIZE))), AllBlocksCleared()],
            data_parallel_rank=2,
        )
        async with self.bridge() as bridge:
            router_events = bridge._decode_frames(self.frames(batch, seq=0))

        assert [e["event_id"] for e in router_events] == [0, 1]
        assert all(e["worker_id"] == WORKER_ID for e in router_events)
        assert all(e["dp_rank"] == 2 for e in router_events)
        assert router_events[0]["event"] == {
            "type": "block_stored",
            "block_hashes": [1],
            "parent_block_hash": None,
            "token_ids": list(range(BLOCK_SIZE)),
            "block_size": BLOCK_SIZE,
            "medium": "GPU",
            "lora_name": None,
        }
        assert router_events[1]["event"] == {"type": "all_blocks_cleared"}

    @pytest.mark.asyncio
    async def test_sequence_gap_still_applies_events(self, ray_instance):
        """A lost engine batch is logged but later events still apply."""
        first = KVEventBatch(ts=1.0, events=[stored([1], list(range(BLOCK_SIZE)))])
        third = KVEventBatch(ts=3.0, events=[stored([3], list(range(BLOCK_SIZE)))])

        async with self.bridge() as bridge:
            assert len(bridge._decode_frames(self.frames(first, seq=0))) == 1
            router_events = bridge._decode_frames(self.frames(third, seq=2))
        assert len(router_events) == 1
        # Event ids stay monotonic across the gap.
        assert router_events[0]["event_id"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
