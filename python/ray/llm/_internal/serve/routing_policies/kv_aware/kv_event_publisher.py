import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import msgspec
import zmq
import zmq.asyncio
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)

from ray import serve
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (
    KV_ROUTER_ACTOR_NAME,
    get_worker_id,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_events import (
    KV_EVENT_ALL_BLOCKS_CLEARED,
    KV_EVENT_BLOCK_REMOVED,
    KV_EVENT_BLOCK_STORED,
    resolve_kv_event_source_endpoint,
)
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve.exceptions import RayServeException

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from ray.llm._internal.serve.core.configs.llm_config import LLMConfig

logger = logging.getLogger(SERVE_LOGGER_NAME)


class KvEventPublisher:
    """Bridges the engine's KV-cache events into the KV router actor.

    Mirrors ``dynamo.llm.KvEventPublisher``: subscribes to the engine's (or,
    with KVBM, the consolidator's) ZMQ KV-event stream and republishes each
    event as a normalized router event carrying the worker's identity and a
    monotonically increasing event id. Publishing to the deployment's
    ``KVRouterActor`` stands in for Dynamo's ``kv-events`` event plane.

    The engine publishes events only while serving requests, so subscribing
    during replica startup (before the replica receives traffic) observes the
    full stream. Awaiting each actor call before consuming the next message
    keeps this replica's events ordered.

    Must be constructed in a running asyncio event loop.
    """

    def __init__(
        self,
        kv_router_actor: "ActorHandle",
        worker_id: int,
        kv_block_size: int,
        zmq_endpoint: str,
        zmq_topic: str = "",
        dp_rank: int = 0,
    ):
        self._kv_router_actor = kv_router_actor
        self._worker_id = worker_id
        self._kv_block_size = kv_block_size
        self._zmq_endpoint = zmq_endpoint
        self._dp_rank = dp_rank
        self._next_event_id = 0
        self._last_engine_seq: Optional[int] = None
        self._decoder = msgspec.msgpack.Decoder(KVEventBatch)

        self._socket = zmq.asyncio.Context.instance().socket(zmq.SUB)
        self._socket.connect(zmq_endpoint)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, zmq_topic)
        self._consume_task = asyncio.create_task(self._consume_loop())
        logger.info(
            "KvEventPublisher for worker %d consuming KV events from %s.",
            worker_id,
            zmq_endpoint,
        )

    async def _consume_loop(self) -> None:
        while True:
            frames = await self._socket.recv_multipart()
            try:
                router_events = self._decode_frames(frames)
            except Exception:
                logger.warning(
                    "Failed to decode KV event message from %s.",
                    self._zmq_endpoint,
                    exc_info=True,
                )
                continue
            if not router_events:
                continue
            try:
                await self._kv_router_actor.on_kv_events.remote(router_events)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Failed to publish %d KV event(s) to the KV router actor.",
                    len(router_events),
                    exc_info=True,
                )

    def _decode_frames(self, frames: List[bytes]) -> List[Dict[str, Any]]:
        if len(frames) != 3:
            logger.warning(
                "Unexpected KV event frame count: expected 3, got %d.", len(frames)
            )
            return []
        _topic, seq_bytes, payload = frames
        if len(seq_bytes) != 8:
            logger.warning(
                "Invalid KV event sequence length: expected 8 bytes, got %d.",
                len(seq_bytes),
            )
            return []
        engine_seq = int.from_bytes(seq_bytes, "big")
        if self._last_engine_seq is not None and engine_seq != (
            self._last_engine_seq + 1
        ):
            logger.warning(
                "KV event sequence gap from %s: expected %d, got %d; "
                "events were lost.",
                self._zmq_endpoint,
                self._last_engine_seq + 1,
                engine_seq,
            )
        self._last_engine_seq = engine_seq

        batch = self._decoder.decode(payload)
        dp_rank = (
            batch.data_parallel_rank
            if batch.data_parallel_rank is not None
            else self._dp_rank
        )
        router_events = []
        for event in batch.events:
            payload_dict = self._normalize_event(event)
            if payload_dict is None:
                continue
            router_events.append(
                {
                    "worker_id": self._worker_id,
                    "event_id": self._next_event_id,
                    "dp_rank": dp_rank,
                    "event": payload_dict,
                }
            )
            self._next_event_id += 1
        return router_events

    def _normalize_event(self, event: Any) -> Optional[Dict[str, Any]]:
        if isinstance(event, BlockStored):
            if event.block_size != self._kv_block_size:
                logger.warning(
                    "KV event block size %d does not match the engine's "
                    "configured block size %d.",
                    event.block_size,
                    self._kv_block_size,
                )
            return {
                "type": KV_EVENT_BLOCK_STORED,
                "block_hashes": list(event.block_hashes),
                "parent_block_hash": event.parent_block_hash,
                "token_ids": list(event.token_ids),
                "block_size": event.block_size,
                "medium": event.medium,
                "lora_name": event.lora_name,
            }
        if isinstance(event, BlockRemoved):
            return {
                "type": KV_EVENT_BLOCK_REMOVED,
                "block_hashes": list(event.block_hashes),
                "medium": event.medium,
            }
        if isinstance(event, AllBlocksCleared):
            return {"type": KV_EVENT_ALL_BLOCKS_CLEARED}
        logger.warning("Ignoring unknown engine KV event: %r", type(event))
        return None

    async def close(self) -> None:
        """Stop consuming and release the ZMQ socket."""
        self._consume_task.cancel()
        try:
            await self._consume_task
        except asyncio.CancelledError:
            pass
        self._socket.close(linger=0)


def maybe_create_kv_event_publisher(
    llm_config: "LLMConfig", kv_block_size: int
) -> Optional[KvEventPublisher]:
    """Create this replica's KvEventPublisher if KV-aware routing is set up.

    Requires KV-cache events enabled in ``engine_kwargs`` and the
    deployment-scoped ``KVRouterActor`` (attached when the deployment routes
    with ``KVAwareRouter``); returns ``None`` otherwise.
    """
    zmq_endpoint = resolve_kv_event_source_endpoint(llm_config)
    if zmq_endpoint is None:
        return None
    try:
        kv_router_actor = serve.get_deployment_actor(KV_ROUTER_ACTOR_NAME)
        replica_context = serve.get_replica_context()
    except (RayServeException, ValueError):
        # Outside a replica, or the actor is not attached to this deployment
        # (KV events were enabled for an external consumer).
        logger.info(
            "KV-cache events are enabled but no %s deployment actor is "
            "reachable; not bridging KV events.",
            KV_ROUTER_ACTOR_NAME,
        )
        return None
    return KvEventPublisher(
        kv_router_actor=kv_router_actor,
        worker_id=get_worker_id(replica_context.replica_id.unique_id),
        kv_block_size=kv_block_size,
        zmq_endpoint=zmq_endpoint,
        dp_rank=llm_config.engine_kwargs.get("data_parallel_rank") or 0,
    )
