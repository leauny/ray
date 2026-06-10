import logging
from typing import Any, Dict, List

from ray.llm._internal.serve.routing_policies.kv_aware.kv_events import (
    KV_EVENT_ALL_BLOCKS_CLEARED,
    KV_EVENT_BLOCK_REMOVED,
    KV_EVENT_BLOCK_STORED,
)
from ray.serve._private.constants import SERVE_LOGGER_NAME

logger = logging.getLogger(SERVE_LOGGER_NAME)


class KvEventConsumer:
    """Consumes the router events bridged from each replica's KvEventPublisher.

    Maintains each worker's cached blocks (block hash -> token ids). Stands in
    for the consumer inside Dynamo's ``KvRouter``, which applies the same
    events to the global KV indexer used for overlap scoring.
    """

    def __init__(self):
        self._blocks_by_worker: Dict[int, Dict[Any, List[int]]] = {}
        self._event_counts_by_worker: Dict[int, Dict[str, int]] = {}

    def consume(self, router_events: List[Dict[str, Any]]) -> None:
        """Apply a batch of router events in order."""
        for router_event in router_events:
            worker_id = router_event["worker_id"]
            event = router_event["event"]
            event_type = event["type"]

            counts = self._event_counts_by_worker.setdefault(worker_id, {})
            counts[event_type] = counts.get(event_type, 0) + 1

            blocks = self._blocks_by_worker.setdefault(worker_id, {})
            if event_type == KV_EVENT_BLOCK_STORED:
                block_size = event["block_size"]
                token_ids = event["token_ids"]
                for i, block_hash in enumerate(event["block_hashes"]):
                    blocks[block_hash] = token_ids[
                        i * block_size : (i + 1) * block_size
                    ]
            elif event_type == KV_EVENT_BLOCK_REMOVED:
                for block_hash in event["block_hashes"]:
                    blocks.pop(block_hash, None)
            elif event_type == KV_EVENT_ALL_BLOCKS_CLEARED:
                blocks.clear()
            else:
                logger.warning("Ignoring unknown KV event type: %s", event_type)

    def remove_worker(self, worker_id: int) -> None:
        """Drop all state for a worker that left the deployment."""
        self._blocks_by_worker.pop(worker_id, None)
        self._event_counts_by_worker.pop(worker_id, None)

    def get_worker_ids(self) -> List[int]:
        """Workers that have produced at least one KV event, sorted."""
        return sorted(self._event_counts_by_worker)

    def get_cached_blocks(self, worker_id: int) -> Dict[Any, List[int]]:
        """A worker's currently cached blocks, keyed by engine block hash."""
        return dict(self._blocks_by_worker.get(worker_id, {}))

    def get_event_counts(self) -> Dict[int, Dict[str, int]]:
        """Per-worker counts of consumed events by type."""
        return {
            worker_id: dict(counts)
            for worker_id, counts in self._event_counts_by_worker.items()
        }
