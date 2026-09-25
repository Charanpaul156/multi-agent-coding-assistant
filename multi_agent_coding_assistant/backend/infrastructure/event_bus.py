"""In-memory pub/sub event bus for asynchronous workflow streaming.

Decoupled from FastAPI, SSE, and HTTP concerns. Provides bounded-queue backpressure,
independent multiple subscribers per job_id, clean termination signaling, and leak-free
cleanup. Satisfies the application-layer EventEmitter protocol.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from typing import AsyncIterator

from backend.domain.job_models import WorkflowEvent

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_SIZE = 100
_TERMINATION_SENTINEL = object()


class EventSubscription:
    """An active subscription handle for a single consumer listening to a job's events."""

    def __init__(self, job_id: str, queue: asyncio.Queue[object], event_bus: EventBus) -> None:
        self.job_id = job_id
        self._queue = queue
        self._event_bus = event_bus
        self._closed = False

    @property
    def is_closed(self) -> bool:
        """Return True if this subscription has been closed."""
        return self._closed

    async def get(self, timeout: float | None = None) -> WorkflowEvent | None:
        """Retrieve the next event, or None if the job is terminated or timeout occurs.

        Raises:
            asyncio.TimeoutError: If timeout is specified and expires before an event arrives.
        """
        if self._closed:
            return None

        if timeout is not None:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        else:
            item = await self._queue.get()

        if item is _TERMINATION_SENTINEL:
            self._closed = True
            return None

        return item  # type: ignore[return-value]

    async def __aiter__(self) -> AsyncIterator[WorkflowEvent]:
        """Asynchronously iterate over incoming events until termination sentinel is received."""
        while not self._closed:
            item = await self._queue.get()
            if item is _TERMINATION_SENTINEL:
                self._closed = True
                break
            yield item  # type: ignore[misc]

    def close(self) -> None:
        """Close this subscription and unregister from the EventBus."""
        if not self._closed:
            self._closed = True
            self._event_bus._unsubscribe(self)

    async def __aenter__(self) -> EventSubscription:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


class EventBus:
    """In-memory pub/sub broker isolating events by job_id.

    Supports both async subscribers/publishers and synchronous workflow publishers
    conforming to the application-level EventEmitter protocol.
    """

    def __init__(self, default_queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        self._default_queue_size = max(1, default_queue_size)
        self._subscribers: dict[str, list[EventSubscription]] = defaultdict(list)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # EventEmitter Protocol Implementation
    # ------------------------------------------------------------------ #

    def emit(self, event: WorkflowEvent) -> None:
        """Synchronous event emission satisfying the EventEmitter protocol."""
        self.publish_sync(event)

    # ------------------------------------------------------------------ #
    # Publishing
    # ------------------------------------------------------------------ #

    def publish_sync(self, event: WorkflowEvent) -> int:
        """Synchronously publish a WorkflowEvent to all active subscribers for event.job_id.

        Non-blocking: if any subscriber queue is full, the oldest event in that
        subscriber's queue is dropped to guarantee publisher progress (backpressure).

        Returns:
            The number of active subscribers that received the event.
        """
        if not isinstance(event, WorkflowEvent):
            raise TypeError(f"event must be a WorkflowEvent, got {type(event).__name__}")

        job_id = event.job_id
        with self._lock:
            subs = list(self._subscribers.get(job_id, []))

        if not subs:
            return 0

        delivered = 0
        for sub in subs:
            if sub.is_closed:
                continue
            self._deliver_item(sub._queue, event, job_id=job_id)
            delivered += 1

        return delivered

    async def publish(self, event: WorkflowEvent) -> int:
        """Asynchronously publish a WorkflowEvent to all active subscribers for event.job_id."""
        return self.publish_sync(event)

    # ------------------------------------------------------------------ #
    # Subscriptions & Lifecycle
    # ------------------------------------------------------------------ #

    def subscribe(self, job_id: str, queue_size: int | None = None) -> EventSubscription:
        """Create and register a new EventSubscription for the specified job_id."""
        size = queue_size if queue_size is not None else self._default_queue_size
        q: asyncio.Queue[object] = asyncio.Queue(maxsize=max(1, size))
        subscription = EventSubscription(job_id=job_id, queue=q, event_bus=self)
        with self._lock:
            self._subscribers[job_id].append(subscription)
        return subscription

    def _unsubscribe(self, subscription: EventSubscription) -> None:
        """Internal helper to unregister a subscription and prune empty buckets."""
        job_id = subscription.job_id
        with self._lock:
            if job_id in self._subscribers:
                self._subscribers[job_id] = [
                    s for s in self._subscribers[job_id] if s is not subscription and not s.is_closed
                ]
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]

    def signal_complete_sync(self, job_id: str) -> int:
        """Synchronously send termination sentinel to all active subscribers for job_id.

        Signals all active consumer streams to terminate gracefully.

        Returns:
            The number of subscribers signaled.
        """
        with self._lock:
            subs = list(self._subscribers.get(job_id, []))

        if not subs:
            return 0

        signaled = 0
        for sub in subs:
            if sub.is_closed:
                continue
            self._deliver_item(sub._queue, _TERMINATION_SENTINEL, job_id=job_id)
            signaled += 1

        return signaled

    async def signal_complete(self, job_id: str) -> int:
        """Send termination sentinel to all active subscribers for job_id."""
        return self.signal_complete_sync(job_id)

    def subscriber_count(self, job_id: str) -> int:
        """Return the count of active subscriptions for a given job_id."""
        with self._lock:
            return len([s for s in self._subscribers.get(job_id, []) if not s.is_closed])

    # ------------------------------------------------------------------ #
    # Internal Delivery Helper
    # ------------------------------------------------------------------ #

    def _deliver_item(self, q: asyncio.Queue[object], item: object, job_id: str = "") -> None:
        """Deliver an item into a queue safely across event loops / threads."""
        loop = getattr(q, "_loop", None)
        if loop is not None and loop.is_running():
            try:
                curr_loop = asyncio.get_running_loop()
            except RuntimeError:
                curr_loop = None

            if curr_loop is loop:
                self._put_with_eviction(q, item, job_id)
            else:
                loop.call_soon_threadsafe(self._put_with_eviction, q, item, job_id)
        else:
            self._put_with_eviction(q, item, job_id)

    @staticmethod
    def _put_with_eviction(q: asyncio.Queue[object], item: object, job_id: str = "") -> None:
        """Insert item into bounded queue, evicting oldest item if full."""
        if q.full():
            try:
                q.get_nowait()
                logger.debug("EventBus queue full for job %s; evicted oldest event", job_id)
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("EventBus queue still full for job %s; dropping event", job_id)
