"""Unit tests for the in-memory pub/sub EventBus."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from backend.domain.job_models import WorkflowEvent
from backend.infrastructure.event_bus import EventBus


@pytest.fixture
def bus() -> EventBus:
    return EventBus(default_queue_size=10)


def _event(job_id: str, event_type: str = "step_start", payload: dict | None = None) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=event_type,
        job_id=job_id,
        timestamp=datetime.now(timezone.utc),
        payload=payload or {"step": "planner"},
    )


# 1. Publish -> Subscriber receives event
@pytest.mark.anyio
async def test_publish_subscriber_receives_event(bus: EventBus):
    sub = bus.subscribe("job_1")
    ev = _event("job_1", "step_start", {"step": "planner"})

    delivered = await bus.publish(ev)
    assert delivered == 1

    received = await sub.get(timeout=1.0)
    assert received is not None
    assert received.event_type == "step_start"
    assert received.job_id == "job_1"
    assert received.payload == {"step": "planner"}
    sub.close()


# 2. Multiple subscribers each receive the event
@pytest.mark.anyio
async def test_multiple_subscribers_each_receive_event(bus: EventBus):
    sub1 = bus.subscribe("job_multi")
    sub2 = bus.subscribe("job_multi")
    assert bus.subscriber_count("job_multi") == 2

    ev = _event("job_multi", "iteration_progress", {"iteration": 1})
    delivered = await bus.publish(ev)
    assert delivered == 2

    ev1 = await sub1.get(timeout=1.0)
    ev2 = await sub2.get(timeout=1.0)
    assert ev1 == ev
    assert ev2 == ev

    sub1.close()
    sub2.close()


# 3. Events are isolated by job_id
@pytest.mark.anyio
async def test_events_isolated_by_job_id(bus: EventBus):
    sub_a = bus.subscribe("job_A")
    sub_b = bus.subscribe("job_B")

    ev_a = _event("job_A", "step_start", {"step": "a"})
    ev_b = _event("job_B", "step_start", {"step": "b"})

    await bus.publish(ev_a)
    await bus.publish(ev_b)

    rec_a = await sub_a.get(timeout=1.0)
    rec_b = await sub_b.get(timeout=1.0)

    assert rec_a.job_id == "job_A"
    assert rec_b.job_id == "job_B"

    # Ensure no cross-talk
    with pytest.raises(asyncio.TimeoutError):
        await sub_a.get(timeout=0.1)

    sub_a.close()
    sub_b.close()


# 4. Subscriber cleanup works
@pytest.mark.anyio
async def test_subscriber_cleanup_works(bus: EventBus):
    sub = bus.subscribe("job_clean")
    assert bus.subscriber_count("job_clean") == 1

    sub.close()
    assert sub.is_closed is True
    assert bus.subscriber_count("job_clean") == 0

    # Publishing after close should deliver to 0 subscribers
    ev = _event("job_clean")
    delivered = await bus.publish(ev)
    assert delivered == 0

    # Async context manager cleanup
    async with bus.subscribe("job_ctx") as ctx_sub:
        assert bus.subscriber_count("job_ctx") == 1
    assert bus.subscriber_count("job_ctx") == 0


# 5. Bounded queue behavior works (backpressure)
@pytest.mark.anyio
async def test_bounded_queue_backpressure(bus: EventBus):
    # Queue with capacity 2
    sub = bus.subscribe("job_bounded", queue_size=2)

    ev1 = _event("job_bounded", "ev1", {"idx": 1})
    ev2 = _event("job_bounded", "ev2", {"idx": 2})
    ev3 = _event("job_bounded", "ev3", {"idx": 3})

    # Publish 3 events into queue of size 2 (should evict ev1, keeping ev2 and ev3)
    await bus.publish(ev1)
    await bus.publish(ev2)
    await bus.publish(ev3)

    rec1 = await sub.get(timeout=1.0)
    rec2 = await sub.get(timeout=1.0)

    assert rec1.payload == {"idx": 2}
    assert rec2.payload == {"idx": 3}
    sub.close()


# 6. Completion / termination signaling works
@pytest.mark.anyio
async def test_completion_termination_signaling(bus: EventBus):
    sub = bus.subscribe("job_term")
    ev = _event("job_term", "test_result", {"passed": 5})

    await bus.publish(ev)
    signaled = await bus.signal_complete("job_term")
    assert signaled == 1

    # First item is the event
    rec = await sub.get(timeout=1.0)
    assert rec == ev

    # Next item should be None (signaling completion)
    term = await sub.get(timeout=1.0)
    assert term is None
    assert sub.is_closed is True

    # Test clean iteration stop
    sub2 = bus.subscribe("job_iter")
    await bus.publish(_event("job_iter", "e1"))
    await bus.publish(_event("job_iter", "e2"))
    await bus.signal_complete("job_iter")

    collected = []
    async for item in sub2:
        collected.append(item.event_type)

    assert collected == ["e1", "e2"]
    assert sub2.is_closed is True


# 7. Event payload remains structured and serializable
def test_event_payload_structured_serializable():
    ev = WorkflowEvent(
        event_type="iteration_progress",
        job_id="job_ser",
        payload={"iteration": 2, "passed": 10, "failed": 0, "score": 95},
    )
    d = ev.to_dict()
    assert d["event_type"] == "iteration_progress"
    assert d["job_id"] == "job_ser"
    assert "timestamp" in d
    assert d["payload"]["iteration"] == 2
    assert d["payload"]["score"] == 95


# 8. Publisher does not require a subscriber to exist
@pytest.mark.anyio
async def test_publisher_without_subscribers(bus: EventBus):
    ev = _event("job_ghost")
    delivered = await bus.publish(ev)
    assert delivered == 0
