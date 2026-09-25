"""Application-level EventEmitter abstraction for workflow progress events.

Decouples application use cases from concrete pub/sub mechanisms (such as EventBus).
Synchronous use cases can emit structured WorkflowEvent instances safely and synchronously
without requiring async/await refactoring.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from backend.domain.job_models import WorkflowEvent


@runtime_checkable
class EventEmitter(Protocol):
    """Protocol for components that can emit WorkflowEvents."""

    def emit(self, event: WorkflowEvent) -> None:
        """Emit a structured workflow event synchronously."""
        ...


class NullEventEmitter:
    """Default no-op event emitter when event streaming is disabled."""

    def emit(self, event: WorkflowEvent) -> None:
        pass


class RecordingEventEmitter:
    """In-memory event emitter for testing and inspection."""

    def __init__(self) -> None:
        self.events: list[WorkflowEvent] = []

    def emit(self, event: WorkflowEvent) -> None:
        if not isinstance(event, WorkflowEvent):
            raise TypeError(f"Expected WorkflowEvent, got {type(event).__name__}")
        self.events.append(event)

    def get_events(self, event_type: str | None = None) -> list[WorkflowEvent]:
        """Return captured events, optionally filtered by event_type."""
        if event_type is not None:
            return [e for e in self.events if e.event_type == event_type]
        return list(self.events)

    def clear(self) -> None:
        self.events.clear()


class ScopedEventEmitter:
    """An EventEmitter bound to a specific job_id.

    Ensures that any event emitted through it is tagged with the bound job_id,
    and provides convenience helper methods for emitting events.
    """

    def __init__(self, emitter: EventEmitter, job_id: str) -> None:
        self._emitter = emitter
        self._job_id = job_id

    @property
    def job_id(self) -> str:
        return self._job_id

    def emit(self, event: WorkflowEvent) -> None:
        """Emit a WorkflowEvent, ensuring it carries the bound job_id."""
        if event.job_id != self._job_id:
            event = WorkflowEvent(
                event_type=event.event_type,
                job_id=self._job_id,
                timestamp=event.timestamp,
                payload=event.payload,
            )
        self._emitter.emit(event)

    def emit_event(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Construct and emit a WorkflowEvent bound to self.job_id."""
        event = WorkflowEvent(
            event_type=event_type,
            job_id=self._job_id,
            payload=dict(payload or {}),
        )
        self._emitter.emit(event)
