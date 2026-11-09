"""Frontend helpers for submitting and tracking advanced orchestration jobs.

These helpers are framework-independent so the Streamlit page can reuse them
without coupling advanced progress state to the standard workflow pipeline.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from urllib.parse import quote

import requests

from frontend.workflow_ui import ParsedSSEEvent, iter_sse_events

ENABLED_ADVANCED_MODES = frozenset(
    {"codebase_chat", "security_audit", "documentation", "bug_fix"}
)
DISABLED_ADVANCED_MODES = frozenset({"test_generation", "feature_development"})
VALID_JOB_STATUSES = frozenset(
    {"queued", "running", "completed", "failed", "cancelled"}
)
TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})
ADVANCED_EVENT_TYPES = frozenset(
    {"step_start", "step_complete", "step_failed", "job_completed", "job_failed"}
)


class AdvancedOrchestrationApiError(RuntimeError):
    """Safe API error that never includes a response body or submitted input."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AdvancedStepProgress:
    """Minimal state retained for one dynamically named orchestration step."""

    status: str = "pending"
    duration_ms: float | None = None


@dataclass(frozen=True)
class AdvancedProgressEvent:
    """Safe progress summary derived from an SSE event."""

    event_type: str
    step: str | None = None
    duration_ms: float | None = None


@dataclass
class AdvancedProgressState:
    """Advanced job progress without retaining raw event payloads."""

    job_id: str = ""
    status: str = "queued"
    is_terminal: bool = False
    current_step: str | None = None
    steps: dict[str, AdvancedStepProgress] = field(default_factory=dict)
    events: list[AdvancedProgressEvent] = field(default_factory=list)


def build_advanced_orchestration_payload(
    mode: str,
    prompt: str,
    *,
    code_input: str | None = None,
    files_input: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build and validate an API payload for one initially enabled mode.

    The disabled modes are rejected here even though the backend API supports
    them, because their current implementations execute generated tests.
    """
    if not isinstance(mode, str):
        raise ValueError("mode must be text")
    if mode in DISABLED_ADVANCED_MODES:
        raise ValueError(f"Advanced orchestration mode '{mode}' is not enabled")
    if mode not in ENABLED_ADVANCED_MODES:
        raise ValueError("Unsupported advanced orchestration mode")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be blank")

    if code_input is not None and not isinstance(code_input, str):
        raise ValueError("code_input must be text")

    normalized_files: dict[str, str] | None = None
    if files_input is not None:
        if not isinstance(files_input, Mapping):
            raise ValueError("files_input must be a mapping of file names to text")
        normalized_files = {}
        for name, content in files_input.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(content, str):
                raise ValueError("files_input must contain named text files")
            normalized_files[name] = content

    if mode in {"documentation", "bug_fix"} and not (
        code_input and code_input.strip()
    ):
        raise ValueError(f"code_input is required for {mode} mode")

    if mode == "security_audit":
        has_code = bool(code_input and code_input.strip())
        has_files = bool(
            normalized_files
            and any(content.strip() for content in normalized_files.values())
        )
        if not has_code and not has_files:
            raise ValueError(
                "code_input or non-empty files_input is required for security_audit mode"
            )

    payload: dict[str, Any] = {"mode": mode, "prompt": prompt.strip()}
    if code_input and code_input.strip():
        payload["code_input"] = code_input
    if normalized_files:
        payload["files_input"] = normalized_files
    return payload


def _decode_json_object(response: Any, *, operation: str) -> dict[str, Any]:
    try:
        data = response.json()
    except (ValueError, TypeError, requests.RequestException) as exc:
        raise AdvancedOrchestrationApiError(
            f"Advanced orchestration {operation} returned invalid JSON."
        ) from None
    if not isinstance(data, dict):
        raise AdvancedOrchestrationApiError(
            f"Advanced orchestration {operation} returned an invalid response."
        )
    return data


def _request_json(
    method: str,
    url: str,
    *,
    expected_status: int,
    operation: str,
    timeout: float,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.Timeout:
        raise AdvancedOrchestrationApiError(
            f"Advanced orchestration {operation} timed out."
        ) from None
    except requests.RequestException:
        raise AdvancedOrchestrationApiError(
            f"Advanced orchestration {operation} could not reach the backend."
        ) from None

    if response.status_code == 404:
        raise AdvancedOrchestrationApiError(
            f"Advanced orchestration {operation} was not found.", status_code=404
        )
    if response.status_code != expected_status:
        raise AdvancedOrchestrationApiError(
            f"Advanced orchestration {operation} failed (HTTP {response.status_code}).",
            status_code=response.status_code,
        )
    return _decode_json_object(response, operation=operation)


def submit_advanced_orchestration_job(
    base_url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float = 15.0,
) -> dict[str, str]:
    """Submit a validated advanced job and return its queued job reference."""
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    validated = build_advanced_orchestration_payload(
        payload.get("mode"),
        payload.get("prompt"),
        code_input=payload.get("code_input"),
        files_input=payload.get("files_input"),
    )
    data = _request_json(
        "POST",
        f"{base_url.rstrip('/')}/advanced-orchestration/jobs",
        expected_status=202,
        operation="job submission",
        timeout=timeout,
        json=validated,
    )
    job_id = data.get("job_id")
    status = data.get("status")
    if not isinstance(job_id, str) or not job_id.strip() or status != "queued":
        raise AdvancedOrchestrationApiError(
            "Advanced orchestration submission returned an invalid job reference."
        )
    return {"job_id": job_id, "status": status}


def get_advanced_orchestration_job_status(
    base_url: str,
    job_id: str,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Fetch and validate lifecycle status without conflating result success."""
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must not be blank")
    safe_job_id = quote(job_id.strip(), safe="")
    data = _request_json(
        "GET",
        f"{base_url.rstrip('/')}/advanced-orchestration/jobs/{safe_job_id}",
        expected_status=200,
        operation="job status retrieval",
        timeout=timeout,
    )
    if data.get("job_id") != job_id.strip():
        raise AdvancedOrchestrationApiError(
            "Advanced orchestration status returned an invalid job ID."
        )
    if data.get("status") not in VALID_JOB_STATUSES:
        raise AdvancedOrchestrationApiError(
            "Advanced orchestration status returned an invalid lifecycle state."
        )
    result = data.get("result")
    if result is not None and not isinstance(result, dict):
        raise AdvancedOrchestrationApiError(
            "Advanced orchestration status returned an invalid result."
        )
    return data


def map_advanced_orchestration_event(
    event: ParsedSSEEvent,
    state: AdvancedProgressState,
) -> AdvancedProgressEvent | None:
    """Apply one supported event while retaining only safe progress fields."""
    event_type = event.event_type
    if event_type not in ADVANCED_EVENT_TYPES:
        return None

    # The shared SSE parser preserves malformed JSON as a ``raw`` payload so
    # other workflows can handle it gracefully. Never let that fallback (or a
    # mismatched envelope) advance the advanced job lifecycle to terminal.
    if event_type in {"job_completed", "job_failed"}:
        payload = event.data.get("payload")
        event_job_id = event.data.get("job_id")
        tracked_job_id = state.job_id
        if (
            event.data.get("event_type") != event_type
            or not isinstance(payload, Mapping)
            or not isinstance(event_job_id, str)
            or not event_job_id.strip()
            or not isinstance(tracked_job_id, str)
            or not tracked_job_id.strip()
            or event_job_id.strip() != tracked_job_id.strip()
        ):
            return None

    raw_payload = event.data.get("payload")
    payload = raw_payload if isinstance(raw_payload, Mapping) else {}
    step: str | None = None
    duration_ms: float | None = None

    if event_type.startswith("step_"):
        raw_step = payload.get("step")
        if not isinstance(raw_step, str) or not raw_step.strip():
            return None
        step = raw_step.strip()[:100]
        progress = state.steps.setdefault(step, AdvancedStepProgress())
        if event_type == "step_start":
            progress.status = "running"
            state.current_step = step
            if state.status == "queued":
                state.status = "running"
        elif event_type == "step_complete":
            progress.status = "completed"
            duration_ms = _valid_duration(payload.get("duration_ms"))
            progress.duration_ms = duration_ms
        else:
            progress.status = "failed"
            state.current_step = step
            if state.status == "queued":
                state.status = "running"
    elif event_type == "job_completed":
        state.status = "completed"
        state.is_terminal = True
    else:
        state.status = "failed"
        state.is_terminal = True

    summary = AdvancedProgressEvent(
        event_type=event_type,
        step=step,
        duration_ms=duration_ms,
    )
    state.events.append(summary)
    return summary


def _valid_duration(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


def format_advanced_progress_event(event: AdvancedProgressEvent) -> str:
    """Render a summary using only the already-filtered progress fields."""
    if event.event_type == "step_start":
        return f"{event.step} started"
    if event.event_type == "step_complete":
        text = f"{event.step} completed"
        if event.duration_ms is not None:
            text += f" ({event.duration_ms:.1f} ms)"
        return text
    if event.event_type == "step_failed":
        return f"{event.step} failed"
    if event.event_type == "job_completed":
        return "Advanced orchestration job completed"
    if event.event_type == "job_failed":
        return "Advanced orchestration job failed"
    return "Advanced orchestration progress updated"


def iter_advanced_orchestration_progress(
    lines: Iterable[str | bytes],
    state: AdvancedProgressState,
) -> Iterator[tuple[AdvancedProgressEvent, str]]:
    """Parse the shared SSE format and yield safe event summaries."""
    for event in iter_sse_events(lines):
        summary = map_advanced_orchestration_event(event, state)
        if summary is not None:
            yield summary, format_advanced_progress_event(summary)
            if state.is_terminal:
                break


def poll_advanced_orchestration_job(
    base_url: str,
    job_id: str,
    *,
    max_attempts: int = 1,
    poll_interval_seconds: float = 1.5,
    timeout: float = 10.0,
    on_status: Callable[[dict[str, Any]], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll status a caller-controlled finite number of times after SSE loss.

    The default performs one request and never retries implicitly. Each
    additional request requires an explicit ``max_attempts`` value.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if (
        isinstance(poll_interval_seconds, bool)
        or not isinstance(poll_interval_seconds, (int, float))
        or not math.isfinite(poll_interval_seconds)
        or poll_interval_seconds < 0
    ):
        raise ValueError("poll_interval_seconds must be a finite non-negative number")

    status: dict[str, Any] | None = None
    for attempt in range(max_attempts):
        status = get_advanced_orchestration_job_status(
            base_url, job_id, timeout=timeout
        )
        if on_status is not None:
            on_status(status)
        if status["status"] in TERMINAL_JOB_STATUSES or attempt + 1 == max_attempts:
            return status
        sleep(poll_interval_seconds)

    # max_attempts validation makes this unreachable; keep the return explicit
    # for static type checkers and defensive callers.
    assert status is not None
    return status
