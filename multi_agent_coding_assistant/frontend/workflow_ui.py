"""Workflow UI helper functions for Streamlit frontend.

Provides testable, framework-independent utilities for:
1. SSE message parsing and streaming
2. Event-to-timeline mapping and progress tracking
3. Status polling fallback
4. Result formatting and sanitization
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

import requests

logger = logging.getLogger(__name__)

# Standard ordered workflow pipeline steps
PIPELINE_STEPS: tuple[str, ...] = (
    "planner",
    "coder",
    "test_generator",
    "executor",
    "test_executor",
    "reviewer",
    "debugger",
    "documentation",
)

STEP_LABELS: dict[str, str] = {
    "planner": "Planner",
    "coder": "Coder",
    "test_generator": "Test Generator",
    "executor": "Application Executor",
    "test_executor": "Test Executor",
    "reviewer": "Reviewer",
    "debugger": "Debugger",
    "documentation": "Documentation",
}

STATUS_ICONS: dict[str, str] = {
    "pending": "⏳",
    "running": "🔄",
    "completed": "✅",
    "failed": "❌",
    "skipped": "➖",
}

FORBIDDEN_PATTERNS = (
    "approval_token",
    "ticket_token",
    "token",
    "secret",
    "credential",
    "GEMINI_API_KEY",
    "chain-of-thought",
)


@dataclass
class ParsedSSEEvent:
    """A single parsed Server-Sent Event message."""

    event_type: str
    data: dict[str, Any]
    raw: str = ""


@dataclass
class WorkflowProgressState:
    """Observable state of an asynchronous workflow execution."""

    job_id: str = ""
    status: str = "queued"  # queued, running, completed, failed
    current_step: str = ""
    current_step_label: str = ""
    step_states: dict[str, str] = field(
        default_factory=lambda: {s: "pending" for s in PIPELINE_STEPS}
    )
    iteration: int = 1
    max_iterations: int = 3
    tests_passed: int | None = None
    tests_failed: int | None = None
    reviewer_score: int | None = None
    is_terminal: bool = False
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


def parse_sse_event(block: str) -> ParsedSSEEvent | None:
    """Parse a single SSE event block into a ParsedSSEEvent.

    Handles standard 'event: ...' and 'data: ...' lines.
    Gracefully handles malformed JSON without raising exceptions.
    Ignores comments (lines starting with ':') and blank blocks.
    """
    lines = block.strip().splitlines()
    event_type = ""
    data_lines = []

    for line in lines:
        s_line = line.strip()
        if not s_line or s_line.startswith(":"):
            # Comment or empty line
            continue
        if s_line.startswith("event:"):
            event_type = s_line[len("event:"):].strip()
        elif s_line.startswith("data:"):
            data_lines.append(s_line[len("data:"):].strip())

    if not event_type and not data_lines:
        return None

    data_str = "\n".join(data_lines)
    if data_str:
        try:
            data_obj = json.loads(data_str)
            if not isinstance(data_obj, dict):
                data_obj = {"value": data_obj}
        except Exception:
            data_obj = {"raw": data_str}
    else:
        data_obj = {}

    if not event_type and isinstance(data_obj, dict):
        event_type = str(data_obj.get("event_type", "message"))

    return ParsedSSEEvent(event_type=event_type, data=data_obj, raw=block)


def iter_sse_events(lines_iterator: Iterable[str | bytes]) -> Iterator[ParsedSSEEvent]:
    """Iterate over incoming raw lines and yield ParsedSSEEvent on empty-line delimiters."""
    buffer: list[str] = []

    for raw_line in lines_iterator:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = raw_line

        # SSE message separator is an empty line
        if not line.strip():
            if buffer:
                block = "\n".join(buffer)
                buffer.clear()
                parsed = parse_sse_event(block)
                if parsed is not None:
                    yield parsed
        else:
            buffer.append(line)

    if buffer:
        block = "\n".join(buffer)
        parsed = parse_sse_event(block)
        if parsed is not None:
            yield parsed


def sanitize_error_message(error: str) -> str:
    """Sanitize error messages to ensure no tokens, keys, or stack traces leak."""
    if not error:
        return ""
    # Strip potential token key-value pairs or tokens
    cleaned = re.sub(r"(token|secret|key|credential)=['\"][^'\"]+['\"]", r"\1=[REDACTED]", error, flags=re.IGNORECASE)
    cleaned = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", cleaned, flags=re.IGNORECASE)
    # Strip internal file paths
    cleaned = re.sub(r'File ".*?", line \d+, in .*', "", cleaned)
    # Limit to concise single line
    first_line = cleaned.strip().split("\n")[0]
    return first_line.strip() or "Workflow encountered an error"


def map_workflow_event(
    event_type: str,
    payload: dict[str, Any],
    state: WorkflowProgressState,
) -> str:
    """Map an incoming backend event into human-readable text and update progress state."""
    state.events.append({"event_type": event_type, "payload": payload, "time": time.time()})

    if event_type == "workflow_started":
        state.status = "running"
        return "Workflow started"

    elif event_type == "step_start":
        step = payload.get("step", "")
        state.current_step = step
        label = STEP_LABELS.get(step, step.title())
        state.current_step_label = label
        if step in state.step_states:
            state.step_states[step] = "running"
        return f"{label} running"

    elif event_type == "step_complete":
        step = payload.get("step", "")
        label = STEP_LABELS.get(step, step.title())
        if step in state.step_states:
            state.step_states[step] = "completed"
        if step == "debugger" and payload.get("corrected"):
            return "Debugger completed: fix suggested"
        return f"{label} complete"

    elif event_type == "iteration_progress":
        state.iteration = int(payload.get("iteration", state.iteration))
        state.max_iterations = int(payload.get("max_iterations", state.max_iterations))
        return f"Iteration {state.iteration} / {state.max_iterations}"

    elif event_type == "test_result":
        passed = payload.get("passed")
        failed = payload.get("failed")
        state.tests_passed = int(passed) if passed is not None else 0
        state.tests_failed = int(failed) if failed is not None else 0
        state.step_states["test_executor"] = "completed"
        return f"Tests: {state.tests_passed} passed / {state.tests_failed} failed"

    elif event_type == "review_result":
        score = payload.get("score")
        if score is not None:
            state.reviewer_score = int(score)
            state.step_states["reviewer"] = "completed"
            return f"Reviewer: {state.reviewer_score}/100"
        return "Reviewer completed"

    elif event_type in ("workflow_completed", "job_completed"):
        state.status = "completed"
        state.is_terminal = True
        # Mark active steps as completed
        for s in PIPELINE_STEPS:
            if state.step_states[s] == "running":
                state.step_states[s] = "completed"
        return "Workflow completed"

    elif event_type in ("workflow_failed", "job_failed"):
        state.status = "failed"
        state.is_terminal = True
        raw_err = payload.get("error", "Workflow failed")
        state.error = sanitize_error_message(str(raw_err))
        if state.current_step and state.current_step in state.step_states:
            state.step_states[state.current_step] = "failed"
        return f"Workflow failed: {state.error}"

    return f"Event: {event_type}"


def format_progress_timeline(state: WorkflowProgressState) -> str:
    """Format the current step states into a Markdown checklist timeline."""
    lines = []
    for step in PIPELINE_STEPS:
        st_val = state.step_states.get(step, "pending")
        icon = STATUS_ICONS.get(st_val, "⏳")
        label = STEP_LABELS.get(step, step.title())
        if st_val == "running":
            lines.append(f"{icon} **{label}** *(in progress...)*")
        elif st_val == "completed":
            lines.append(f"{icon} {label}")
        elif st_val == "failed":
            lines.append(f"{icon} **{label}** *(failed)*")
        else:
            lines.append(f"{icon} {label}")
    return "\n\n".join(lines)


def poll_workflow_job(
    base_url: str,
    job_id: str,
    max_wait_seconds: float = 120.0,
    poll_interval: float = 1.5,
    on_poll_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any] | None:
    """Fallback polling for job status when SSE is unavailable or drops.

    Polls GET /workflow/jobs/{job_id} until terminal state or timeout.
    """
    url = f"{base_url}/workflow/jobs/{job_id}"
    start_time = time.monotonic()

    while time.monotonic() - start_time < max_wait_seconds:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if on_poll_callback:
                    on_poll_callback(data)
                status = data.get("status")
                if status in ("completed", "failed", "cancelled"):
                    return data
            elif resp.status_code == 404:
                logger.warning("Polling: job %s not found (404)", job_id)
                return None
        except requests.exceptions.RequestException as exc:
            logger.warning("Polling request exception for job %s: %s", job_id, exc)

        time.sleep(poll_interval)

    logger.warning("Polling timed out after %.1fs for job %s", max_wait_seconds, job_id)
    return None
