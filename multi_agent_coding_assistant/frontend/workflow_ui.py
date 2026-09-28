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
import os
import re
import subprocess
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

STAGE_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("job_queued", "Job Queued"),
    ("workflow_started", "Workflow Started"),
    ("planner", "Planner"),
    ("coder", "Coder"),
    ("test_generator", "Test Generator"),
    ("executor", "Application Executor"),
    ("test_executor", "Test Executor"),
    ("reviewer", "Reviewer"),
    ("debugger", "Debugger"),
    ("documentation", "Documentation"),
    ("workflow_terminal", "Workflow Completed"),
)

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
    start_time: float | None = None
    elapsed_seconds: float = 0.0


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


def sanitize_display_text(text: str) -> str:
    """Ensure no tokens, keys, credentials, or internal tracebacks leak to the UI."""
    if not text:
        return ""
    cleaned = re.sub(
        r"(approval_token|ticket_token|token|secret|key|credential|api_key|password)\s*[:=]\s*['\"][^'\"]+['\"]",
        r"\1=[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(Bearer\s+)[A-Za-z0-9_\-\.]+",
        r"\1[REDACTED]",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(AIza[0-9A-Za-z-_]{35})",
        r"[REDACTED_API_KEY]",
        cleaned,
    )
    cleaned = re.sub(
        r'File ".*?", line \d+, in .*',
        "",
        cleaned,
    )
    return cleaned.strip()


def sanitize_error_message(error: str) -> str:
    """Sanitize error messages to ensure no tokens, keys, or stack traces leak."""
    if not error:
        return ""
    cleaned = sanitize_display_text(error)
    first_line = cleaned.strip().split("\n")[0]
    return first_line.strip() or "Workflow encountered an error"


def map_workflow_event(
    event_type: str,
    payload: dict[str, Any],
    state: WorkflowProgressState,
) -> str:
    """Map an incoming backend event into human-readable text and update progress state."""
    state.events.append({"event_type": event_type, "payload": payload, "time": time.time()})

    if state.start_time is None and event_type in ("workflow_started", "step_start"):
        state.start_time = time.monotonic()

    if state.start_time is not None:
        state.elapsed_seconds = round(time.monotonic() - state.start_time, 1)

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


def get_timeline_stages(state: WorkflowProgressState) -> list[dict[str, Any]]:
    """Return a visual timeline representation of recognizable workflow stages."""
    stages: list[dict[str, Any]] = []

    # 1. Job Queued
    queued_status = "completed" if (state.job_id or state.status != "pending") else "pending"
    stages.append({
        "id": "job_queued",
        "label": "Job Queued",
        "status": queued_status,
        "detail": f"ID: {state.job_id[:12]}..." if len(state.job_id) > 12 else (state.job_id or ""),
    })

    # 2. Workflow Started
    if state.status in ("running", "completed", "failed"):
        started_status = "completed"
    elif state.status == "queued":
        started_status = "pending"
    else:
        started_status = "pending"
    stages.append({
        "id": "workflow_started",
        "label": "Workflow Started",
        "status": started_status,
        "detail": f"Iteration {state.iteration}/{state.max_iterations}" if started_status == "completed" else "",
    })

    # Pipeline steps
    for step_id in PIPELINE_STEPS:
        st_val = state.step_states.get(step_id, "pending")
        label = STEP_LABELS.get(step_id, step_id.title())
        detail = ""
        if step_id == "test_executor" and state.tests_passed is not None:
            detail = f"{state.tests_passed} passed / {state.tests_failed or 0} failed"
        elif step_id == "reviewer" and state.reviewer_score is not None:
            detail = f"Score: {state.reviewer_score}/100"
        elif step_id == "debugger" and st_val == "pending" and state.status == "completed":
            st_val = "skipped"
            detail = "No defects detected"

        stages.append({
            "id": step_id,
            "label": label,
            "status": st_val,
            "detail": detail,
        })

    # Final stage
    if state.status == "completed":
        final_status = "completed"
        final_label = "Workflow Completed"
    elif state.status == "failed":
        final_status = "failed"
        final_label = f"Workflow Failed: {state.error or 'Execution Error'}"
    else:
        final_status = "pending"
        final_label = "Workflow Completion"

    stages.append({
        "id": "workflow_terminal",
        "label": final_label,
        "status": final_status,
        "detail": "",
    })

    return stages


def get_repository_info(repo_path: str | None) -> dict[str, Any]:
    """Inspect local repository path for existence and git branch information."""
    if not repo_path or not str(repo_path).strip():
        return {
            "valid": False,
            "exists": False,
            "is_dir": False,
            "is_git": False,
            "branch": None,
            "name": "",
            "error": "Repository path not specified",
        }

    clean_path = os.path.abspath(str(repo_path).strip())
    if not os.path.exists(clean_path):
        return {
            "valid": False,
            "exists": False,
            "is_dir": False,
            "is_git": False,
            "branch": None,
            "name": os.path.basename(clean_path),
            "error": f"Path does not exist: {clean_path}",
        }

    if not os.path.isdir(clean_path):
        return {
            "valid": False,
            "exists": True,
            "is_dir": False,
            "is_git": False,
            "branch": None,
            "name": os.path.basename(clean_path),
            "error": "Path is not a directory",
        }

    git_dir = os.path.join(clean_path, ".git")
    is_git = os.path.exists(git_dir)
    branch = None

    if is_git:
        try:
            head_file = os.path.join(git_dir, "HEAD") if os.path.isdir(git_dir) else None
            if head_file and os.path.exists(head_file):
                with open(head_file, "r", encoding="utf-8") as f:
                    head_content = f.read().strip()
                if head_content.startswith("ref: refs/heads/"):
                    branch = head_content[len("ref: refs/heads/"):]
                elif len(head_content) >= 7:
                    branch = head_content[:7]
        except Exception:
            branch = None

        if not branch:
            try:
                proc = subprocess.run(
                    ["git", "-C", clean_path, "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                if proc.returncode == 0:
                    branch = proc.stdout.strip()
            except Exception:
                pass

    return {
        "valid": True,
        "exists": True,
        "is_dir": True,
        "is_git": is_git,
        "branch": branch,
        "name": os.path.basename(clean_path),
        "error": None,
    }


def check_backend_health(base_url: str = "http://localhost:8000", timeout: float = 0.5) -> dict[str, Any]:
    """Check connectivity and health of the assistant backend API."""
    url = f"{base_url.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                data = {"status": resp.text.strip()}
            return {"connected": True, "status": data.get("status", "ok"), "url": base_url, "error": None}
        return {
            "connected": False,
            "status": f"HTTP {resp.status_code}",
            "url": base_url,
            "error": f"Server returned status {resp.status_code}",
        }
    except Exception as exc:
        return {
            "connected": False,
            "status": "offline",
            "url": base_url,
            "error": sanitize_error_message(str(exc)),
        }


def format_elapsed_time(seconds: float | None) -> str:
    """Format elapsed seconds into a readable string."""
    if seconds is None or seconds < 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes}m {rem_seconds:.1f}s"


def summarize_prompt(prompt: str, max_chars: int = 60) -> str:
    """Return a single-line concise summary of a prompt."""
    if not prompt:
        return ""
    clean = re.sub(r"\s+", " ", prompt).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def record_recent_run(
    runs: list[dict[str, Any]],
    run_data: dict[str, Any],
    max_runs: int = 10,
) -> list[dict[str, Any]]:
    """Record or update a workflow run in the recent runs session history."""
    if not isinstance(run_data, dict):
        return runs

    job_id = run_data.get("job_id")
    updated = False
    new_runs: list[dict[str, Any]] = []

    for r in runs:
        if job_id and r.get("job_id") == job_id:
            merged = {**r, **run_data}
            new_runs.append(merged)
            updated = True
        else:
            new_runs.append(r)

    if not updated:
        new_runs.insert(0, run_data)

    return new_runs[:max_runs]


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
