"""Deterministic unit tests for Streamlit workflow UI helpers.

Verifies:
1. SSE event parsing
2. Multiple SSE events streaming
3. Malformed JSON handling
4. Event-to-status mapping
5. Iteration progress
6. Test result display data
7. Reviewer result display data
8. Terminal event handling
9. Error event handling
10. No secret/token leakage
11. Timeline checklist formatting
12. Fallback polling functionality with mock
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from frontend.workflow_ui import (
    PIPELINE_STEPS,
    ParsedSSEEvent,
    WorkflowProgressState,
    format_progress_timeline,
    iter_sse_events,
    map_workflow_event,
    parse_sse_event,
    poll_workflow_job,
    sanitize_error_message,
)


# 1. SSE event parsing
def test_parse_sse_event_valid() -> None:
    block = (
        "event: step_start\n"
        'data: {"step": "planner", "detail": "analyzing"}\n\n'
    )
    parsed = parse_sse_event(block)
    assert parsed is not None
    assert parsed.event_type == "step_start"
    assert parsed.data["step"] == "planner"
    assert parsed.data["detail"] == "analyzing"


# 2. Multiple SSE events
def test_iter_sse_events_multiple() -> None:
    lines = [
        "event: workflow_started",
        'data: {"job_id": "job_1"}',
        "",
        "event: step_start",
        'data: {"step": "coder"}',
        "",
        "event: step_complete",
        'data: {"step": "coder", "code_length": 50}',
        "",
    ]
    events = list(iter_sse_events(lines))
    assert len(events) == 3
    assert events[0].event_type == "workflow_started"
    assert events[1].event_type == "step_start"
    assert events[1].data["step"] == "coder"
    assert events[2].event_type == "step_complete"
    assert events[2].data["code_length"] == 50


# 3. Malformed JSON handling
def test_parse_sse_event_malformed_json() -> None:
    block = "event: custom\ndata: {not-valid-json: 123}\n\n"
    parsed = parse_sse_event(block)
    assert parsed is not None
    assert parsed.event_type == "custom"
    # Does not raise, preserves raw data string
    assert "raw" in parsed.data
    assert "{not-valid-json: 123}" in parsed.data["raw"]


def test_parse_sse_event_comments_ignored() -> None:
    block = ": keep-alive\n\n"
    parsed = parse_sse_event(block)
    assert parsed is None


# 4. Event-to-status mapping
def test_map_workflow_event_step_lifecycle() -> None:
    state = WorkflowProgressState(job_id="test_job")

    # Initially all pending
    assert state.step_states["planner"] == "pending"

    # Step start
    msg1 = map_workflow_event("step_start", {"step": "planner"}, state)
    assert "Planner running" in msg1
    assert state.step_states["planner"] == "running"
    assert state.current_step == "planner"

    # Step complete
    msg2 = map_workflow_event("step_complete", {"step": "planner"}, state)
    assert "Planner complete" in msg2
    assert state.step_states["planner"] == "completed"


# 5. Iteration progress
def test_map_workflow_event_iteration_progress() -> None:
    state = WorkflowProgressState(job_id="test_job")
    msg = map_workflow_event("iteration_progress", {"iteration": 2, "max_iterations": 3}, state)
    assert msg == "Iteration 2 / 3"
    assert state.iteration == 2
    assert state.max_iterations == 3


# 6. Test result display data
def test_map_workflow_event_test_result() -> None:
    state = WorkflowProgressState(job_id="test_job")
    msg = map_workflow_event(
        "test_result",
        {"iteration": 1, "success": True, "passed": 27, "failed": 3, "exit_code": 0},
        state,
    )
    assert msg == "Tests: 27 passed / 3 failed"
    assert state.tests_passed == 27
    assert state.tests_failed == 3
    assert state.step_states["test_executor"] == "completed"


# 7. Reviewer result display data
def test_map_workflow_event_reviewer_result() -> None:
    state = WorkflowProgressState(job_id="test_job")
    msg = map_workflow_event("review_result", {"iteration": 1, "score": 98, "has_issues": False}, state)
    assert msg == "Reviewer: 98/100"
    assert state.reviewer_score == 98
    assert state.step_states["reviewer"] == "completed"


# 8. Terminal event handling
def test_map_workflow_event_completion() -> None:
    state = WorkflowProgressState(job_id="test_job")
    state.step_states["documentation"] = "running"

    msg = map_workflow_event("workflow_completed", {"status": "completed"}, state)
    assert msg == "Workflow completed"
    assert state.status == "completed"
    assert state.is_terminal is True
    # Any active running step is marked completed
    assert state.step_states["documentation"] == "completed"


# 9. Error event handling
def test_map_workflow_event_failure() -> None:
    state = WorkflowProgressState(job_id="test_job")
    state.current_step = "coder"
    state.step_states["coder"] = "running"

    msg = map_workflow_event("workflow_failed", {"error": "Quota limit reached on API call"}, state)
    assert "Workflow failed: Quota limit reached on API call" in msg
    assert state.status == "failed"
    assert state.is_terminal is True
    assert state.error == "Quota limit reached on API call"
    assert state.step_states["coder"] == "failed"


# 10. No secret/token leakage
def test_sanitize_error_message_strips_tokens() -> None:
    error_with_tokens = "Failed with token='secret_token_12345' and Bearer secret_pass_9999"
    sanitized = sanitize_error_message(error_with_tokens)
    assert "secret_token_12345" not in sanitized
    assert "secret_pass_9999" not in sanitized
    assert "[REDACTED]" in sanitized

    stack_trace_error = 'Traceback (most recent call last):\n  File "foo.py", line 12, in bar\nValueError: invalid'
    sanitized_trace = sanitize_error_message(stack_trace_error)
    assert 'File "foo.py"' not in sanitized_trace


# 11. Timeline checklist formatting
def test_format_progress_timeline() -> None:
    state = WorkflowProgressState(job_id="test_job")
    state.step_states["planner"] = "completed"
    state.step_states["coder"] = "running"
    state.step_states["test_generator"] = "pending"

    timeline_md = format_progress_timeline(state)
    assert "✅ Planner" in timeline_md
    assert "🔄 **Coder** *(in progress...)*" in timeline_md
    assert "⏳ Test Generator" in timeline_md


# 12. Fallback polling functionality with mock
def test_poll_workflow_job_success() -> None:
    with patch("frontend.workflow_ui.requests.get") as mock_get:
        # First poll: running, second poll: completed
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"job_id": "job_1", "status": "running"}

        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {
            "job_id": "job_1",
            "status": "completed",
            "result": {"success": True},
        }

        mock_get.side_effect = [resp1, resp2]

        callback_data = []
        result = poll_workflow_job(
            "http://localhost:8000",
            "job_1",
            max_wait_seconds=5.0,
            poll_interval=0.01,
            on_poll_callback=lambda d: callback_data.append(d["status"]),
        )

        assert result is not None
        assert result["status"] == "completed"
        assert callback_data == ["running", "completed"]
