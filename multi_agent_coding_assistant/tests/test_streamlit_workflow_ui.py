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
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from frontend.workflow_ui import (
    PIPELINE_STEPS,
    ParsedSSEEvent,
    WorkflowProgressState,
    check_backend_health,
    format_elapsed_time,
    format_progress_timeline,
    get_repository_info,
    get_timeline_stages,
    iter_sse_events,
    map_workflow_event,
    parse_sse_event,
    poll_workflow_job,
    record_recent_run,
    sanitize_display_text,
    sanitize_error_message,
    summarize_prompt,
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


APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "app.py"))


# 13. App imports successfully
def test_app_imports_successfully() -> None:
    import frontend.app
    import frontend.workflow_ui
    assert frontend.app.BACKEND_URL == "http://localhost:8000"
    assert len(frontend.workflow_ui.PIPELINE_STEPS) == 8


# 14. AppTest initial render without exceptions
def test_apptest_initial_render_without_exceptions() -> None:
    at = AppTest.from_file(APP_PATH).run()
    assert len(at.exception) == 0
    # Verifies the 4 navigation tabs render
    assert len(at.tabs) == 4
    # Verifies prompt input area exists
    assert at.text_area(key="main_workflow_prompt_input") is not None
    # Verifies run buttons exist
    assert at.button(key="btn_run_live_stream") is not None
    assert at.button(key="btn_run_standard") is not None


# 15. AppTest prompt validation for empty input
def test_apptest_prompt_validation_empty() -> None:
    at = AppTest.from_file(APP_PATH).run()
    # Click run button with empty prompt
    at.button(key="btn_run_live_stream").click().run()
    assert len(at.exception) == 0
    assert len(at.warning) > 0
    assert "Please enter a coding prompt" in at.warning[0].value


# 16. AppTest example prompt preset selection
def test_apptest_example_prompt_selection() -> None:
    at = AppTest.from_file(APP_PATH).run()
    at.button(key="btn_example_calc").click().run()
    assert len(at.exception) == 0
    prompt_val = at.text_area(key="main_workflow_prompt_input").value
    assert "calculator" in prompt_val.lower()


# 17. AppTest final result rendering
def test_apptest_final_result_rendering() -> None:
    at = AppTest.from_file(APP_PATH).run()
    at.session_state.async_workflow_result = {
        "workflow_status": "completed",
        "success": True,
        "execution_time_ms": 1500.0,
        "generated_code": "def solve():\n    return 42\n",
        "generated_tests": "def test_solve():\n    assert solve() == 42\n",
        "planning": {
            "problem_summary": "Solve problem",
            "project_type": "Module",
            "requirements": ["requirement 1"],
            "modules": ["main"],
        },
        "test_execution": {
            "passed": 5,
            "failed": 0,
            "exit_code": 0,
            "stdout": "5 passed",
        },
        "review": {
            "overall_score": 95,
            "strengths": ["Well structured"],
            "recommendations": ["Add type hints"],
            "final_summary": "High quality implementation",
        },
        "documentation": {
            "markdown_documentation": "# Module Documentation\nProvides solve() function.",
        },
    }
    at.run()
    assert len(at.exception) == 0
    # Verify metrics rendered
    metric_labels = [m.label for m in at.metric]
    assert "Workflow Status" in metric_labels
    assert "Tests Passed" in metric_labels
    assert "Reviewer Score" in metric_labels


# 18. Repository info helper
def test_get_repository_info_valid() -> None:
    cwd = os.getcwd()
    info = get_repository_info(cwd)
    assert info["valid"] is True
    assert info["exists"] is True
    assert info["is_dir"] is True
    assert isinstance(info["is_git"], bool)


def test_get_repository_info_invalid() -> None:
    info = get_repository_info("/non/existent/path/xyz")
    assert info["valid"] is False
    assert info["exists"] is False
    assert info["error"] is not None

    empty_info = get_repository_info("")
    assert empty_info["valid"] is False


# 19. Backend health check helper
def test_check_backend_health_online() -> None:
    with patch("frontend.workflow_ui.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_resp

        health = check_backend_health("http://localhost:8000")
        assert health["connected"] is True
        assert health["status"] == "ok"


def test_check_backend_health_offline() -> None:
    with patch("frontend.workflow_ui.requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        health = check_backend_health("http://localhost:8000")
        assert health["connected"] is False
        assert health["status"] == "offline"


# 20. Elapsed time formatter
def test_format_elapsed_time() -> None:
    assert format_elapsed_time(None) == "0.0s"
    assert format_elapsed_time(-5.0) == "0.0s"
    assert format_elapsed_time(12.34) == "12.3s"
    assert format_elapsed_time(75.0) == "1m 15.0s"


# 21. Prompt summarizer
def test_summarize_prompt() -> None:
    assert summarize_prompt("") == ""
    assert summarize_prompt("   Hello    world   ") == "Hello world"
    short = "Build a calculator"
    assert summarize_prompt(short, max_chars=30) == short
    long_prompt = "A" * 100
    summary = summarize_prompt(long_prompt, max_chars=20)
    assert len(summary) <= 20
    assert summary.endswith("...")


# 22. Recent runs session recorder
def test_record_recent_run() -> None:
    runs = []
    run1 = {"job_id": "job_1", "status": "Running", "prompt_summary": "Task 1"}
    runs = record_recent_run(runs, run1, max_runs=3)
    assert len(runs) == 1
    assert runs[0]["job_id"] == "job_1"

    # Update existing run
    run1_update = {"job_id": "job_1", "status": "Completed"}
    runs = record_recent_run(runs, run1_update, max_runs=3)
    assert len(runs) == 1
    assert runs[0]["status"] == "Completed"

    # Add up to max_runs
    runs = record_recent_run(runs, {"job_id": "job_2"}, max_runs=3)
    runs = record_recent_run(runs, {"job_id": "job_3"}, max_runs=3)
    runs = record_recent_run(runs, {"job_id": "job_4"}, max_runs=3)
    assert len(runs) == 3
    assert runs[0]["job_id"] == "job_4"


# 23. Timeline stages helper
def test_get_timeline_stages() -> None:
    state = WorkflowProgressState(job_id="test_job_123")
    state.status = "running"
    state.step_states["planner"] = "completed"
    state.step_states["coder"] = "running"

    stages = get_timeline_stages(state)
    assert len(stages) >= 10
    stage_ids = [s["id"] for s in stages]
    assert "job_queued" in stage_ids
    assert "workflow_started" in stage_ids
    assert "planner" in stage_ids
    assert "coder" in stage_ids
    assert "workflow_terminal" in stage_ids

    # Check status
    planner_stage = next(s for s in stages if s["id"] == "planner")
    assert planner_stage["status"] == "completed"
    coder_stage = next(s for s in stages if s["id"] == "coder")
    assert coder_stage["status"] == "running"


# 24. Sanitization of secrets and tokens
def test_sanitize_display_text_extended() -> None:
    leaked = (
        "Auth failed: approval_token='appr_99999', ticket_token='tkt_88888', "
        "api_key='secret_123', password='my_password', and Bearer tok_77777"
    )
    sanitized = sanitize_display_text(leaked)
    assert "appr_99999" not in sanitized
    assert "tkt_88888" not in sanitized
    assert "secret_123" not in sanitized
    assert "my_password" not in sanitized
    assert "tok_77777" not in sanitized
    assert "[REDACTED]" in sanitized


# 25. Visual theme palette verification
def test_visual_theme_palette_system() -> None:
    at = AppTest.from_file(APP_PATH).run()
    assert len(at.exception) == 0
    # Collect all raw markdown and styles
    md_contents = " ".join([m.value for m in at.markdown])
    # Verify core color system palette
    required_colors = (
        "#F4F7FB",  # Main application background
        "#EEF2F7",  # Secondary/sidebar background
        "#FFFFFF",  # Cards and major panels
        "#F8FAFC",  # Input backgrounds
        "#DCE3EC",  # Borders/dividers
        "#172033",  # Primary text
        "#64748B",  # Secondary text
        "#3B82F6",  # Primary accent
        "#16A34A",  # Success
        "#D97706",  # Warning
        "#DC2626",  # Error
    )
    for color in required_colors:
        assert color in md_contents, f"Expected palette color {color} in injected theme CSS"
