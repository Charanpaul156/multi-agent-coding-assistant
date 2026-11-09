from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from frontend.advanced_orchestration_ui import (
    AdvancedOrchestrationApiError,
    AdvancedProgressState,
    build_advanced_orchestration_payload,
    format_advanced_progress_event,
    get_advanced_orchestration_job_status,
    iter_advanced_orchestration_progress,
    map_advanced_orchestration_event,
    poll_advanced_orchestration_job,
    submit_advanced_orchestration_job,
)
from frontend.workflow_ui import iter_sse_events


@pytest.mark.parametrize(
    ("mode", "inputs"),
    [
        ("codebase_chat", {}),
        ("security_audit", {"files_input": {"main.py": "print('ok')"}}),
        ("documentation", {"code_input": "def answer(): return 42"}),
        ("bug_fix", {"code_input": "def answer(): return 42"}),
    ],
)
def test_build_payload_for_enabled_modes(mode: str, inputs: dict) -> None:
    payload = build_advanced_orchestration_payload(
        mode,
        "  Inspect this project  ",
        **inputs,
    )

    assert payload["mode"] == mode
    assert payload["prompt"] == "Inspect this project"
    assert set(payload) == {"mode", "prompt", *inputs}


@pytest.mark.parametrize("mode", ["documentation", "bug_fix"])
def test_code_modes_require_nonblank_code(mode: str) -> None:
    with pytest.raises(ValueError, match="code_input is required"):
        build_advanced_orchestration_payload(mode, "Fix this", code_input="  ")


def test_security_audit_requires_code_or_nonempty_file_content() -> None:
    with pytest.raises(ValueError, match="code_input or non-empty files_input"):
        build_advanced_orchestration_payload(
            "security_audit",
            "Audit this",
            files_input={"empty.py": "  "},
        )


@pytest.mark.parametrize("mode", ["test_generation", "feature_development"])
def test_generated_test_execution_modes_are_disabled(mode: str) -> None:
    with pytest.raises(ValueError, match="not enabled"):
        build_advanced_orchestration_payload(mode, "Run this")


def test_payload_rejects_unknown_mode_and_blank_prompt() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        build_advanced_orchestration_payload("unknown", "prompt")
    with pytest.raises(ValueError, match="prompt must not be blank"):
        build_advanced_orchestration_payload("codebase_chat", "  ")


def _response(status_code: int, body: object) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    if isinstance(body, Exception):
        response.json.side_effect = body
    else:
        response.json.return_value = body
    return response


def test_submit_advanced_job_returns_queued_reference() -> None:
    payload = build_advanced_orchestration_payload(
        "documentation", "Document code", code_input="def answer(): return 42"
    )
    with patch(
        "frontend.advanced_orchestration_ui.requests.request",
        return_value=_response(202, {"job_id": "adv-123", "status": "queued"}),
    ) as request:
        result = submit_advanced_orchestration_job("http://backend/", payload)

    assert result == {"job_id": "adv-123", "status": "queued"}
    request.assert_called_once_with(
        "POST",
        "http://backend/advanced-orchestration/jobs",
        timeout=15.0,
        json=payload,
    )


@pytest.mark.parametrize(
    "response",
    [
        _response(202, {"job_id": "", "status": "queued"}),
        _response(202, {"job_id": "adv-123", "status": "running"}),
        _response(202, ValueError("malformed JSON")),
        _response(202, ["not", "an", "object"]),
    ],
)
def test_submit_handles_malformed_responses(response: MagicMock) -> None:
    payload = {"mode": "codebase_chat", "prompt": "Explain module"}
    with patch(
        "frontend.advanced_orchestration_ui.requests.request",
        return_value=response,
    ):
        with pytest.raises(AdvancedOrchestrationApiError):
            submit_advanced_orchestration_job("http://backend", payload)


def test_submit_handles_http_error_without_echoing_response_body() -> None:
    sentinel = "PRIVATE_INPUT_SENTINEL_583"
    with patch(
        "frontend.advanced_orchestration_ui.requests.request",
        return_value=_response(422, {"detail": sentinel}),
    ):
        with pytest.raises(AdvancedOrchestrationApiError) as error:
            submit_advanced_orchestration_job(
                "http://backend",
                {"mode": "codebase_chat", "prompt": sentinel},
            )

    assert error.value.status_code == 422
    assert sentinel not in str(error.value)


def test_submit_handles_timeout_without_logging_or_echoing_inputs(caplog) -> None:
    sentinel = "PRIVATE_CODE_SENTINEL_741"
    with patch(
        "frontend.advanced_orchestration_ui.requests.request",
        side_effect=requests.Timeout("timed out"),
    ):
        with pytest.raises(AdvancedOrchestrationApiError, match="timed out"):
            submit_advanced_orchestration_job(
                "http://backend",
                {
                    "mode": "documentation",
                    "prompt": "Document this",
                    "code_input": sentinel,
                },
            )

    assert sentinel not in caplog.text


def test_get_status_preserves_job_and_orchestration_outcomes() -> None:
    body = {
        "job_id": "adv-123",
        "status": "completed",
        "created_at": "2026-01-01T00:00:00Z",
        "started_at": "2026-01-01T00:00:01Z",
        "completed_at": "2026-01-01T00:00:02Z",
        "result": {"success": False, "summary": "Validation failed"},
        "error": None,
        "metadata": {"job_type": "advanced_orchestration"},
    }
    with patch(
        "frontend.advanced_orchestration_ui.requests.request",
        return_value=_response(200, body),
    ) as request:
        result = get_advanced_orchestration_job_status("http://backend", "adv-123")

    assert result["status"] == "completed"
    assert result["result"]["success"] is False
    request.assert_called_once_with(
        "GET",
        "http://backend/advanced-orchestration/jobs/adv-123",
        timeout=10.0,
    )


def test_get_status_handles_404() -> None:
    with patch(
        "frontend.advanced_orchestration_ui.requests.request",
        return_value=_response(404, {"detail": "not found"}),
    ):
        with pytest.raises(AdvancedOrchestrationApiError) as error:
            get_advanced_orchestration_job_status("http://backend", "missing")

    assert error.value.status_code == 404
    assert "missing" not in str(error.value)


@pytest.mark.parametrize(
    "body",
    [
        {"status": "running"},
        {"job_id": "adv-123", "status": "unknown"},
        {"job_id": "adv-123", "status": "completed", "result": "not an object"},
    ],
)
def test_get_status_rejects_malformed_response(body: dict) -> None:
    with patch(
        "frontend.advanced_orchestration_ui.requests.request",
        return_value=_response(200, body),
    ):
        with pytest.raises(AdvancedOrchestrationApiError):
            get_advanced_orchestration_job_status("http://backend", "adv-123")


def test_step_events_map_dynamic_names_and_keep_only_progress_fields() -> None:
    sentinel = "SECRET_SOURCE_SENTINEL_583"
    state = AdvancedProgressState(job_id="adv-123")
    lines = [
        "event: step_start",
        'data: {"event_type":"step_start","job_id":"adv-123","payload":{"step":"ResearcherAgent","prompt":"'
        + sentinel
        + '"}}',
        "",
        "event: step_complete",
        'data: {"event_type":"step_complete","job_id":"adv-123","payload":{"step":"ResearcherAgent","duration_ms":12.5,"source_code":"'
        + sentinel
        + '"}}',
        "",
        "event: step_start",
        'data: {"event_type":"step_start","job_id":"adv-123","payload":{"step":"ValidatorAgent"}}',
        "",
        "event: step_failed",
        'data: {"event_type":"step_failed","job_id":"adv-123","payload":{"step":"ValidatorAgent","error":"'
        + sentinel
        + '"}}',
        "",
    ]

    messages = [message for _, message in iter_advanced_orchestration_progress(lines, state)]

    assert messages == [
        "ResearcherAgent started",
        "ResearcherAgent completed (12.5 ms)",
        "ValidatorAgent started",
        "ValidatorAgent failed",
    ]
    assert state.status == "running"
    assert state.steps["ResearcherAgent"].status == "completed"
    assert state.steps["ResearcherAgent"].duration_ms == 12.5
    assert state.steps["ValidatorAgent"].status == "failed"
    assert sentinel not in repr(state)
    assert sentinel not in " ".join(messages)


@pytest.mark.parametrize(
    ("event_type", "expected_status"),
    [("job_completed", "completed"), ("job_failed", "failed")],
)
def test_terminal_events_update_job_lifecycle(event_type: str, expected_status: str) -> None:
    state = AdvancedProgressState(job_id="adv-123")
    parsed = next(
        iter_sse_events(
            [
                f"event: {event_type}",
                'data: {"event_type":"' + event_type + '","job_id":"adv-123","payload":{"status":"' + expected_status + '"}}',
                "",
            ]
        )
    )

    progress_event = map_advanced_orchestration_event(parsed, state)

    assert progress_event is not None
    assert state.status == expected_status
    assert state.is_terminal is True
    assert format_advanced_progress_event(progress_event) == (
        "Advanced orchestration job completed"
        if event_type == "job_completed"
        else "Advanced orchestration job failed"
    )


@pytest.mark.parametrize("event_type", ["job_completed", "job_failed"])
def test_terminal_event_with_another_job_id_is_rejected(event_type: str) -> None:
    state = AdvancedProgressState(job_id="adv-123")
    parsed = next(
        iter_sse_events(
            [
                f"event: {event_type}",
                "data: "
                + json.dumps(
                    {
                        "event_type": event_type,
                        "job_id": "adv-other",
                        "payload": {},
                    }
                ),
                "",
            ]
        )
    )

    assert map_advanced_orchestration_event(parsed, state) is None
    assert state.status == "queued"
    assert state.is_terminal is False
    assert state.events == []


@pytest.mark.parametrize("event_type", ["job_completed", "job_failed"])
@pytest.mark.parametrize(
    ("include_job_id", "job_id"),
    [(False, None), (True, ""), (True, None), (True, 123)],
    ids=["missing", "empty", "null", "non-string"],
)
def test_terminal_event_with_invalid_job_id_is_rejected(
    event_type: str, include_job_id: bool, job_id: object
) -> None:
    state = AdvancedProgressState(job_id="adv-123")
    envelope = {"event_type": event_type, "payload": {}}
    if include_job_id:
        envelope["job_id"] = job_id
    parsed = next(
        iter_sse_events(
            [f"event: {event_type}", "data: " + json.dumps(envelope), ""]
        )
    )

    assert map_advanced_orchestration_event(parsed, state) is None
    assert state.status == "queued"
    assert state.is_terminal is False
    assert state.events == []


@pytest.mark.parametrize("event_type", ["job_completed", "job_failed"])
def test_malformed_terminal_event_does_not_update_job_lifecycle(event_type: str) -> None:
    state = AdvancedProgressState(job_id="adv-123")
    parsed = next(
        iter_sse_events(
            [
                f"event: {event_type}",
                'data: {"event_type":"' + event_type + '","payload":',
                "",
            ]
        )
    )

    progress_event = map_advanced_orchestration_event(parsed, state)

    assert progress_event is None
    assert state.status == "queued"
    assert state.is_terminal is False


@pytest.mark.parametrize("event_type", ["job_completed", "job_failed"])
def test_terminal_event_requires_matching_valid_envelope(event_type: str) -> None:
    state = AdvancedProgressState(job_id="adv-123")
    parsed = next(
        iter_sse_events(
            [
                f"event: {event_type}",
                'data: {"event_type":"step_start","payload":{}}',
                "",
            ]
        )
    )

    assert map_advanced_orchestration_event(parsed, state) is None
    assert state.is_terminal is False


def test_completed_job_may_have_unsuccessful_orchestration_result() -> None:
    body = {
        "job_id": "adv-123",
        "status": "completed",
        "result": {"success": False, "summary": "Review did not pass"},
    }
    with patch(
        "frontend.advanced_orchestration_ui.requests.request",
        return_value=_response(200, body),
    ):
        status = get_advanced_orchestration_job_status("http://backend", "adv-123")

    assert status["status"] == "completed"
    assert status["result"]["success"] is False


def test_polling_fallback_after_sse_interruption_stops_at_completed_status() -> None:
    running = {"job_id": "adv-123", "status": "running", "result": None}
    completed = {
        "job_id": "adv-123",
        "status": "completed",
        "result": {"success": False},
    }
    with patch(
        "frontend.advanced_orchestration_ui.requests.request",
        side_effect=[_response(200, running), _response(200, completed)],
    ) as request:
        on_status = MagicMock()
        sleep = MagicMock()
        result = poll_advanced_orchestration_job(
            "http://backend",
            "adv-123",
            max_attempts=4,
            poll_interval_seconds=0.25,
            on_status=on_status,
            sleep=sleep,
        )

    assert result["status"] == "completed"
    assert result["result"]["success"] is False
    assert request.call_count == 2
    assert sleep.call_args_list == [call(0.25)]
    assert on_status.call_count == 2


def test_polling_attempts_are_finite_and_caller_controlled() -> None:
    running = {"job_id": "adv-123", "status": "running", "result": None}
    with patch(
        "frontend.advanced_orchestration_ui.requests.request",
        return_value=_response(200, running),
    ) as request:
        result = poll_advanced_orchestration_job(
            "http://backend", "adv-123", max_attempts=2, sleep=MagicMock()
        )

    assert result["status"] == "running"
    assert request.call_count == 2


def test_polling_stops_when_job_failed() -> None:
    failed = {"job_id": "adv-123", "status": "failed", "error": "safe error"}
    with patch(
        "frontend.advanced_orchestration_ui.requests.request",
        return_value=_response(200, failed),
    ) as request:
        result = poll_advanced_orchestration_job(
            "http://backend", "adv-123", max_attempts=5
        )

    assert result["status"] == "failed"
    request.assert_called_once()


def test_progress_summaries_do_not_log_or_retain_sensitive_input(caplog) -> None:
    prompt_secret = "SECRET_PROMPT_SENTINEL_741"
    code_secret = "SECRET_CODE_SENTINEL_852"
    payload = build_advanced_orchestration_payload(
        "bug_fix", prompt_secret, code_input=code_secret
    )
    assert payload["prompt"] == prompt_secret
    assert payload["code_input"] == code_secret

    state = AdvancedProgressState(job_id="adv-123")
    event = next(
        iter_sse_events(
            [
                "event: step_start",
                'data: {"event_type":"step_start","job_id":"adv-123","payload":{"step":"DebuggerAgent","prompt":"'
                + prompt_secret
                + '","code_input":"'
                + code_secret
                + '"}}',
                "",
            ]
        )
    )
    progress = map_advanced_orchestration_event(event, state)
    assert progress is not None
    summary = format_advanced_progress_event(progress)

    assert prompt_secret not in summary
    assert code_secret not in summary
    assert prompt_secret not in repr(state)
    assert code_secret not in repr(state)
    assert prompt_secret not in caplog.text
    assert code_secret not in caplog.text
