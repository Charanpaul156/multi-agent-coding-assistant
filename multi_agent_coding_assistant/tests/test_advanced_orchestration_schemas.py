from datetime import datetime, timezone
from enum import Enum

import pytest
from pydantic import BaseModel, ValidationError

from backend.api.job_schemas import (
    AdvancedOrchestrationJobPayload,
    serialize_orchestration_result,
)
from backend.domain.orchestration_models import (
    AgentExecutionRecord,
    AgentExecutionStatus,
    OrchestrationResult,
    WorkflowMode,
)


@pytest.mark.parametrize("mode", list(WorkflowMode))
def test_advanced_request_accepts_each_workflow_mode(mode: WorkflowMode) -> None:
    payload = {"mode": mode.value, "prompt": "  inspect this  "}
    if mode in {
        WorkflowMode.TEST_GENERATION,
        WorkflowMode.DOCUMENTATION,
        WorkflowMode.BUG_FIX,
    }:
        payload["code_input"] = "print('hello')"
    elif mode == WorkflowMode.SECURITY_AUDIT:
        payload["files_input"] = {"main.py": "print('hello')"}

    request = AdvancedOrchestrationJobPayload.model_validate(payload)

    assert request.mode == mode
    assert request.prompt == "inspect this"


def test_advanced_request_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError, match="mode"):
        AdvancedOrchestrationJobPayload(mode="made_up", prompt="inspect")


@pytest.mark.parametrize(
    "mode",
    [
        WorkflowMode.TEST_GENERATION,
        WorkflowMode.DOCUMENTATION,
        WorkflowMode.BUG_FIX,
    ],
)
def test_code_modes_require_code_input(mode: WorkflowMode) -> None:
    with pytest.raises(ValidationError, match="code_input is required"):
        AdvancedOrchestrationJobPayload(mode=mode, prompt="inspect")


def test_security_audit_requires_code_or_nonempty_file_content() -> None:
    with pytest.raises(ValidationError, match="code_input or non-empty files_input"):
        AdvancedOrchestrationJobPayload(
            mode=WorkflowMode.SECURITY_AUDIT,
            prompt="audit",
            files_input={"empty.py": "  "},
        )


def test_orchestration_result_serializes_all_result_fields_and_nested_values() -> None:
    class DetailKind(str, Enum):
        CHECK = "check"

    class NestedModel(BaseModel):
        checked_at: datetime
        kind: DetailKind

    checked_at = datetime(2025, 1, 2, tzinfo=timezone.utc)
    record = AgentExecutionRecord(
        agent_name="ValidatorAgent",
        status=AgentExecutionStatus.COMPLETED,
        started_at="2025-01-02T00:00:00+00:00",
        completed_at="2025-01-02T00:00:01+00:00",
        duration_ms=1000.0,
        input_summary="source",
        output_summary="valid",
        details={
            "nested": NestedModel(checked_at=checked_at, kind=DetailKind.CHECK),
            "statuses": [AgentExecutionStatus.COMPLETED],
        },
    )
    result = OrchestrationResult(
        mode=WorkflowMode.SECURITY_AUDIT,
        success=False,
        summary="Audit found issues",
        agent_records=[record],
        total_duration_ms=1001.5,
        output_data={"validation": {"valid": False}},
        error="Policy failure",
    )

    serialized = serialize_orchestration_result(result)

    assert serialized["mode"] == "security_audit"
    assert serialized["success"] is False
    assert serialized["summary"] == "Audit found issues"
    assert serialized["error"] == "Policy failure"
    assert serialized["total_duration_ms"] == 1001.5
    assert serialized["output_data"] == {"validation": {"valid": False}}
    assert serialized["agent_records"][0]["status"] == "completed"
    assert serialized["agent_records"][0]["details"] == {
        "nested": {"checked_at": "2025-01-02T00:00:00Z", "kind": "check"},
        "statuses": ["completed"],
    }


def test_orchestration_serializer_rejects_wrong_input_type() -> None:
    with pytest.raises(TypeError, match="OrchestrationResult"):
        serialize_orchestration_result({"success": True})  # type: ignore[arg-type]


def test_orchestration_serializer_rejects_non_json_numbers() -> None:
    result = OrchestrationResult(
        mode=WorkflowMode.FEATURE_DEVELOPMENT,
        success=True,
        summary="done",
        output_data={"invalid": float("nan")},
    )

    with pytest.raises(TypeError, match="non-JSON-compatible"):
        serialize_orchestration_result(result)
