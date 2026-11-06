"""Unit tests for the Advanced Multi-Agent Orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock
import pytest

from agents.code_reviewer_agent import ReviewFinding
from agents.debugger_agent import DebugReport
from agents.documentation_agent import DocumentationReport
from agents.rag_agent import ResearchReport, SourceReference
from agents.test_generator_agent import TestGenerationReport
from backend.application.advanced_orchestrator import AdvancedOrchestrator
from backend.application.event_emitter import RecordingEventEmitter
from backend.domain.job_models import WorkflowEvent
from backend.domain.orchestration_models import (
    AgentExecutionStatus,
    OrchestrationResult,
    WorkflowMode,
)
from backend.tools.test_executor import TestExecutionResponse


@dataclass(frozen=True)
class BaselineDebugReport:
    issue_detected: bool
    error_type: str
    root_cause: str
    affected_component: str
    explanation: str
    suggested_changes: list[str]
    corrected_code: str
    confidence: float
    final_summary: str
    error_category: str = "RuntimeError"
    line_number: int | None = None
    targeted_diff: str | None = None
    retry_count: int = 0
    max_retries: int = 3


@pytest.fixture
def mock_dependencies():
    planner = MagicMock()
    coder = MagicMock()
    reviewer = MagicMock()
    debugger = MagicMock()
    test_gen = MagicMock()
    test_exec = MagicMock()
    validator = MagicMock()
    docs = MagicMock()
    rag = MagicMock()

    return {
        "planner_use_case": planner,
        "coder_agent": coder,
        "reviewer_agent": reviewer,
        "debugger_agent": debugger,
        "test_generator_agent": test_gen,
        "test_executor": test_exec,
        "validator_agent": validator,
        "documentation_agent": docs,
        "rag_agent": rag,
    }


def test_orchestrator_codebase_chat_mode(mock_dependencies) -> None:
    mock_dependencies["rag_agent"].research.return_value = ResearchReport(
        query="How does auth work?",
        summary="Found auth source",
        answer="The auth module handles login",
        sources=[
            SourceReference(
                file_path="backend/auth.py", start_line=10, end_line=16,
                relevance_score=0.95, snippet="def login(...): ...",
                language="python", ranking_reasons=["query match"],
            )
        ],
        related_dependencies=["bcrypt"],
        architecture_insights=["Authentication is isolated."],
        confidence=0.95,
        evidence_status="available",
        warnings=["Partial index"],
        assumptions=["Login is the relevant path"],
        recommendations=["Inspect token handling"],
    )

    orchestrator = AdvancedOrchestrator(**mock_dependencies)
    result = orchestrator.execute("How does auth work?", mode=WorkflowMode.CODEBASE_CHAT)

    assert isinstance(result, OrchestrationResult)
    assert result.success is True
    assert result.mode == WorkflowMode.CODEBASE_CHAT
    assert len(result.agent_records) == 1
    assert result.agent_records[0].agent_name == "ResearcherAgent"
    research = result.output_data["research"]
    assert research["answer"] == "The auth module handles login"
    assert research["answer_status"] == "supported"
    assert research["sources"][0]["snippet"] == "def login(...): ..."
    assert research["warnings"] == ["Partial index"]
    assert research["assumptions"] == ["Login is the relevant path"]
    assert research["recommendations"] == ["Inspect token handling"]
    # Verify Planner or Coder was NOT called
    mock_dependencies["planner_use_case"].execute.assert_not_called()
    mock_dependencies["coder_agent"].generate_code.assert_not_called()


def test_codebase_chat_records_retrieval_failure(mock_dependencies) -> None:
    mock_dependencies["rag_agent"].research.side_effect = RuntimeError(
        "retrieval unavailable"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "How does auth work?", mode=WorkflowMode.CODEBASE_CHAT
    )

    assert result.success is False
    assert result.error == "retrieval unavailable"
    assert result.agent_records[0].status == AgentExecutionStatus.FAILED


def test_orchestrator_security_audit_mode(mock_dependencies) -> None:
    val_mock = mock_dependencies["validator_agent"]
    val_mock.validate_code.return_value = MagicMock(
        valid=True, error_count=0, warning_count=0, issues=[], summary="All good"
    )

    rev_mock = mock_dependencies["reviewer_agent"]
    finding = ReviewFinding(
        category="security",
        severity="High",
        message="Unsafe interpolation.",
        file_path="src/db.py",
        finding_id="REV-001",
        title="Unsafe SQL",
        evidence="execute(query)",
        impact="May permit injection.",
        confidence=0.9,
        status="confirmed",
    )
    rev_mock.review_code.return_value = MagicMock(
        overall_score=85,
        security_concerns=[],
        architectural_violations=[],
        findings=[finding],
        code_duplications=[],
    )

    orchestrator = AdvancedOrchestrator(**mock_dependencies)
    result = orchestrator.execute(
        "Audit this file",
        mode=WorkflowMode.SECURITY_AUDIT,
        code_input="def foo(): pass",
    )

    assert result.success is True
    assert result.mode == WorkflowMode.SECURITY_AUDIT
    assert len(result.agent_records) == 2
    assert result.agent_records[0].agent_name == "ValidatorAgent"
    assert result.agent_records[1].agent_name == "ReviewerAgent"
    serialized_finding = result.output_data["review"]["findings"][0]
    assert serialized_finding["finding_id"] == "REV-001"
    assert serialized_finding["file_path"] == "src/db.py"
    assert serialized_finding["evidence"] == "execute(query)"
    mock_dependencies["planner_use_case"].execute.assert_not_called()


def test_security_audit_validation_failure_with_successful_review(
    mock_dependencies,
) -> None:
    mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
        valid=False, error_count=2, warning_count=0, issues=[], summary="Invalid"
    )
    mock_dependencies["reviewer_agent"].review_code.return_value = MagicMock(
        overall_score=90,
        security_concerns=[],
        architectural_violations=[],
        findings=[],
        code_duplications=[],
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Audit", mode=WorkflowMode.SECURITY_AUDIT, code_input="def broken(:"
    )

    assert result.success is False
    assert result.output_data["validation"]["valid"] is False
    assert result.output_data["review"]["overall_score"] == 90
    assert result.agent_records[0].status == AgentExecutionStatus.FAILED
    assert result.agent_records[1].status == AgentExecutionStatus.COMPLETED
    assert "validation failures" in result.summary.casefold()


def test_security_audit_low_score_fails_with_threshold_summary_and_findings(
    mock_dependencies,
) -> None:
    mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
        valid=True, error_count=0, warning_count=0, issues=[], summary="Valid"
    )
    finding = ReviewFinding(
        category="security", severity="High", message="Unsafe query",
        evidence="cursor.execute(query)",
    )
    mock_dependencies["reviewer_agent"].review_code.return_value = MagicMock(
        overall_score=42, security_concerns=["SQL injection risk"],
        architectural_violations=[], findings=[finding], code_duplications=[],
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Audit", mode=WorkflowMode.SECURITY_AUDIT, code_input="query = input()"
    )

    assert result.success is False
    assert "failed" in result.summary.casefold()
    assert "42/100" in result.summary
    assert "60-point threshold" in result.summary
    assert result.output_data["review"]["findings"][0]["message"] == "Unsafe query"


def test_orchestrator_documentation_mode(mock_dependencies) -> None:
    doc_mock = mock_dependencies["documentation_agent"]
    doc_mock.generate_documentation.return_value = MagicMock(
        summary="Math docs",
        module_description="Desc",
        function_docs=[],
        class_docs=[],
        markdown_documentation="# Math",
    )

    orchestrator = AdvancedOrchestrator(**mock_dependencies)
    result = orchestrator.execute(
        "Generate docs",
        mode=WorkflowMode.DOCUMENTATION,
        code_input="def add(a, b): return a + b",
    )

    assert result.success is True
    assert result.mode == WorkflowMode.DOCUMENTATION
    assert len(result.agent_records) == 1
    assert result.agent_records[0].agent_name == "DocumentationAgent"


def test_feature_development_success_reports_validation_and_execution(
    mock_dependencies,
) -> None:
    from agents.planner_agent import ImplementationPlan
    from backend.application.planning_use_cases import GeneratePlanResult

    plan = ImplementationPlan(
        problem_summary="Build feature", project_type="module", requirements=[],
        modules=[], functions=[], classes=[], external_libraries=[],
        database_needed=False, api_needed=[], algorithm="", edge_cases=[],
        estimated_complexity="low", future_improvements=[],
    )
    mock_dependencies["planner_use_case"].execute.return_value = GeneratePlanResult(
        plan=plan
    )
    mock_dependencies["coder_agent"].generate_code.return_value = "def example(): return 1\n"
    mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
        valid=True, error_count=0, warning_count=0, issues=[], summary="Valid"
    )
    mock_dependencies["test_generator_agent"].generate_tests.return_value = TestGenerationReport(
        test_overview="One", test_framework="pytest", test_cases=[], edge_cases=[],
        negative_cases=[], coverage_suggestions=[], generated_test_code="def test_x(): pass",
        final_summary="One test",
    )
    mock_dependencies["test_executor"].execute.return_value = TestExecutionResponse(
        success=True, stdout="1 passed", stderr="", execution_time_ms=2,
        exit_code=0, passed=1, failed=0,
    )
    mock_dependencies["reviewer_agent"].review_code.return_value = MagicMock(
        overall_score=90, findings=[], recommendations=[], code_duplications=[]
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute("Build feature")

    assert result.success is True
    assert result.output_data["validation"]["valid"] is True
    assert result.output_data["test_execution"]["passed"] == 1
    assert result.output_data["review"]["score"] == 90
    assert [record.agent_name for record in result.agent_records] == [
        "PlannerAgent", "CoderAgent", "ValidatorAgent", "TestGeneratorAgent",
        "TestExecutor", "ReviewerAgent",
    ]


def test_orchestrator_bug_fix_consumes_debug_report_and_reports_unverified_status(
    mock_dependencies,
) -> None:
    debug_report = DebugReport(
        issue_detected=True,
        error_type="ZeroDivisionError",
        root_cause="Division by zero in divide.",
        affected_component="divide",
        explanation="The divisor is not guarded.",
        suggested_changes=["Guard against zero."],
        corrected_code="def divide(a, b):\n    if b == 0: return 0\n    return a / b\n",
        confidence=0.9,
        final_summary="Proposed correction; not executed.",
        error_category="RuntimeError",
        line_number=2,
        targeted_diff="@@ -1 +1 @@",
        root_cause_status="confirmed",
        evidence=["return a / b"],
        verification_status="unverified",
    )
    mock_dependencies["debugger_agent"].debug_code.return_value = debug_report
    mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
        valid=True, error_count=0, summary="Syntax valid"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Investigate division failure",
        mode=WorkflowMode.BUG_FIX,
        code_input="def divide(a, b): return a / b",
    )

    assert result.success is True
    assert "not been executed" in result.summary
    assert result.output_data["debug"]["error_category"] == "RuntimeError"
    assert result.output_data["debug"]["line_number"] == 2
    assert result.output_data["debug"]["targeted_diff"] == "@@ -1 +1 @@"
    assert result.output_data["debug"]["root_cause_status"] == "confirmed"
    mock_dependencies["validator_agent"].validate_code.assert_called_once_with(
        debug_report.corrected_code
    )


def test_orchestrator_bug_fix_propagates_debugger_failure_as_failed_result(
    mock_dependencies,
) -> None:
    mock_dependencies["debugger_agent"].debug_code.side_effect = RuntimeError(
        "debugger unavailable"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Investigate failure",
        mode=WorkflowMode.BUG_FIX,
        code_input="def f(): return 1",
    )

    assert result.success is False
    assert result.error == "debugger unavailable"
    mock_dependencies["validator_agent"].validate_code.assert_not_called()


def test_orchestrator_bug_fix_reports_validation_failure_without_success_claim(
    mock_dependencies,
) -> None:
    mock_dependencies["debugger_agent"].debug_code.return_value = DebugReport(
        issue_detected=True,
        error_type="SyntaxError",
        root_cause="Unclosed expression.",
        affected_component="module",
        explanation="The proposed correction is invalid.",
        suggested_changes=["Close the expression."],
        corrected_code="def broken(:\n    pass\n",
        confidence=0.8,
        final_summary="A correction is proposed.",
        error_category="SyntaxError",
    )
    mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
        valid=False, error_count=1, summary="Syntax error"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Investigate syntax failure",
        mode=WorkflowMode.BUG_FIX,
        code_input="def f(): return 1",
    )

    assert result.success is False
    assert "failed validation" in result.summary


def test_bug_fix_no_issue_reports_source_validation_without_claiming_a_fix(
    mock_dependencies,
) -> None:
    mock_dependencies["debugger_agent"].debug_code.return_value = DebugReport(
        issue_detected=False, error_type="UnknownError", root_cause="None",
        affected_component="module", explanation="No issue was established.",
        suggested_changes=[], corrected_code="def f(): return 1",
        confidence=0.0, final_summary="No issue detected.",
    )
    mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
        valid=True, error_count=0, summary="Valid"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Investigate", mode=WorkflowMode.BUG_FIX, code_input="def f(): return 1"
    )

    assert result.success is True
    assert "no issue detected" in result.summary.casefold()
    assert "not been executed" not in result.summary.casefold()


def test_bug_fix_records_validator_exception(mock_dependencies) -> None:
    mock_dependencies["debugger_agent"].debug_code.return_value = DebugReport(
        issue_detected=True, error_type="ValueError", root_cause="Bad value",
        affected_component="f", explanation="Invalid input.",
        suggested_changes=["Validate input."], corrected_code="def f(): return 1",
        confidence=0.8, final_summary="Proposed correction.",
    )
    mock_dependencies["validator_agent"].validate_code.side_effect = RuntimeError(
        "validator unavailable"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Investigate", mode=WorkflowMode.BUG_FIX, code_input="def f(): return 1"
    )

    assert result.success is False
    assert result.error == "validator unavailable"
    assert result.output_data["debug"]["root_cause_status"] == "hypothesis"
    assert result.output_data["debug"]["verification_status"] == "unverified"


def test_bug_fix_accepts_baseline_debug_report_without_extended_fields(
    mock_dependencies,
) -> None:
    baseline_report = BaselineDebugReport(
        issue_detected=True, error_type="ValueError", root_cause="Bad input",
        affected_component="f", explanation="Invalid input.",
        suggested_changes=["Validate input."], corrected_code="def f(): return 1",
        confidence=0.8, final_summary="Proposed correction.",
        error_category="ValueError", line_number=3, targeted_diff="@@ diff @@",
        retry_count=0, max_retries=3,
    )
    mock_dependencies["debugger_agent"].debug_code.return_value = baseline_report
    mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
        valid=True, error_count=0, summary="Valid"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Investigate", mode=WorkflowMode.BUG_FIX, code_input="def f(): return 1"
    )

    assert result.success is True
    assert result.output_data["debug"]["root_cause_status"] == "hypothesis"
    assert result.output_data["debug"]["evidence"] == []
    assert result.output_data["debug"]["verification_status"] == "unverified"


def test_test_generation_mode_uses_current_report_contracts(mock_dependencies) -> None:
    test_report = TestGenerationReport(
        test_overview="One test",
        test_framework="pytest",
        test_cases=[],
        edge_cases=[],
        negative_cases=[],
        coverage_suggestions=["Add a boundary test."],
        generated_test_code="def test_example():\n    assert True\n",
        final_summary="Generated one test.",
    )
    execution = TestExecutionResponse(
        success=True,
        stdout="1 passed",
        stderr="",
        execution_time_ms=12.0,
        exit_code=0,
        passed=1,
        failed=0,
    )
    mock_dependencies["test_generator_agent"].generate_tests.return_value = test_report
    mock_dependencies["test_executor"].execute.return_value = execution

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Generate tests", mode=WorkflowMode.TEST_GENERATION,
        code_input="def example(): return True",
    )

    assert result.success is True
    assert result.output_data["test_generation"]["negative_cases_count"] == 0
    assert result.output_data["test_generation"]["coverage_suggestions"] == [
        "Add a boundary test."
    ]
    assert result.output_data["test_execution"]["passed"] == 1
    mock_dependencies["test_executor"].execute.assert_called_once()
    assert [record.status for record in result.agent_records] == [
        AgentExecutionStatus.COMPLETED,
        AgentExecutionStatus.COMPLETED,
    ]


def test_test_generation_mode_records_generation_failure(mock_dependencies) -> None:
    mock_dependencies["test_generator_agent"].generate_tests.side_effect = RuntimeError(
        "generation failed"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Generate tests", mode=WorkflowMode.TEST_GENERATION,
        code_input="def example(): return True",
    )

    assert result.success is False
    assert result.error == "generation failed"
    assert len(result.agent_records) == 1
    assert result.agent_records[0].status == AgentExecutionStatus.FAILED
    mock_dependencies["test_executor"].execute.assert_not_called()


def test_test_generation_mode_records_executor_failure(mock_dependencies) -> None:
    mock_dependencies["test_generator_agent"].generate_tests.return_value = (
        TestGenerationReport(
            test_overview="One test", test_framework="pytest", test_cases=[],
            edge_cases=[], negative_cases=[], coverage_suggestions=[],
            generated_test_code="def test_example(): assert True",
            final_summary="Generated one test",
        )
    )
    mock_dependencies["test_executor"].execute.side_effect = RuntimeError(
        "executor unavailable"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Generate tests", mode=WorkflowMode.TEST_GENERATION,
        code_input="def example(): return True",
    )

    assert result.success is False
    assert result.error == "executor unavailable"
    assert [record.status for record in result.agent_records] == [
        AgentExecutionStatus.COMPLETED, AgentExecutionStatus.FAILED,
    ]


@pytest.mark.parametrize("failure_stage", ["planner", "coder", "test_generator"])
def test_feature_development_records_early_stage_exceptions(
    mock_dependencies, failure_stage: str
) -> None:
    from agents.planner_agent import ImplementationPlan
    from backend.application.planning_use_cases import GeneratePlanResult

    if failure_stage == "planner":
        mock_dependencies["planner_use_case"].execute.side_effect = RuntimeError(
            "planner unavailable"
        )
    else:
        plan = ImplementationPlan(
            problem_summary="Build feature", project_type="module", requirements=[],
            modules=[], functions=[], classes=[], external_libraries=[],
            database_needed=False, api_needed=[], algorithm="", edge_cases=[],
            estimated_complexity="low", future_improvements=[],
        )
        mock_dependencies["planner_use_case"].execute.return_value = GeneratePlanResult(
            plan=plan
        )
        if failure_stage == "coder":
            mock_dependencies["coder_agent"].generate_code.side_effect = RuntimeError(
                "coder unavailable"
            )
        else:
            mock_dependencies["coder_agent"].generate_code.return_value = "def f(): return 1"
            mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
                valid=True, error_count=0, warning_count=0, issues=[], summary="Valid"
            )
            mock_dependencies["test_generator_agent"].generate_tests.side_effect = (
                RuntimeError("test generator unavailable")
            )

    result = AdvancedOrchestrator(**mock_dependencies).execute("Build feature")

    assert result.success is False
    expected_message = {
        "planner": "planner unavailable",
        "coder": "coder unavailable",
        "test_generator": "test generator unavailable",
    }[failure_stage]
    assert result.error == expected_message
    assert result.agent_records[-1].status == AgentExecutionStatus.FAILED


def test_feature_development_stops_after_validation_failure(mock_dependencies) -> None:
    from agents.planner_agent import ImplementationPlan
    from backend.application.planning_use_cases import GeneratePlanResult

    plan = ImplementationPlan(
        problem_summary="Build feature", project_type="module", requirements=[],
        modules=[], functions=[], classes=[], external_libraries=[],
        database_needed=False, api_needed=[], algorithm="", edge_cases=[],
        estimated_complexity="low", future_improvements=[],
    )
    mock_dependencies["planner_use_case"].execute.return_value = GeneratePlanResult(
        plan=plan
    )
    mock_dependencies["coder_agent"].generate_code.return_value = "def broken(:\n"
    mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
        valid=False, error_count=1, warning_count=0, issues=[], summary="Invalid syntax"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute("Build feature")

    assert result.success is False
    assert result.output_data["code"] == "def broken(:\n"
    assert result.output_data["validation"]["valid"] is False
    assert "validation failed" in result.summary.casefold()
    assert [record.agent_name for record in result.agent_records] == [
        "PlannerAgent", "CoderAgent", "ValidatorAgent"
    ]
    mock_dependencies["test_generator_agent"].generate_tests.assert_not_called()
    mock_dependencies["test_executor"].execute.assert_not_called()
    mock_dependencies["reviewer_agent"].review_code.assert_not_called()


def test_feature_development_reports_execution_failure(mock_dependencies) -> None:
    from agents.planner_agent import ImplementationPlan
    from backend.application.planning_use_cases import GeneratePlanResult

    plan = ImplementationPlan(
        problem_summary="Build feature", project_type="module", requirements=[],
        modules=[], functions=[], classes=[], external_libraries=[],
        database_needed=False, api_needed=[], algorithm="", edge_cases=[],
        estimated_complexity="low", future_improvements=[],
    )
    mock_dependencies["planner_use_case"].execute.return_value = GeneratePlanResult(
        plan=plan
    )
    mock_dependencies["coder_agent"].generate_code.return_value = "def example(): return 1\n"
    mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
        valid=True, error_count=0, warning_count=0, issues=[], summary="Valid"
    )
    mock_dependencies["test_generator_agent"].generate_tests.return_value = TestGenerationReport(
        test_overview="One", test_framework="pytest", test_cases=[], edge_cases=[],
        negative_cases=[], coverage_suggestions=[], generated_test_code="def test_x(): pass",
        final_summary="One test",
    )
    mock_dependencies["test_executor"].execute.return_value = TestExecutionResponse(
        success=False, stdout="1 failed", stderr="", execution_time_ms=2,
        exit_code=1, passed=0, failed=1,
    )
    mock_dependencies["reviewer_agent"].review_code.return_value = MagicMock(
        overall_score=50, findings=[], recommendations=[], code_duplications=[]
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute("Build feature")

    assert result.success is False
    assert result.output_data["test_execution"]["success"] is False
    assert "tests failed" in result.summary.casefold()


def test_feature_development_records_reviewer_exception_with_partial_results(
    mock_dependencies,
) -> None:
    from agents.planner_agent import ImplementationPlan
    from backend.application.planning_use_cases import GeneratePlanResult

    plan = ImplementationPlan(
        problem_summary="Build feature", project_type="module", requirements=[],
        modules=[], functions=[], classes=[], external_libraries=[],
        database_needed=False, api_needed=[], algorithm="", edge_cases=[],
        estimated_complexity="low", future_improvements=[],
    )
    mock_dependencies["planner_use_case"].execute.return_value = GeneratePlanResult(
        plan=plan
    )
    mock_dependencies["coder_agent"].generate_code.return_value = "def example(): return 1\n"
    mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
        valid=True, error_count=0, warning_count=0, issues=[], summary="Valid"
    )
    mock_dependencies["test_generator_agent"].generate_tests.return_value = TestGenerationReport(
        test_overview="One", test_framework="pytest", test_cases=[], edge_cases=[],
        negative_cases=[], coverage_suggestions=[], generated_test_code="def test_x(): pass",
        final_summary="One test",
    )
    mock_dependencies["test_executor"].execute.return_value = TestExecutionResponse(
        success=True, stdout="1 passed", stderr="", execution_time_ms=2,
        exit_code=0, passed=1, failed=0,
    )
    mock_dependencies["reviewer_agent"].review_code.side_effect = RuntimeError(
        "review unavailable"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute("Build feature")

    assert result.success is False
    assert result.error == "review unavailable"
    assert result.output_data["code"]
    assert result.output_data["validation"]["valid"] is True
    assert result.output_data["test_execution"]["success"] is True
    assert result.agent_records[-1].agent_name == "ReviewerAgent"
    assert result.agent_records[-1].status == AgentExecutionStatus.FAILED


def test_security_audit_records_reviewer_failure_and_validation_result(
    mock_dependencies,
) -> None:
    mock_dependencies["validator_agent"].validate_code.return_value = MagicMock(
        valid=False, error_count=1, warning_count=0, issues=[], summary="Invalid"
    )
    mock_dependencies["reviewer_agent"].review_code.side_effect = RuntimeError(
        "review unavailable"
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Audit", mode=WorkflowMode.SECURITY_AUDIT, code_input="def broken(:"
    )

    assert result.success is False
    assert result.error == "review unavailable"
    assert result.output_data["validation"]["valid"] is False
    assert [record.status for record in result.agent_records] == [
        AgentExecutionStatus.FAILED, AgentExecutionStatus.FAILED
    ]


def test_codebase_chat_preserves_evidence_metadata(mock_dependencies) -> None:
    mock_dependencies["rag_agent"].research.return_value = ResearchReport(
        query="q", summary="No evidence", answer="Unknown", sources=[],
        confidence=0.0, evidence_status="unavailable",
        warnings=["No retriever"], assumptions=["None"],
        recommendations=["Index repository"],
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Ask", mode=WorkflowMode.CODEBASE_CHAT
    )

    research = result.output_data["research"]
    assert research["evidence_status"] == "unavailable"
    assert result.success is False
    assert research["answer"] is None
    assert research["unsupported_answer"] == "Unknown"
    assert research["warnings"] == ["No retriever"]
    assert research["assumptions"] == ["None"]
    assert research["recommendations"] == ["Index repository"]


def test_codebase_chat_rejects_available_status_without_sources(mock_dependencies) -> None:
    mock_dependencies["rag_agent"].research.return_value = ResearchReport(
        query="q", summary="Claim without citations", answer="Unsupported claim",
        sources=[], confidence=0.95, evidence_status="available",
        warnings=[], assumptions=[], recommendations=[],
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Ask", mode=WorkflowMode.CODEBASE_CHAT
    )

    assert result.success is False
    research = result.output_data["research"]
    assert research["answer"] is None
    assert research["unsupported_answer"] == "Unsupported claim"
    assert research["unsupported_reason"] == "no source references were returned"


def test_codebase_chat_rejects_low_confidence_answer(mock_dependencies) -> None:
    mock_dependencies["rag_agent"].research.return_value = ResearchReport(
        query="q", summary="Weak evidence", answer="Possibly true",
        sources=[
            SourceReference(
                file_path="module.py", start_line=1, end_line=2,
                relevance_score=0.2, snippet="pass", language="python",
            )
        ], confidence=0.2, evidence_status="available",
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Ask", mode=WorkflowMode.CODEBASE_CHAT
    )

    assert result.success is False
    research = result.output_data["research"]
    assert research["answer"] is None
    assert research["unsupported_reason"] == "confidence is missing or below 0.5"


def test_documentation_failure_records_failed_step_and_partial_result(
    mock_dependencies,
) -> None:
    mock_dependencies["documentation_agent"].generate_documentation.side_effect = (
        RuntimeError("documentation unavailable")
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Document", mode=WorkflowMode.DOCUMENTATION, code_input="def f(): pass"
    )

    assert result.success is False
    assert result.error == "documentation unavailable"
    assert len(result.agent_records) == 1
    assert result.agent_records[0].status == AgentExecutionStatus.FAILED


def test_step_timestamps_reflect_actual_start_and_completion(mock_dependencies) -> None:
    mock_dependencies["documentation_agent"].generate_documentation.return_value = (
        MagicMock(
            summary="Docs", module_description="", function_docs=[], class_docs=[],
            markdown_documentation="# Docs",
        )
    )

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Document", mode=WorkflowMode.DOCUMENTATION, code_input="def f(): pass"
    )

    record = result.agent_records[0]
    from datetime import datetime

    assert datetime.fromisoformat(record.started_at) <= datetime.fromisoformat(
        record.completed_at
    )
    assert record.input_summary == "Document"


def test_event_emitter_receives_ordered_workflow_events_for_success(
    mock_dependencies,
) -> None:
    mock_dependencies["documentation_agent"].generate_documentation.return_value = (
        DocumentationReport(
            summary="Docs", module_description="Module", function_docs=[],
            class_docs=[], usage_examples=[], markdown_documentation="# Docs",
        )
    )
    emitter = RecordingEventEmitter()

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Document", mode=WorkflowMode.DOCUMENTATION,
        code_input="def f(): pass", event_emitter=emitter, job_id="job-123",
    )

    assert result.success is True
    events = emitter.get_events()
    assert [event.event_type for event in events] == ["step_start", "step_complete"]
    assert all(isinstance(event, WorkflowEvent) for event in events)
    assert all(event.job_id == "job-123" for event in events)
    assert events[0].payload == {"step": "DocumentationAgent"}
    assert events[1].payload["step"] == "DocumentationAgent"
    assert events[1].payload["duration_ms"] >= 0


def test_event_emitter_receives_failed_step_event_in_order(mock_dependencies) -> None:
    mock_dependencies["documentation_agent"].generate_documentation.side_effect = (
        RuntimeError("documentation unavailable")
    )
    emitter = RecordingEventEmitter()

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Document", mode=WorkflowMode.DOCUMENTATION,
        code_input="def f(): pass", event_emitter=emitter, job_id="job-456",
    )

    assert result.success is False
    events = emitter.get_events()
    assert [event.event_type for event in events] == ["step_start", "step_failed"]
    assert events[0].payload == {"step": "DocumentationAgent"}
    assert events[1].payload == {
        "step": "DocumentationAgent", "error": "documentation unavailable",
    }
    assert all(event.job_id == "job-456" for event in events)


def test_event_emitter_interface_failure_is_returned_not_swallowed(
    mock_dependencies,
) -> None:
    class BrokenEmitter:
        job_id = "job-broken"

    result = AdvancedOrchestrator(**mock_dependencies).execute(
        "Document", mode=WorkflowMode.DOCUMENTATION,
        code_input="def f(): pass", event_emitter=BrokenEmitter(),
    )

    assert result.success is False
    assert "emit" in (result.error or "").casefold()
    assert result.agent_records[0].status == AgentExecutionStatus.FAILED
