"""Advanced Multi-Agent Orchestrator.

Provides mode-based dynamic agent selection, execution dependency management,
structured inter-agent communication, telemetry tracking, and bounded retries.

PRESERVATION RULE:
- NEVER modifies the Planner Agent. Reuses GeneratePlanUseCase directly when planning is required.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.coder_agent import CoderAgent
from agents.code_reviewer_agent import ReviewerAgent
from agents.debugger_agent import DebuggerAgent
from agents.documentation_agent import DocumentationAgent
from agents.rag_agent import RagAgent
from agents.validator_agent import ValidatorAgent
from backend.application.event_emitter import EventEmitter
from backend.application.planning_use_cases import GeneratePlanRequest, GeneratePlanUseCase
from backend.domain.change_models import ChangeSet, FileChange
from backend.domain.job_models import WorkflowEvent
from backend.domain.orchestration_models import (
    AgentExecutionRecord,
    AgentExecutionStatus,
    OrchestrationResult,
    WorkflowMode,
)
from backend.tools.python_executor import ExecuteCodeUseCase, ExecutionRequest
from backend.tools.test_executor import ExecuteTestsUseCase, TestExecutionRequest

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_review_finding(finding: Any) -> Dict[str, Any]:
    return {
        "finding_id": getattr(finding, "finding_id", None),
        "title": getattr(finding, "title", None),
        "description": getattr(finding, "description", None),
        "category": getattr(finding, "category", "general"),
        "severity": getattr(finding, "severity", "Informational"),
        "message": getattr(finding, "message", ""),
        "status": getattr(finding, "status", "potential"),
        "file_path": getattr(finding, "file_path", None),
        "line": getattr(finding, "line_number", None),
        "line_start": getattr(finding, "line_start", None),
        "line_end": getattr(finding, "line_end", None),
        "evidence": getattr(finding, "evidence", None),
        "impact": getattr(finding, "impact", None),
        "suggestion": getattr(finding, "suggestion", None),
        "confidence": getattr(finding, "confidence", 0.0),
    }


class AdvancedOrchestrator:
    """Dynamic multi-agent orchestrator deciding and coordinating specialized agents."""

    def __init__(
        self,
        *,
        planner_use_case: GeneratePlanUseCase,
        coder_agent: CoderAgent,
        reviewer_agent: ReviewerAgent,
        debugger_agent: DebuggerAgent,
        test_generator_agent: Any,
        test_executor: ExecuteTestsUseCase,
        validator_agent: ValidatorAgent,
        documentation_agent: DocumentationAgent,
        rag_agent: RagAgent,
        python_executor: Optional[ExecuteCodeUseCase] = None,
        max_retries: int = 3,
    ) -> None:
        self._planner = planner_use_case
        self._coder = coder_agent
        self._reviewer = reviewer_agent
        self._debugger = debugger_agent
        self._test_generator = test_generator_agent
        self._test_executor = test_executor
        self._validator = validator_agent
        self._documentation = documentation_agent
        self._rag = rag_agent
        self._python_executor = python_executor
        self._max_retries = max(1, int(max_retries))

    def execute(
        self,
        prompt: str,
        *,
        mode: WorkflowMode = WorkflowMode.FEATURE_DEVELOPMENT,
        code_input: Optional[str] = None,
        files_input: Optional[Dict[str, str]] = None,
        repository_path: Optional[str | Path] = None,
        language: Optional[str] = None,
        event_emitter: Optional[EventEmitter] = None,
        job_id: Optional[str] = None,
    ) -> OrchestrationResult:
        """Run the orchestrated workflow according to the selected mode."""
        start_time = time.perf_counter()
        agent_records: List[AgentExecutionRecord] = []
        output_data: Dict[str, Any] = {}
        active_steps: Dict[float, tuple[str, str, str]] = {}

        logger.info("AdvancedOrchestrator: executing in mode '%s'", mode.value)
        event_job_id = job_id or getattr(event_emitter, "job_id", None)
        event_emission_error: Optional[str] = None

        def emit_step_event(event_type: str, payload: Dict[str, Any]) -> None:
            nonlocal event_emission_error
            if event_emitter is None:
                return
            if not event_job_id:
                event_emission_error = (
                    "job_id is required when event_emitter is provided"
                )
                raise ValueError(event_emission_error)
            try:
                event_emitter.emit(
                    WorkflowEvent(
                        event_type=event_type,
                        job_id=event_job_id,
                        payload=payload,
                    )
                )
            except Exception as exc:
                event_emission_error = str(exc)
                raise

        def record_agent_start(name: str, input_summary: str) -> float:
            t0 = time.perf_counter()
            active_steps[t0] = (name, input_summary_truncate(input_summary), _now_iso())
            emit_step_event("step_start", {"step": name})
            return t0

        def record_agent_finish(
            name: str,
            t0: float,
            status: AgentExecutionStatus,
            output_summary: str,
            details: Dict[str, Any],
            err: Optional[str] = None,
        ) -> None:
            dur = (time.perf_counter() - t0) * 1000
            active = active_steps.pop(t0, None)
            started_at = active[2] if active else _now_iso()
            rec = AgentExecutionRecord(
                agent_name=name,
                status=status,
                started_at=started_at,
                completed_at=_now_iso(),
                duration_ms=dur,
                input_summary=(active[1] if active else input_summary_truncate(prompt)),
                output_summary=output_summary,
                error=err,
                details=details,
            )
            agent_records.append(rec)
            if status == AgentExecutionStatus.COMPLETED:
                emit_step_event(
                    "step_complete", {"step": name, "duration_ms": dur}
                )
            elif status == AgentExecutionStatus.FAILED:
                emit_step_event(
                    "step_failed",
                    {
                        "step": name,
                        "error": "Step failed. See the orchestration result for details.",
                    },
                )

        def input_summary_truncate(text: str, max_len: int = 120) -> str:
            t = (text or "").strip()
            return t[:max_len] + "..." if len(t) > max_len else t

        try:
            # ------------------------------------------------------------------ #
            # MODE: CODEBASE_CHAT (RAG Researcher only)
            # ------------------------------------------------------------------ #
            if mode == WorkflowMode.CODEBASE_CHAT:
                t0 = record_agent_start("ResearcherAgent", prompt)
                report = self._rag.research(prompt)
                evidence_status = getattr(report, "evidence_status", "unavailable")
                if not isinstance(evidence_status, str):
                    evidence_status = "unavailable"
                confidence = getattr(report, "confidence", 0.0)
                has_confidence = (
                    isinstance(confidence, (int, float))
                    and not isinstance(confidence, bool)
                    and confidence >= 0.5
                )
                research_success = (
                    evidence_status == "available"
                    and bool(report.sources)
                    and has_confidence
                )
                if research_success:
                    unsupported_reason = None
                elif evidence_status != "available":
                    unsupported_reason = f"evidence status is {evidence_status}"
                elif not report.sources:
                    unsupported_reason = "no source references were returned"
                else:
                    unsupported_reason = "confidence is missing or below 0.5"
                output_data["research"] = {
                    "summary": report.summary,
                    "answer": report.answer if research_success else None,
                    "unsupported_answer": report.answer if not research_success else None,
                    "answer_status": "supported" if research_success else "insufficient_evidence",
                    "sources": [
                        {
                            "file_path": s.file_path,
                            "start_line": s.start_line,
                            "end_line": s.end_line,
                            "relevance": s.relevance_score,
                            "snippet": getattr(s, "snippet", ""),
                            "language": getattr(s, "language", ""),
                            "ranking_reasons": getattr(s, "ranking_reasons", []),
                        }
                        for s in report.sources
                    ],
                    "dependencies": report.related_dependencies,
                    "insights": report.architecture_insights,
                    "confidence": report.confidence,
                    "evidence_status": evidence_status,
                    "warnings": getattr(report, "warnings", []),
                    "assumptions": getattr(report, "assumptions", []),
                    "recommendations": getattr(report, "recommendations", []),
                    "unsupported_reason": unsupported_reason,
                }
                record_agent_finish(
                    "ResearcherAgent",
                    t0,
                    AgentExecutionStatus.COMPLETED if research_success else AgentExecutionStatus.FAILED,
                    report.summary,
                    {"sources_count": len(report.sources), "evidence_status": evidence_status},
                    None if research_success else f"Research evidence insufficient: {unsupported_reason}",
                )
                total_dur = (time.perf_counter() - start_time) * 1000
                return OrchestrationResult(
                    mode=mode,
                    success=research_success,
                    summary=(
                        report.summary
                        if research_success
                        else f"Research answer is unsupported because {unsupported_reason} ({evidence_status}): {report.summary}"
                    ),
                    agent_records=agent_records,
                    total_duration_ms=total_dur,
                    output_data=output_data,
                )

            # ------------------------------------------------------------------ #
            # MODE: SECURITY_AUDIT (Validator + Reviewer)
            # ------------------------------------------------------------------ #
            elif mode == WorkflowMode.SECURITY_AUDIT:
                target_code = code_input or ""
                if not target_code and files_input:
                    target_code = "\n\n".join(files_input.values())

                t0 = record_agent_start("ValidatorAgent", prompt)
                val_report = self._validator.validate_code(target_code)
                output_data["validation"] = {
                    "valid": val_report.valid,
                    "error_count": val_report.error_count,
                    "warning_count": val_report.warning_count,
                    "issues": [
                        {
                            "rule_id": i.rule_id,
                            "category": i.category,
                            "severity": i.severity,
                            "message": i.message,
                            "line": i.line_number,
                        }
                        for i in val_report.issues
                    ],
                }
                record_agent_finish(
                    "ValidatorAgent",
                    t0,
                    AgentExecutionStatus.COMPLETED if val_report.valid else AgentExecutionStatus.FAILED,
                    val_report.summary,
                    {"error_count": val_report.error_count},
                    None if val_report.valid else val_report.summary,
                )

                t1 = record_agent_start("ReviewerAgent", "Security audit")
                review_report = self._reviewer.review_code(target_code)
                output_data["review"] = {
                    "overall_score": review_report.overall_score,
                    "security_concerns": review_report.security_concerns,
                    "architectural_violations": review_report.architectural_violations,
                    "findings": [
                        _serialize_review_finding(f)
                        for f in review_report.findings
                    ],
                    "code_duplications": getattr(review_report, "code_duplications", []),
                }
                record_agent_finish(
                    "ReviewerAgent",
                    t1,
                    AgentExecutionStatus.COMPLETED,
                    f"Security score: {review_report.overall_score}/100",
                    {"score": review_report.overall_score},
                )

                total_dur = (time.perf_counter() - start_time) * 1000
                return OrchestrationResult(
                    mode=mode,
                    success=val_report.valid and review_report.overall_score >= 60,
                    summary=(
                        f"Audit passed: validation succeeded and review score "
                        f"{review_report.overall_score}/100 met the 60-point threshold."
                        if val_report.valid and review_report.overall_score >= 60
                        else (
                            f"Audit failed: validation failed and review score "
                            f"{review_report.overall_score}/100 was below the "
                            "60-point threshold."
                            if not val_report.valid and review_report.overall_score < 60
                            else (
                                f"Audit failed after validation failures; review score "
                                f"{review_report.overall_score}/100 met the threshold."
                                if not val_report.valid
                                else f"Audit failed: review score "
                                f"{review_report.overall_score}/100 was below "
                                "the 60-point threshold."
                            )
                        )
                    ),
                    agent_records=agent_records,
                    total_duration_ms=total_dur,
                    output_data=output_data,
                )

            # ------------------------------------------------------------------ #
            # MODE: TEST_GENERATION (Test Generator + Test Executor)
            # ------------------------------------------------------------------ #
            elif mode == WorkflowMode.TEST_GENERATION:
                target_code = code_input or ""
                t0 = record_agent_start("TestGeneratorAgent", prompt)
                test_report = self._test_generator.generate_tests(target_code)
                output_data["test_generation"] = {
                    "test_overview": test_report.test_overview,
                    "test_cases_count": len(test_report.test_cases),
                    "edge_cases_count": len(test_report.edge_cases),
                    "negative_cases_count": len(test_report.negative_cases),
                    "coverage_suggestions": test_report.coverage_suggestions,
                    "generated_test_code": test_report.generated_test_code,
                }
                record_agent_finish(
                    "TestGeneratorAgent",
                    t0,
                    AgentExecutionStatus.COMPLETED,
                    test_report.test_overview,
                    {
                        "total_tests": (
                            len(test_report.test_cases)
                            + len(test_report.edge_cases)
                            + len(test_report.negative_cases)
                        )
                    },
                )

                t1 = record_agent_start("TestExecutor", "Executing generated test suite")
                exec_res = self._test_executor.execute(
                    TestExecutionRequest(
                        generated_code=target_code,
                        generated_tests=test_report.generated_test_code,
                    )
                )
                output_data["test_execution"] = {
                    "success": exec_res.success,
                    "passed": exec_res.passed,
                    "failed": exec_res.failed,
                    "stdout": exec_res.stdout,
                    "stderr": exec_res.stderr,
                    "exit_code": exec_res.exit_code,
                    "duration_ms": exec_res.execution_time_ms,
                }
                record_agent_finish(
                    "TestExecutor",
                    t1,
                    AgentExecutionStatus.COMPLETED if exec_res.success else AgentExecutionStatus.FAILED,
                    f"Passed: {exec_res.passed}, Failed: {exec_res.failed}",
                    {"passed": exec_res.passed, "failed": exec_res.failed},
                    None if exec_res.success else (exec_res.stderr or exec_res.stdout or "Test execution failed"),
                )

                total_dur = (time.perf_counter() - start_time) * 1000
                return OrchestrationResult(
                    mode=mode,
                    success=exec_res.success,
                    summary=(
                        f"Tests generated and executed successfully ({exec_res.passed or 0} passed)."
                        if exec_res.success
                        else f"Generated tests failed during execution ({exec_res.failed or 0} failed)."
                    ),
                    agent_records=agent_records,
                    total_duration_ms=total_dur,
                    output_data=output_data,
                )

            # ------------------------------------------------------------------ #
            # MODE: DOCUMENTATION (Documentation Agent)
            # ------------------------------------------------------------------ #
            elif mode == WorkflowMode.DOCUMENTATION:
                target_code = code_input or ""
                t0 = record_agent_start("DocumentationAgent", prompt)
                doc_report = self._documentation.generate_documentation(target_code)
                output_data["documentation"] = {
                    "summary": doc_report.summary,
                    "module_description": doc_report.module_description,
                    "function_docs": doc_report.function_docs,
                    "class_docs": doc_report.class_docs,
                    "markdown_documentation": doc_report.markdown_documentation,
                }
                record_agent_finish(
                    "DocumentationAgent",
                    t0,
                    AgentExecutionStatus.COMPLETED,
                    doc_report.summary,
                    {"has_markdown": bool(doc_report.markdown_documentation)},
                )
                total_dur = (time.perf_counter() - start_time) * 1000
                return OrchestrationResult(
                    mode=mode,
                    success=True,
                    summary=doc_report.summary,
                    agent_records=agent_records,
                    total_duration_ms=total_dur,
                    output_data=output_data,
                )

            # ------------------------------------------------------------------ #
            # MODE: BUG_FIX (Debugger -> Validator)
            # ------------------------------------------------------------------ #
            elif mode == WorkflowMode.BUG_FIX:
                target_code = code_input or ""
                t0 = record_agent_start("DebuggerAgent", prompt)
                debug_rep = self._debugger.debug_code(target_code, reviewer_feedback=prompt)
                output_data["debug"] = {
                    "issue_detected": debug_rep.issue_detected,
                    "error_type": debug_rep.error_type,
                    "error_category": debug_rep.error_category,
                    "line_number": debug_rep.line_number,
                    "root_cause": debug_rep.root_cause,
                    "root_cause_status": getattr(debug_rep, "root_cause_status", "hypothesis"),
                    "evidence": getattr(debug_rep, "evidence", []),
                    "suggested_changes": debug_rep.suggested_changes,
                    "corrected_code": debug_rep.corrected_code,
                    "targeted_diff": debug_rep.targeted_diff,
                    "retry_count": debug_rep.retry_count,
                    "max_retries": debug_rep.max_retries,
                    "verification_status": getattr(debug_rep, "verification_status", "unverified"),
                    "confidence": debug_rep.confidence,
                    "explanation": debug_rep.explanation,
                    "final_summary": debug_rep.final_summary,
                }
                record_agent_finish(
                    "DebuggerAgent",
                    t0,
                    AgentExecutionStatus.COMPLETED,
                    debug_rep.final_summary,
                    {
                        "category": debug_rep.error_category,
                        "line": debug_rep.line_number,
                        "root_cause_status": getattr(
                            debug_rep, "root_cause_status", "hypothesis"
                        ),
                    },
                    None,
                )

                t1 = record_agent_start(
                    "ValidatorAgent", "Validate proposed corrected code"
                )
                val_rep = self._validator.validate_code(debug_rep.corrected_code)
                record_agent_finish(
                    "ValidatorAgent",
                    t1,
                    AgentExecutionStatus.COMPLETED if val_rep.valid else AgentExecutionStatus.FAILED,
                    val_rep.summary,
                    {"errors": val_rep.error_count},
                    None if val_rep.valid else val_rep.summary,
                )

                total_dur = (time.perf_counter() - start_time) * 1000
                return OrchestrationResult(
                    mode=mode,
                    success=val_rep.valid,
                    summary=(
                        (
                            f"No issue detected ({debug_rep.error_category}); "
                            "source passed validation."
                        )
                        if not debug_rep.issue_detected and val_rep.valid
                        else (
                            f"No issue detected ({debug_rep.error_category}), "
                            "but source failed validation."
                        )
                        if not debug_rep.issue_detected
                        else (
                            f"Proposed {debug_rep.error_category} correction passed "
                            "validation; it has not been executed."
                            if val_rep.valid
                            else (
                                f"Proposed {debug_rep.error_category} correction "
                                "failed validation."
                            )
                        )
                    ),
                    agent_records=agent_records,
                    total_duration_ms=total_dur,
                    output_data=output_data,
                )

            # ------------------------------------------------------------------ #
            # DEFAULT MODE: FEATURE_DEVELOPMENT (Full agent pipeline)
            # ------------------------------------------------------------------ #
            else:
                # 1. Planner Agent (PRESERVED)
                t0 = record_agent_start("PlannerAgent", prompt)
                plan_res = self._planner.execute(GeneratePlanRequest(prompt=prompt))
                output_data["plan"] = {
                    "problem_summary": plan_res.plan.problem_summary,
                    "modules": plan_res.plan.modules,
                    "functions": plan_res.plan.functions,
                    "classes": plan_res.plan.classes,
                    "algorithm": plan_res.plan.algorithm,
                    "edge_cases": plan_res.plan.edge_cases,
                }
                record_agent_finish(
                    "PlannerAgent",
                    t0,
                    AgentExecutionStatus.COMPLETED,
                    plan_res.plan.problem_summary,
                    {"modules_count": len(plan_res.plan.modules)},
                )

                # 2. Coder Agent
                t1 = record_agent_start("CoderAgent", "Generate implementation")
                code_output = self._coder.generate_code(
                    prompt, language=language
                )
                output_data["code"] = code_output
                record_agent_finish(
                    "CoderAgent",
                    t1,
                    AgentExecutionStatus.COMPLETED,
                    "Generated source implementation",
                    {"length": len(code_output)},
                )

                # 3. Validator Agent
                t2 = record_agent_start("ValidatorAgent", "Validate syntax & static rules")
                val_res = self._validator.validate_code(code_output)
                output_data["validation"] = {
                    "valid": val_res.valid,
                    "error_count": val_res.error_count,
                    "warning_count": val_res.warning_count,
                    "summary": val_res.summary,
                    "issues": [
                        {
                            "rule_id": issue.rule_id,
                            "category": issue.category,
                            "severity": issue.severity,
                            "message": issue.message,
                            "line": issue.line_number,
                        }
                        for issue in val_res.issues
                    ],
                }
                record_agent_finish(
                    "ValidatorAgent",
                    t2,
                    AgentExecutionStatus.COMPLETED if val_res.valid else AgentExecutionStatus.FAILED,
                    val_res.summary,
                    {"error_count": val_res.error_count},
                    None if val_res.valid else val_res.summary,
                )

                if not val_res.valid:
                    total_dur = (time.perf_counter() - start_time) * 1000
                    return OrchestrationResult(
                        mode=mode,
                        success=False,
                        summary=(
                            "Feature development stopped after validation failed; "
                            "tests and review were not run."
                        ),
                        agent_records=agent_records,
                        total_duration_ms=total_dur,
                        output_data=output_data,
                    )

                # 4. Test Generator
                t3 = record_agent_start("TestGeneratorAgent", "Generate pytest suite")
                tests_res = self._test_generator.generate_tests(code_output)
                output_data["tests"] = tests_res.generated_test_code
                record_agent_finish(
                    "TestGeneratorAgent",
                    t3,
                    AgentExecutionStatus.COMPLETED,
                    tests_res.test_overview,
                    {"cases": len(tests_res.test_cases)},
                )

                # 5. Test Executor
                t4 = record_agent_start("TestExecutor", "Run tests")
                test_run_res = self._test_executor.execute(
                    TestExecutionRequest(
                        generated_code=code_output,
                        generated_tests=tests_res.generated_test_code,
                    )
                )
                output_data["test_execution"] = {
                    "success": test_run_res.success,
                    "stdout": test_run_res.stdout,
                    "stderr": test_run_res.stderr,
                    "exit_code": test_run_res.exit_code,
                    "passed": test_run_res.passed,
                    "failed": test_run_res.failed,
                    "duration_ms": test_run_res.execution_time_ms,
                }
                record_agent_finish(
                    "TestExecutor",
                    t4,
                    AgentExecutionStatus.COMPLETED if test_run_res.success else AgentExecutionStatus.FAILED,
                    f"Passed: {test_run_res.passed}, Failed: {test_run_res.failed}",
                    {"passed": test_run_res.passed, "failed": test_run_res.failed},
                    None if test_run_res.success else (test_run_res.stderr or test_run_res.stdout or "Test execution failed"),
                )

                # 6. Reviewer Agent
                t5 = record_agent_start("ReviewerAgent", "Review quality and security")
                review_res = self._reviewer.review_code(
                    code_output,
                    generated_tests=tests_res.generated_test_code,
                    test_stdout=test_run_res.stdout,
                    test_stderr=test_run_res.stderr,
                    test_exit_code=test_run_res.exit_code,
                )
                output_data["review"] = {
                    "score": review_res.overall_score,
                    "findings": [f.message for f in review_res.findings],
                    "structured_findings": [
                        _serialize_review_finding(f) for f in review_res.findings
                    ],
                    "recommendations": review_res.recommendations,
                    "code_duplications": getattr(review_res, "code_duplications", []),
                }
                record_agent_finish(
                    "ReviewerAgent",
                    t5,
                    AgentExecutionStatus.COMPLETED,
                    f"Review score: {review_res.overall_score}/100",
                    {"score": review_res.overall_score},
                )

                total_dur = (time.perf_counter() - start_time) * 1000
                return OrchestrationResult(
                    mode=mode,
                    success=val_res.valid and test_run_res.success,
                    summary=(
                        f"Feature development completed: tests passed; "
                        f"review score {review_res.overall_score}/100."
                        if test_run_res.success
                        else (
                            f"Feature development validation passed, but tests "
                            f"failed; review score {review_res.overall_score}/100."
                        )
                    ),
                    agent_records=agent_records,
                    total_duration_ms=total_dur,
                    output_data=output_data,
                )

        except Exception as exc:
            active_step = next(iter(active_steps.items()), None)
            failed_step_name = active_step[1][0] if active_step is not None else "unknown"
            logger.error(
                "AdvancedOrchestrator failed during step %s (%s)",
                failed_step_name,
                type(exc).__name__,
            )
            if active_step is not None:
                t0, (name, input_summary, started_at) = active_step
                duration_ms = (time.perf_counter() - t0) * 1000
                agent_records.append(
                    AgentExecutionRecord(
                        agent_name=name,
                        status=AgentExecutionStatus.FAILED,
                        started_at=started_at,
                        completed_at=_now_iso(),
                        duration_ms=duration_ms,
                        input_summary=input_summary,
                        output_summary="Step failed",
                        error=str(exc),
                        details={},
                    )
                )
                active_steps.pop(t0, None)
                if event_emitter and event_emission_error is None:
                    try:
                        emit_step_event(
                            "step_failed",
                            {
                                "step": name,
                                "error": "Step failed. See the orchestration result for details.",
                            },
                        )
                    except Exception as event_exc:
                        exc = RuntimeError(
                            f"{exc}; failed to emit step failure event: {event_exc}"
                        )
            total_dur = (time.perf_counter() - start_time) * 1000
            return OrchestrationResult(
                mode=mode,
                success=False,
                summary=f"Workflow failed: {exc}",
                agent_records=agent_records,
                total_duration_ms=total_dur,
                error=(
                    f"{exc}; event emission failed: {event_emission_error}"
                    if event_emission_error
                    and event_emission_error not in str(exc)
                    else str(exc)
                ),
                output_data=output_data,
            )
