"""API schemas and serialization helpers for asynchronous workflow jobs.

These schemas define the public HTTP contract for:
- POST /workflow/jobs (creation and 202 Accepted response)
- GET /workflow/jobs/{job_id} (lifecycle status, timestamps, and serialized result)

Ensures that internal model details, raw prompts, API keys, and sensitive tokens
never leak through the public job API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class CreateWorkflowJobPayload(BaseModel):
    """Payload for submitting an asynchronous workflow job."""

    prompt: str = Field(..., description="Task prompt for the multi-agent workflow")
    repository_root: str | None = Field(default=None, description="Optional root directory of the repository")
    apply_repository_changes: bool = Field(default=False, description="Whether to apply repository changes directly")


class CreateJobResponse(BaseModel):
    """Immediate response returned with HTTP 202 Accepted."""

    job_id: str
    status: str = "queued"


class JobStatusApiResponse(BaseModel):
    """Current lifecycle status and result of an asynchronous workflow job."""

    job_id: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def serialize_workflow_result(result: Any) -> Any:
    """Safely serialize a WorkflowResult (or arbitrary job result) to a JSON-compatible dict."""
    if result is None:
        return None
    if isinstance(result, (dict, list, str, int, float, bool)):
        return result

    # Inspect WorkflowResult dataclass
    if hasattr(result, "workflow_status"):
        data: dict[str, Any] = {
            "success": getattr(result, "success", False),
            "workflow_status": (
                getattr(result.workflow_status, "value", str(result.workflow_status))
                if getattr(result, "workflow_status", None) is not None
                else "unknown"
            ),
            "generated_code": getattr(result, "generated_code", None),
            "generated_tests": getattr(result, "generated_tests", None),
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
            "error": getattr(result, "error", None),
            "test_error": getattr(result, "test_error", None),
            "documentation_error": getattr(result, "documentation_error", None),
        }

        planning = getattr(result, "planning", None)
        if planning is not None:
            data["planning"] = {
                "problem_summary": getattr(planning, "problem_summary", ""),
                "project_type": getattr(planning, "project_type", ""),
                "requirements": list(getattr(planning, "requirements", []) or []),
                "modules": list(getattr(planning, "modules", []) or []),
                "functions": list(getattr(planning, "functions", []) or []),
                "classes": list(getattr(planning, "classes", []) or []),
                "external_libraries": list(getattr(planning, "external_libraries", []) or []),
                "database_needed": getattr(planning, "database_needed", False),
                "api_needed": list(getattr(planning, "api_needed", []) or []),
                "algorithm": getattr(planning, "algorithm", ""),
                "edge_cases": list(getattr(planning, "edge_cases", []) or []),
                "estimated_complexity": getattr(planning, "estimated_complexity", ""),
                "future_improvements": list(getattr(planning, "future_improvements", []) or []),
            }
        else:
            data["planning"] = None

        execution = getattr(result, "execution", None)
        if execution is not None:
            data["execution"] = {
                "success": getattr(execution, "success", False),
                "stdout": getattr(execution, "stdout", ""),
                "stderr": getattr(execution, "stderr", ""),
                "execution_time_ms": getattr(execution, "execution_time_ms", 0.0),
                "exit_code": getattr(execution, "exit_code", 0),
            }
        else:
            data["execution"] = None

        test_exec = getattr(result, "test_execution", None)
        if test_exec is not None:
            data["test_execution"] = {
                "success": getattr(test_exec, "success", False),
                "stdout": getattr(test_exec, "stdout", ""),
                "stderr": getattr(test_exec, "stderr", ""),
                "execution_time_ms": getattr(test_exec, "execution_time_ms", 0.0),
                "exit_code": getattr(test_exec, "exit_code", 0),
                "passed": getattr(test_exec, "passed", None),
                "failed": getattr(test_exec, "failed", None),
            }
        else:
            data["test_execution"] = None

        review = getattr(result, "review", None)
        if review is not None:
            data["review"] = {
                "overall_score": getattr(review, "overall_score", 0),
                "strengths": list(getattr(review, "strengths", []) or []),
                "weaknesses": list(getattr(review, "weaknesses", []) or []),
                "pep8_issues": list(getattr(review, "pep8_issues", []) or []),
                "performance_suggestions": list(getattr(review, "performance_suggestions", []) or []),
                "security_concerns": list(getattr(review, "security_concerns", []) or []),
                "logic_issues": list(getattr(review, "logic_issues", []) or []),
                "maintainability": list(getattr(review, "maintainability", []) or []),
                "error_handling": list(getattr(review, "error_handling", []) or []),
                "recommendations": list(getattr(review, "recommendations", []) or []),
                "final_summary": getattr(review, "final_summary", ""),
            }
        else:
            data["review"] = None

        doc = getattr(result, "documentation", None)
        if doc is not None:
            data["documentation"] = {
                "summary": getattr(doc, "summary", ""),
                "module_description": getattr(doc, "module_description", ""),
                "function_docs": getattr(doc, "function_docs", []),
                "class_docs": getattr(doc, "class_docs", []),
                "usage_examples": getattr(doc, "usage_examples", []),
                "markdown_documentation": getattr(doc, "markdown_documentation", ""),
            }
        else:
            data["documentation"] = None

        iterations = getattr(result, "iterations", [])
        serialized_iterations = []
        for it in (iterations or []):
            it_data = {
                "iteration_number": getattr(it, "iteration_number", 1),
                "generated_code": getattr(it, "generated_code", ""),
                "generated_tests": getattr(it, "generated_tests", None),
                "test_error": getattr(it, "test_error", None),
                "review_error": getattr(it, "review_error", None),
            }
            if getattr(it, "execution", None) is not None:
                it_data["execution"] = {
                    "success": getattr(it.execution, "success", False),
                    "stdout": getattr(it.execution, "stdout", ""),
                    "stderr": getattr(it.execution, "stderr", ""),
                    "execution_time_ms": getattr(it.execution, "execution_time_ms", 0.0),
                    "exit_code": getattr(it.execution, "exit_code", 0),
                }
            if getattr(it, "test_execution", None) is not None:
                it_data["test_execution"] = {
                    "success": getattr(it.test_execution, "success", False),
                    "stdout": getattr(it.test_execution, "stdout", ""),
                    "stderr": getattr(it.test_execution, "stderr", ""),
                    "execution_time_ms": getattr(it.test_execution, "execution_time_ms", 0.0),
                    "exit_code": getattr(it.test_execution, "exit_code", 0),
                    "passed": getattr(it.test_execution, "passed", None),
                    "failed": getattr(it.test_execution, "failed", None),
                }
            if getattr(it, "review", None) is not None:
                it_data["review"] = {
                    "overall_score": getattr(it.review, "overall_score", 0),
                    "final_summary": getattr(it.review, "final_summary", ""),
                }
            serialized_iterations.append(it_data)
        data["iterations"] = serialized_iterations

        return data

    return str(result)
