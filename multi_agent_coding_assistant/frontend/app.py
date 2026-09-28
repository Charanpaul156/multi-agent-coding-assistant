"""Streamlit frontend entrypoint.

UI for the Multi-Agent Coding Assistant.
"""

from __future__ import annotations

import logging
import re
import time

import requests
import streamlit as st

from frontend.workflow_ui import (
    WorkflowProgressState,
    format_progress_timeline,
    iter_sse_events,
    map_workflow_event,
    poll_workflow_job,
    sanitize_error_message,
)

logger = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Multi-Agent Coding Assistant", layout="wide")
st.title("Multi-Agent Coding Assistant")


def _normalize_newlines_for_display(text: str) -> str:
    """Normalize only newline characters for display.

    The API returns JSON strings; this function avoids aggressive unescaping
    and only converts literal escaped-newline sequences into real newlines.
    """

    # If the backend already returned real newlines, keep them unchanged.
    # Only handle the case where the string contains literal backslash-n.
    if "\\n" in text or "\\r" in text:
        text = text.replace("\\r\\n", "\r\n")
        text = text.replace("\\n", "\n")
        text = text.replace("\\r", "\r")

    # Ensure consistent line endings (Windows-style) for display/pasting.
    text = re.sub(r"\r?\n", "\n", text)
    return text


with st.sidebar:
    st.subheader("Backend")
    if st.button("Check Backend Health"):
        try:
            resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except Exception:
                    payload = resp.text
                st.success(f"Backend status: {payload}")
            else:
                st.error(
                    f"Backend returned status {resp.status_code}: {resp.text}"
                )
        except requests.exceptions.RequestException as exc:
            st.error(f"Failed to reach backend: {exc}")
        except Exception as exc:  # pragma: no cover
            st.error(f"Health check error: {exc}")


st.subheader("Planning")
plan_prompt = st.text_area(
    "Planning request",
    height=120,
    placeholder="Build a Banking Management System",
)

if "plan" not in st.session_state:
    st.session_state.plan = None

if st.button("Generate Plan"):
    if not plan_prompt.strip():
        st.error("Prompt must not be empty")
    else:
        try:
            with st.spinner("Generating implementation plan..."):
                resp = requests.post(
                    f"{BACKEND_URL}/generate-plan",
                    json={"prompt": plan_prompt},
                    timeout=120,
                )

            if resp.status_code != 200:
                st.error(f"Request failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.session_state.plan = data.get("plan", None)
                st.success("Plan generated")
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to generate plan: {exc}")

if st.session_state.plan:
    st.subheader("Implementation Plan")
    plan = st.session_state.plan

    with st.expander("Problem Summary"):
        st.write(plan.get("problem_summary", ""))

    with st.expander("Project Type"):
        st.write(plan.get("project_type", ""))

    with st.expander("Requirements"):
        st.write("\n".join(plan.get("requirements", []) or []))

    with st.expander("Modules"):
        st.write("\n".join(plan.get("modules", []) or []))

    with st.expander("Functions"):
        st.write("\n".join(plan.get("functions", []) or []))

    with st.expander("Classes"):
        st.write("\n".join(plan.get("classes", []) or []))

    with st.expander("Libraries"):
        st.write("\n".join(plan.get("external_libraries", []) or []))

    with st.expander("Database Needed"):
        st.write(plan.get("database_needed", False))

    with st.expander("API Requirements"):
        st.write("\n".join(plan.get("api_needed", []) or []))

    with st.expander("Algorithm"):
        st.write(plan.get("algorithm", ""))

    with st.expander("Edge Cases"):
        st.write("\n".join(plan.get("edge_cases", []) or []))

    with st.expander("Estimated Complexity"):
        st.write(plan.get("estimated_complexity", ""))

    with st.expander("Future Improvements"):
        st.write("\n".join(plan.get("future_improvements", []) or []))


st.subheader("Generate Python code")
prompt = st.text_area(
    "Programming request",
    height=150,
    placeholder="Create a Python calculator using functions",
)

code_placeholder = ""
if "generated_code" not in st.session_state:
    st.session_state.generated_code = code_placeholder

if st.button("Generate Code"):
    if not prompt.strip():
        st.error("Prompt must not be empty")
    else:
        try:
            with st.spinner("Generating code..."):
                resp = requests.post(
                    f"{BACKEND_URL}/generate-code",
                    json={"prompt": prompt},
                    timeout=120,
                )

            if resp.status_code != 200:
                st.error(f"Request failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                generated_code = data.get("generated_code", "")
                st.session_state.generated_code = _normalize_newlines_for_display(
                    generated_code
                )
                st.success("Code generated")
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to generate code: {exc}")


st.subheader("Generated Python code")
if st.session_state.generated_code:
    st.code(st.session_state.generated_code, language="python")

    st.download_button(
        label="Download Python File",
        data=st.session_state.generated_code,
        file_name="generated_code.py",
        mime="text/x-python",
    )

    st.caption("Tip: select the code block and copy (Ctrl+C / Cmd+C).")

    if st.button("Execute Code"):
        try:
            with st.spinner("Executing code..."):
                resp = requests.post(
                    f"{BACKEND_URL}/execute-code",
                    json={"generated_code": st.session_state.generated_code},
                    timeout=30,
                )

            if resp.status_code != 200:
                st.error(f"Execution failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.session_state.execution = data
                st.subheader("Execution Output")

                st.write(f"Exit Code: {data.get('exit_code', '')}")
                st.write(f"Execution Time (ms): {data.get('execution_time_ms', '')}")

                stdout = data.get("stdout", "")
                stderr = data.get("stderr", "")

                if stdout:
                    st.code(stdout, language="text")

                if stderr:
                    st.subheader("Errors")
                    st.code(stderr, language="text")
                else:
                    st.caption("No errors.")
        except Exception as exc:  # pragma: no cover
            st.error(f"Execution error: {exc}")


st.subheader("Review Code")
if "review" not in st.session_state:
    st.session_state.review = None

if st.session_state.generated_code:
    if st.button("Review Code"):
        try:
            with st.spinner("Reviewing code..."):
                review_payload = {"generated_code": st.session_state.generated_code}
                if "execution" in st.session_state:
                    exec_data = st.session_state.execution
                    review_payload["stdout"] = exec_data.get("stdout", "")
                    review_payload["stderr"] = exec_data.get("stderr", "")
                    review_payload["exit_code"] = exec_data.get("exit_code", None)

                resp = requests.post(
                    f"{BACKEND_URL}/review-code",
                    json=review_payload,
                    timeout=120,
                )

            if resp.status_code != 200:
                st.error(f"Review failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.session_state.review = data.get("review", None)
                st.success("Review complete")
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to review code: {exc}")

    if st.session_state.review:
        review = st.session_state.review
        st.metric("Overall Score", review.get("overall_score", "N/A"))

        with st.expander("Strengths"):
            st.write("\n".join(review.get("strengths", []) or []) or "None")

        with st.expander("Weaknesses"):
            st.write("\n".join(review.get("weaknesses", []) or []) or "None")

        with st.expander("PEP8 Issues"):
            st.write("\n".join(review.get("pep8_issues", []) or []) or "None")

        with st.expander("Performance Suggestions"):
            st.write("\n".join(review.get("performance_suggestions", []) or []) or "None")

        with st.expander("Security Concerns"):
            st.write("\n".join(review.get("security_concerns", []) or []) or "None")

        with st.expander("Logic Issues"):
            st.write("\n".join(review.get("logic_issues", []) or []) or "None")

        with st.expander("Maintainability"):
            st.write("\n".join(review.get("maintainability", []) or []) or "None")

        with st.expander("Error Handling"):
            st.write("\n".join(review.get("error_handling", []) or []) or "None")

        with st.expander("Recommendations"):
            st.write("\n".join(review.get("recommendations", []) or []) or "None")

        st.subheader("Final Summary")
        st.write(review.get("final_summary", ""))


st.subheader("Generate Tests")
if "tests_report" not in st.session_state:
    st.session_state.tests_report = None

if st.session_state.generated_code:
    if st.button("Generate Tests"):
        try:
            with st.spinner("Generating pytest test suite..."):
                resp = requests.post(
                    f"{BACKEND_URL}/generate-tests",
                    json={"generated_code": st.session_state.generated_code},
                    timeout=120,
                )

            if resp.status_code != 200:
                st.error(f"Test generation failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.session_state.tests_report = data.get("report", None)
                st.success("Test suite generated")
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to generate tests: {exc}")

if st.session_state.tests_report:
    report = st.session_state.tests_report

    st.subheader("Test Overview")
    st.write(report.get("test_overview", ""))

    st.write(f"Test Framework: {report.get('test_framework', '')}")

    def _render_test_cases(cases, title):
        if not cases:
            return
        with st.expander(title):
            for case in cases:
                st.markdown(f"**{case.get('name', '')}**")
                st.write(case.get("description", ""))
                st.markdown(f"- Input example: `{case.get('input_example', '')}`")
                st.write(f"Expected behavior: {case.get('expected_behavior', '')}")

    _render_test_cases(report.get("test_cases", []), "Test Cases")
    _render_test_cases(report.get("edge_cases", []), "Edge Cases")
    _render_test_cases(report.get("negative_cases", []), "Negative/Error Cases")

    with st.expander("Coverage Suggestions"):
        st.write("\n".join(report.get("coverage_suggestions", []) or []) or "None")

    st.subheader("Generated pytest code")
    test_code = _normalize_newlines_for_display(report.get("generated_test_code", ""))
    st.code(test_code, language="python")

    st.download_button(
        label="Download tests_generated.py",
        data=test_code,
        file_name="tests_generated.py",
        mime="text/x-python",
    )

    st.subheader("Final Summary")
    st.write(report.get("final_summary", ""))


st.subheader("Debug Code")
if "debug_report" not in st.session_state:
    st.session_state.debug_report = None

if st.session_state.generated_code:
    if st.button("Debug Code"):
        try:
            with st.spinner("Debugging code..."):
                debug_payload = {"generated_code": st.session_state.generated_code}
                if st.session_state.tests_report:
                    debug_payload["generated_tests"] = (
                        st.session_state.tests_report.get("generated_test_code", "")
                    )
                if "execution" in st.session_state:
                    exec_data = st.session_state.execution
                    debug_payload["execution_stdout"] = exec_data.get("stdout", "")
                    debug_payload["execution_stderr"] = exec_data.get("stderr", "")
                    debug_payload["execution_exit_code"] = exec_data.get("exit_code", None)
                if st.session_state.review:
                    debug_payload["reviewer_feedback"] = st.session_state.review.get(
                        "final_summary", ""
                    )

                resp = requests.post(
                    f"{BACKEND_URL}/debug-code",
                    json=debug_payload,
                    timeout=120,
                )

            if resp.status_code != 200:
                st.error(f"Debug failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.session_state.debug_report = data.get("debug_report", None)
                st.success("Debugging complete")
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to debug code: {exc}")

if st.session_state.debug_report:
    debug = st.session_state.debug_report
    st.write(f"Issue Detected: {debug.get('issue_detected', 'N/A')}")
    st.write(f"Error Type: {debug.get('error_type', '')}")
    st.write(f"Root Cause: {debug.get('root_cause', '')}")
    st.write(f"Affected Component: {debug.get('affected_component', '')}")
    st.write(f"Confidence: {debug.get('confidence', 'N/A')}")

    with st.expander("Explanation"):
        st.write(debug.get("explanation", ""))

    with st.expander("Suggested Changes"):
        st.write("\n".join(debug.get("suggested_changes", []) or []) or "None")

    if debug.get("corrected_code"):
        st.subheader("Corrected Code")
        st.code(
            _normalize_newlines_for_display(debug.get("corrected_code", "")),
            language="python",
        )

    with st.expander("Final Summary"):
        st.write(debug.get("final_summary", ""))


st.subheader("Repository RAG")
st.caption(
    "Index a repository and search it to retrieve relevant code context. "
    "The repository path must be inside a configured allowed root."
)

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

repo_path = st.text_input(
    "Repository path",
    placeholder=r"C:\path\to\repository",
)

if st.button("Index Repository"):
    if not repo_path.strip():
        st.error("Repository path must not be empty")
    else:
        try:
            with st.spinner("Indexing repository..."):
                resp = requests.post(
                    f"{BACKEND_URL}/index-repository",
                    json={"repository_path": repo_path.strip()},
                    timeout=120,
                )
            if resp.status_code != 200:
                st.error(f"Indexing failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.session_state.rag_index = data
                if data.get("success"):
                    st.success(
                        f"Indexed {data.get('file_count', 0)} files / "
                        f"{data.get('chunk_count', 0)} chunks"
                    )
                else:
                    st.warning(data.get("error", "Indexing incomplete"))
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to index repository: {exc}")

if st.button("Show RAG Status"):
    try:
        with st.spinner("Fetching RAG status..."):
            resp = requests.get(f"{BACKEND_URL}/rag/status", timeout=30)
        if resp.status_code != 200:
            st.error(f"Status failed: {resp.status_code} - {resp.text}")
        else:
            data = resp.json()
            st.session_state.rag_status = data
            st.write(f"Chunks indexed: {data.get('chunk_count', 0)}")
            st.write(f"Repositories: {', '.join(data.get('repositories', []) or []) or 'none'}")
    except Exception as exc:  # pragma: no cover
        st.error(f"Failed to fetch RAG status: {exc}")

st.markdown("### Search Repository")
search_query = st.text_area(
    "Search query",
    height=80,
    placeholder="Find where authentication is implemented",
)

if "rag_results" not in st.session_state:
    st.session_state.rag_results = []

if st.button("Search Repository"):
    if not search_query.strip():
        st.error("Search query must not be empty")
    else:
        try:
            with st.spinner("Searching repository..."):
                resp = requests.post(
                    f"{BACKEND_URL}/search-repository",
                    json={"query": search_query.strip()},
                    timeout=60,
                )
            if resp.status_code != 200:
                st.error(f"Search failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.session_state.rag_results = data.get("results", [])
                st.success(f"Found {len(st.session_state.rag_results)} result(s)")
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to search repository: {exc}")

if st.session_state.rag_results:
    for idx, item in enumerate(st.session_state.rag_results):
        title = f"{item.get('file_path', '?')} (lines {item.get('start_line', '?')}-{item.get('end_line', '?')})"
        with st.expander(title):
            if item.get("distance") is not None:
                st.write(f"Distance: {item['distance']:.4f}")
            st.code(
                _normalize_newlines_for_display(item.get("content", "")),
                language="python",
            )


st.subheader("Modify Repository")
st.caption(
    "Repository-aware code modification. Changes are generated, validated, "
    "and shown as a diff before being applied. Use the explicit Apply action "
    "to write changes to the repository. The repository path must be inside "
    "a configured allowed root."
)

if "modify_repository_result" not in st.session_state:
    st.session_state.modify_repository_result = None

if "modify_repository_proposal" not in st.session_state:
    st.session_state.modify_repository_proposal = None

mod_repo_path = st.text_input(
    "Repository",
    placeholder=r"C:\path\to\repository",
    key="mod_repo_path",
)

mod_request = st.text_area(
    "User request",
    height=100,
    placeholder="Add password reset functionality to this project",
    key="mod_request",
)

if st.button("Generate Changes"):
    if not mod_repo_path.strip():
        st.error("Repository path must not be empty")
    elif not mod_request.strip():
        st.error("Request must not be empty")
    else:
        try:
            with st.spinner(
                "Retrieving context -> planning -> generating changes -> validating..."
            ):
                resp = requests.post(
                    f"{BACKEND_URL}/modify-repository",
                    json={
                        "repository": mod_repo_path.strip(),
                        "request": mod_request.strip(),
                        "dry_run": True,
                        "apply": False,
                    },
                    timeout=300,
                )
            if resp.status_code != 200:
                st.error(f"Modification failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.session_state.modify_repository_result = data
                st.session_state.modify_repository_proposal = {
                    "repository": mod_repo_path.strip(),
                    "request": mod_request.strip(),
                }
                if data.get("success", False):
                    st.success("Proposed changes generated")
                else:
                    st.warning(
                        data.get("error", "Modification did not complete")
                    )
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to modify repository: {exc}")

def _render_modify_repository_result(result: dict | None) -> None:
    if not result:
        return

    approval_status = result.get(
        "approval_status",
        "preview" if result.get("dry_run", True) else "applied",
    )
    st.write(f"Approval status: **{str(approval_status).upper()}**")
    st.write(
        "Application status: "
        + (
            "proposal generated (preview)"
            if result.get("dry_run", True)
            else ("applied" if result.get("success", False) else "failed")
        )
    )
    st.write(f"Dry run: {result.get('dry_run', False)}")

    if approval_status == "preview":
        st.info("⚠️ Changes are in PREVIEW mode. NO repository files have been written. Explicit approval is required.")
    elif approval_status == "rejected":
        st.warning("❌ Proposal was rejected. No files were written to the repository.")
    elif approval_status in ("approved", "applied"):
        st.success("✅ Changes have been applied to the repository.")

    if result.get("error"):
        st.error(result.get("error"))

    validation_results = result.get("validation_results", []) or []
    if validation_results:
        st.subheader("Validation Results")
        for item in validation_results:
            title = f"{item.get('file_path', '?')} - {item.get('operation', '?')}"
            if item.get("valid"):
                st.success(title)
            else:
                st.error(title)
            if item.get("messages"):
                for message in item.get("messages", []):
                    st.write(f"- {message}")

    proposed_changes = result.get("proposed_changes", []) or []
    if proposed_changes:
        st.subheader("Proposed Changes")
        for change in proposed_changes:
            title = f"{change.get('operation', '')}: {change.get('file_path', '')}"
            with st.expander(title, expanded=False):
                st.write(f"File: {change.get('file_path', '')}")
                st.write(f"Operation: {change.get('operation', '')}")
                st.write(f"Description: {change.get('description', '') or 'None'}")
                if change.get("original_hash"):
                    st.write(f"Original hash: {change.get('original_hash')}")
                if change.get("new_content"):
                    st.code(
                        _normalize_newlines_for_display(change.get("new_content", "")),
                        language="python",
                    )

    diff_entries = result.get("diff", []) or []
    if diff_entries:
        st.subheader("Diff Preview")
        for entry in diff_entries:
            with st.expander(
                f"{entry.get('operation', '')}: {entry.get('file_path', '')}",
                expanded=False,
            ):
                st.code(
                    _normalize_newlines_for_display(entry.get("diff_text", "")),
                    language="diff",
                )

    applied_files = result.get("applied_files", []) or []
    st.subheader("Application Status")
    if result.get("success", False):
        if result.get("dry_run", True):
            st.info("Dry-run preview completed successfully. Repository files remain untouched.")
        else:
            st.success("Changes applied successfully.")
    else:
        st.warning("Repository modification did not complete successfully.")

    if applied_files:
        st.write("Applied files:")
        for file_path in applied_files:
            st.write(f"- {file_path}")


_render_modify_repository_result(st.session_state.modify_repository_result)

st.markdown("---")
st.markdown("### Human Approval Gate")
st.caption(
    "Repository modifications require explicit human approval. "
    "NO repository files are written until approved."
)

res = st.session_state.modify_repository_result
token = res.get("approval_token") if res else None
repo = (
    st.session_state.modify_repository_proposal.get("repository")
    if st.session_state.modify_repository_proposal
    else None
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Approve Changes", type="primary"):
        if not token:
            st.error("No active proposal to approve. Generate changes first.")
        else:
            try:
                with st.spinner("Applying approved changes to repository..."):
                    resp = requests.post(
                        f"{BACKEND_URL}/modify-repository/approve",
                        json={
                            "approval_token": token,
                            "repository": repo,
                        },
                        timeout=300,
                    )
                if resp.status_code != 200:
                    st.error(f"Approval failed: {resp.status_code} - {resp.text}")
                else:
                    data = resp.json()
                    st.session_state.modify_repository_result = data
                    if data.get("success", False):
                        st.success("Approved changes applied successfully!")
                    else:
                        st.error(data.get("error", "Application failed after approval."))
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to approve changes: {exc}")

with col2:
    if st.button("Reject Changes"):
        if not token:
            st.error("No active proposal to reject. Generate changes first.")
        else:
            try:
                with st.spinner("Rejecting proposed changes..."):
                    resp = requests.post(
                        f"{BACKEND_URL}/modify-repository/reject",
                        json={
                            "approval_token": token,
                            "repository": repo,
                        },
                        timeout=60,
                    )
                if resp.status_code != 200:
                    st.error(f"Rejection failed: {resp.status_code} - {resp.text}")
                else:
                    data = resp.json()
                    st.session_state.modify_repository_result = data
                    st.warning("Proposed changes rejected. Filesystem remains untouched.")
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to reject changes: {exc}")


st.subheader("Run Complete Workflow")
workflow_prompt = st.text_area(
    "Workflow request",
    height=120,
    placeholder="Build a Student Management System",
)

if "workflow" not in st.session_state:
    st.session_state.workflow = None

if st.button("Run Complete Workflow"):
    if not workflow_prompt.strip():
        st.error("Prompt must not be empty")
    else:
        try:
            with st.spinner(
                "Running full workflow (Planner -> Coder -> Test Generator -> "
                "Executor -> Test Executor -> Reviewer -> Debugger)..."
            ):
                resp = requests.post(
                    f"{BACKEND_URL}/run-workflow",
                    json={"prompt": workflow_prompt},
                    timeout=300,
                )

            if resp.status_code != 200:
                st.error(f"Workflow failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.session_state.workflow = data.get("workflow", None)
                if data.get("success", False):
                    st.success("Workflow completed")
                    st.session_state.workflow_code = data["workflow"].get(
                        "generated_code", ""
                    )
                else:
                    st.warning("Workflow did not complete")
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to run workflow: {exc}")


def _render_execution(exec_obj, title):
    """Render an execution/test-execution object inside an expander."""
    if not exec_obj:
        return
    with st.expander(title):
        st.write(f"Exit Code: {exec_obj.get('exit_code', '')}")
        st.write(f"Execution Time (ms): {exec_obj.get('execution_time_ms', '')}")
        passed = exec_obj.get("passed")
        failed = exec_obj.get("failed")
        if passed is not None:
            st.write(f"Passed: {passed}")
        if failed is not None:
            st.write(f"Failed: {failed}")
        if exec_obj.get("stdout"):
            st.code(exec_obj.get("stdout"), language="text")
        if exec_obj.get("stderr"):
            st.code(exec_obj.get("stderr"), language="text")


def _render_workflow_result(wf: dict | None, title_prefix: str = "Workflow") -> None:
    """Render full workflow result object including plan, code, tests, review, doc, and iterations."""
    if not wf:
        return

    st.write(f"Workflow Status: {wf.get('workflow_status', '')}")
    st.write(f"Execution Time (ms): {wf.get('execution_time_ms', '')}")

    if wf.get("error"):
        st.error(sanitize_error_message(str(wf.get("error"))))

    if wf.get("test_error"):
        st.warning(f"Test generation issue: {wf.get('test_error')}")

    # 1. Implementation Plan
    wf_plan = wf.get("planning")
    if wf_plan:
        with st.expander(f"{title_prefix} Plan"):
            st.write(f"Problem Summary: {wf_plan.get('problem_summary', '')}")
            st.write(f"Project Type: {wf_plan.get('project_type', '')}")
            st.write("\n".join(wf_plan.get("requirements", []) or []))
            st.write("\n".join(wf_plan.get("modules", []) or []))

    # 2. Final Generated Code
    wf_code = wf.get("generated_code")
    if wf_code:
        st.subheader(f"{title_prefix} Final Generated Code")
        norm_code = _normalize_newlines_for_display(wf_code)
        st.code(norm_code, language="python")
        st.download_button(
            label=f"Download {title_prefix.lower()}_generated_code.py",
            data=norm_code,
            file_name=f"{title_prefix.lower()}_generated_code.py",
            mime="text/x-python",
            key=f"dl_code_{title_prefix.lower()}_{hash(title_prefix)}",
        )

    # 3. Final Generated Tests
    wf_tests = wf.get("generated_tests")
    if wf_tests:
        st.subheader(f"{title_prefix} Final Generated Tests")
        norm_tests = _normalize_newlines_for_display(wf_tests)
        st.code(norm_tests, language="python")
        st.download_button(
            label=f"Download {title_prefix.lower()}_tests_generated.py",
            data=norm_tests,
            file_name=f"{title_prefix.lower()}_tests_generated.py",
            mime="text/x-python",
            key=f"dl_tests_{title_prefix.lower()}_{hash(title_prefix)}",
        )

    # 4. Final Application Execution
    _render_execution(wf.get("execution"), f"{title_prefix} Final Application Execution")

    # 5. Final Test Execution
    _render_execution(wf.get("test_execution"), f"{title_prefix} Final Test Execution")

    # 6. Final Reviewer Report
    wf_review = wf.get("review")
    if wf_review:
        with st.expander(f"{title_prefix} Final Review"):
            st.metric("Overall Score", wf_review.get("overall_score", "N/A"))
            st.write("\n".join(wf_review.get("strengths", []) or []))
            st.write("\n".join(wf_review.get("recommendations", []) or []))

    # 7. Documentation
    wf_doc = wf.get("documentation")
    if wf_doc:
        with st.expander(f"{title_prefix} Generated Documentation"):
            st.markdown(wf_doc.get("markdown_documentation", ""))

    # 8. Iteration history
    iterations = wf.get("iterations") or []
    if iterations:
        st.subheader(f"{title_prefix} Debugging Iterations")
        for iteration in iterations:
            iter_num = iteration.get("iteration_number", "?")
            iter_label = f"Iteration {iter_num}"
            with st.expander(iter_label):
                iter_code = iteration.get("generated_code")
                if iter_code:
                    st.write("Generated Code")
                    st.code(
                        _normalize_newlines_for_display(iter_code),
                        language="python",
                    )

                iter_tests = iteration.get("generated_tests")
                if iter_tests:
                    st.write("Generated Tests")
                    st.code(
                        _normalize_newlines_for_display(iter_tests),
                        language="python",
                    )

                _render_execution(
                    iteration.get("execution"),
                    f"Application Execution (Iteration {iter_num})",
                )
                _render_execution(
                    iteration.get("test_execution"),
                    f"Test Execution (Iteration {iter_num})",
                )

                iter_review = iteration.get("review")
                if iter_review:
                    with st.expander(f"Review (Iteration {iter_num})"):
                        st.metric(
                            "Overall Score",
                            iter_review.get("overall_score", "N/A"),
                        )
                        st.write("\n".join(iter_review.get("logic_issues", []) or []))
                        st.write("\n".join(iter_review.get("recommendations", []) or []))

                iter_debug = iteration.get("debug_report")
                if iter_debug:
                    with st.expander(f"Debug Report (Iteration {iter_num})"):
                        st.write(
                            f"Issue Detected: {iter_debug.get('issue_detected', 'N/A')}"
                        )
                        st.write(f"Error Type: {iter_debug.get('error_type', '')}")
                        st.write(f"Root Cause: {iter_debug.get('root_cause', '')}")
                        st.write(
                            f"Affected Component: "
                            f"{iter_debug.get('affected_component', '')}"
                        )
                        st.write(f"Confidence: {iter_debug.get('confidence', 'N/A')}")
                        if iter_debug.get("corrected_code"):
                            st.write("Corrected Code")
                            st.code(
                                _normalize_newlines_for_display(
                                    iter_debug.get("corrected_code", "")
                                ),
                                language="python",
                            )
                        st.write(
                            f"Summary: {iter_debug.get('final_summary', '')}"
                        )


if st.session_state.workflow:
    _render_workflow_result(st.session_state.workflow, title_prefix="Workflow")


# ---------------------------------------------------------------------------
# Asynchronous Workflow with Real-Time SSE Streaming
# ---------------------------------------------------------------------------

st.subheader("Run Asynchronous Workflow (Live Streaming)")
st.caption(
    "Submit an asynchronous background workflow and monitor real-time agent "
    "progress via Server-Sent Events (SSE). The UI updates dynamically without blocking."
)

async_workflow_prompt = st.text_area(
    "Asynchronous workflow request",
    height=120,
    placeholder="Create a Python function called multiply that takes two numbers and returns their product.",
    key="async_wf_prompt",
)

if "async_job_id" not in st.session_state:
    st.session_state.async_job_id = None
if "async_workflow_status" not in st.session_state:
    st.session_state.async_workflow_status = None
if "async_progress_state" not in st.session_state:
    st.session_state.async_progress_state = None
if "async_workflow_result" not in st.session_state:
    st.session_state.async_workflow_result = None
if "async_event_log" not in st.session_state:
    st.session_state.async_event_log = []

col_async_btn, _ = st.columns([1, 4])
with col_async_btn:
    run_async_clicked = st.button("Run Async Workflow", type="primary")

if run_async_clicked:
    if not async_workflow_prompt.strip():
        st.error("Prompt must not be empty")
    else:
        st.session_state.async_workflow_result = None
        st.session_state.async_event_log = []
        progress_state = WorkflowProgressState()
        st.session_state.async_progress_state = progress_state

        status_box = st.empty()
        timeline_box = st.empty()
        metrics_box = st.empty()
        activity_box = st.empty()

        status_box.info("Submitting asynchronous workflow job...")

        try:
            submit_resp = requests.post(
                f"{BACKEND_URL}/workflow/jobs",
                json={"prompt": async_workflow_prompt.strip()},
                timeout=15,
            )
            if submit_resp.status_code != 202:
                status_box.error(
                    f"Job submission failed: {submit_resp.status_code} - "
                    f"{sanitize_error_message(submit_resp.text)}"
                )
            else:
                submit_data = submit_resp.json()
                job_id = submit_data.get("job_id", "")
                st.session_state.async_job_id = job_id
                progress_state.job_id = job_id
                progress_state.status = submit_data.get("status", "queued")
                status_box.success(f"Job accepted! Job ID: `{job_id}`")

                def _render_live_ui():
                    timeline_box.markdown(format_progress_timeline(progress_state))
                    m1, m2, m3, m4 = metrics_box.columns(4)
                    m1.metric("Status", progress_state.status.title())
                    m2.metric("Iteration", f"{progress_state.iteration} / {progress_state.max_iterations}")
                    t_str = (
                        f"{progress_state.tests_passed} pass / {progress_state.tests_failed} fail"
                        if progress_state.tests_passed is not None
                        else "Pending"
                    )
                    m3.metric("Tests", t_str)
                    r_str = (
                        f"{progress_state.reviewer_score}/100"
                        if progress_state.reviewer_score is not None
                        else "Pending"
                    )
                    m4.metric("Reviewer Score", r_str)

                _render_live_ui()

                # Stream from SSE
                stream_url = f"{BACKEND_URL}/workflow/jobs/{job_id}/stream"
                status_box.info(f"Connecting to live event stream: `{job_id}` ...")

                sse_succeeded = False
                try:
                    with requests.get(stream_url, stream=True, timeout=300) as sse_resp:
                        if sse_resp.status_code == 200:
                            sse_succeeded = True
                            status_box.info("Live stream connected. Tracking progress in real time...")
                            for parsed_ev in iter_sse_events(sse_resp.iter_lines(decode_unicode=True)):
                                ev_type = parsed_ev.event_type
                                payload = (
                                    parsed_ev.data.get("payload", {})
                                    if isinstance(parsed_ev.data, dict)
                                    else {}
                                )
                                log_msg = map_workflow_event(ev_type, payload, progress_state)
                                timestamp_str = time.strftime("%H:%M:%S")
                                st.session_state.async_event_log.append(f"[{timestamp_str}] {log_msg}")

                                _render_live_ui()
                                activity_box.caption(f"Latest activity: **{log_msg}**")

                                if progress_state.is_terminal:
                                    break
                except Exception as sse_exc:
                    logger.warning("SSE stream connection issue: %s", sse_exc)

                # Fallback polling if SSE disconnected early or failed
                if not progress_state.is_terminal:
                    status_box.warning("Live stream disconnected. Polling background job status...")

                    def _poll_cb(poll_data):
                        st_name = poll_data.get("status", "running")
                        progress_state.status = st_name
                        status_box.info(f"Polling job status: **{st_name}** ...")
                        _render_live_ui()

                    poll_result = poll_workflow_job(
                        BACKEND_URL,
                        job_id,
                        max_wait_seconds=180.0,
                        poll_interval=1.5,
                        on_poll_callback=_poll_cb,
                    )
                    if poll_result:
                        p_status = poll_result.get("status")
                        if p_status == "completed":
                            progress_state.status = "completed"
                            progress_state.is_terminal = True
                        elif p_status in ("failed", "cancelled"):
                            progress_state.status = "failed"
                            progress_state.is_terminal = True
                            progress_state.error = sanitize_error_message(
                                poll_result.get("error", "Job failed")
                            )

                # Retrieve complete final job result
                try:
                    final_resp = requests.get(f"{BACKEND_URL}/workflow/jobs/{job_id}", timeout=15)
                    if final_resp.status_code == 200:
                        final_data = final_resp.json()
                        st.session_state.async_workflow_status = final_data.get("status")
                        st.session_state.async_workflow_result = final_data.get("result")
                except Exception as final_exc:
                    logger.error("Failed to retrieve final job result: %s", final_exc)

                _render_live_ui()
                if progress_state.status == "completed":
                    status_box.success("Asynchronous workflow completed successfully!")
                else:
                    status_box.error(
                        f"Asynchronous workflow failed: {progress_state.error or 'Execution failed'}"
                    )
        except Exception as exc:
            st.error(f"Error executing asynchronous workflow: {sanitize_error_message(str(exc))}")

# Render active state and final results on reruns
if st.session_state.async_progress_state and not run_async_clicked:
    p_state = st.session_state.async_progress_state
    st.markdown("#### Workflow Timeline")
    st.markdown(format_progress_timeline(p_state))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", p_state.status.title())
    m2.metric("Iteration", f"{p_state.iteration} / {p_state.max_iterations}")
    t_str = (
        f"{p_state.tests_passed} pass / {p_state.tests_failed} fail"
        if p_state.tests_passed is not None
        else "Pending"
    )
    m3.metric("Tests", t_str)
    r_str = (
        f"{p_state.reviewer_score}/100"
        if p_state.reviewer_score is not None
        else "Pending"
    )
    m4.metric("Reviewer Score", r_str)

if st.session_state.async_event_log:
    with st.expander("Live Workflow Event Log", expanded=False):
        for log_entry in st.session_state.async_event_log:
            st.write(log_entry)

if st.session_state.async_workflow_result:
    _render_workflow_result(
        st.session_state.async_workflow_result,
        title_prefix="Asynchronous Workflow",
    )
