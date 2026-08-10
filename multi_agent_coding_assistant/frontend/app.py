"""Streamlit frontend entrypoint.

UI for the Multi-Agent Coding Assistant.
"""

from __future__ import annotations

import re

import requests
import streamlit as st

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

if "mod_repo" not in st.session_state:
    st.session_state.mod_repo = None

mod_repo_path = st.text_input(
    "Modify - Repository path",
    placeholder=r"C:\path\to\repository",
    key="mod_repo_path",
)

mod_request = st.text_area(
    "Modify - Request",
    height=100,
    placeholder="Add password reset functionality to this project",
    key="mod_request",
)

mod_dry_run = st.checkbox("Dry run (only show proposed changes, do not apply)", value=True)

if st.button("Generate Changes (Plan)"):
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
                        "dry_run": mod_dry_run,
                    },
                    timeout=300,
                )
            if resp.status_code != 200:
                st.error(f"Modification failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.session_state.mod_repo = data
                if data.get("success", False):
                    st.success("Proposed changes generated")
                else:
                    st.warning(
                        data.get("error", "Modification did not complete")
                    )
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to modify repository: {exc}")

if st.session_state.mod_repo:
    mod = st.session_state.mod_repo
    st.write(f"Status: {mod.get('status', '')}")
    st.write(f"Dry run: {mod.get('dry_run', False)}")

    if mod.get("error"):
        st.error(mod.get("error"))

    if mod.get("validation_errors"):
        st.subheader("Validation Issues")
        for err in mod.get("validation_errors", []):
            st.warning(err)

    changes = mod.get("changes", []) or []
    if changes:
        st.subheader("Proposed Changes")
        for ch in changes:
            st.markdown(
                f"**{ch.get('operation', '')}** {ch.get('file_path', '')}"
            )
            if ch.get("description"):
                st.write(ch.get("description", ""))
            if ch.get("original_hash"):
                st.caption(f"Original hash: {ch['original_hash'][:12]}...")

    diff_text = mod.get("diff")
    if diff_text:
        st.subheader("Diff")
        st.code(_normalize_newlines_for_display(diff_text), language="diff")

    if mod.get("applied_files"):
        st.subheader("Applied Files")
        st.write("\n".join(mod.get("applied_files", [])))

    if mod.get("rollback"):
        st.caption("Rollback information captured for the applied transaction.")

    if not mod.get("dry_run", True) and not mod.get("success", False):
        pass

st.markdown("---")
st.markdown("### Apply Changes")
st.caption(
    "Apply button intentionally separate from generation. Only appears after "
    "a successful dry-run proposal."
)
apply_target_repo = st.text_input(
    "Apply - Repository path",
    placeholder=r"C:\path\to\repository",
    key="apply_repo_path",
)
apply_request = st.text_area(
    "Apply - Request",
    height=80,
    placeholder="Apply the previously proposed changes",
    key="apply_request",
)
if st.button("Apply Changes"):
    if not apply_target_repo.strip():
        st.error("Repository path must not be empty")
    elif not apply_request.strip():
        st.error("Request must not be empty")
    else:
        try:
            with st.spinner("Applying validated changes..."):
                resp = requests.post(
                    f"{BACKEND_URL}/modify-repository",
                    json={
                        "repository": apply_target_repo.strip(),
                        "request": apply_request.strip(),
                        "dry_run": False,
                    },
                    timeout=300,
                )
            if resp.status_code != 200:
                st.error(f"Apply failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.session_state.mod_repo = data
                if data.get("success", False):
                    st.success("Changes applied")
                    st.write(f"Applied files: {', '.join(data.get('applied_files', []) or [])}")
                else:
                    st.warning(
                        data.get("error", "Apply did not complete")
                    )
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to apply changes: {exc}")


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


if st.session_state.workflow:
    wf = st.session_state.workflow
    st.write(f"Workflow Status: {wf.get('workflow_status', '')}")
    st.write(f"Execution Time (ms): {wf.get('execution_time_ms', '')}")

    if wf.get("error"):
        st.error(wf.get("error"))

    if wf.get("test_error"):
        st.warning(f"Test generation issue: {wf.get('test_error')}")

    # 1. Implementation Plan
    wf_plan = wf.get("planning")
    if wf_plan:
        with st.expander("Workflow Plan"):
            st.write(f"Problem Summary: {wf_plan.get('problem_summary', '')}")
            st.write(f"Project Type: {wf_plan.get('project_type', '')}")
            st.write("\n".join(wf_plan.get("requirements", []) or []))
            st.write("\n".join(wf_plan.get("modules", []) or []))

    # 2. Final Generated Code
    wf_code = wf.get("generated_code")
    if wf_code:
        st.subheader("Workflow Final Generated Code")
        st.code(_normalize_newlines_for_display(wf_code), language="python")

    # 3. Final Generated Tests
    wf_tests = wf.get("generated_tests")
    if wf_tests:
        st.subheader("Workflow Final Generated Tests")
        st.code(_normalize_newlines_for_display(wf_tests), language="python")
        st.download_button(
            label="Download workflow tests_generated.py",
            data=_normalize_newlines_for_display(wf_tests),
            file_name="workflow_tests_generated.py",
            mime="text/x-python",
        )

    # 4. Final Application Execution
    _render_execution(wf.get("execution"), "Workflow Final Application Execution")

    # 5. Final Test Execution
    _render_execution(wf.get("test_execution"), "Workflow Final Test Execution")

    # 6. Final Reviewer Report
    wf_review = wf.get("review")
    if wf_review:
        with st.expander("Workflow Final Review"):
            st.metric("Overall Score", wf_review.get("overall_score", "N/A"))
            st.write("\n".join(wf_review.get("strengths", []) or []))
            st.write("\n".join(wf_review.get("recommendations", []) or []))

    # 7. Iteration history
    iterations = wf.get("iterations") or []
    if iterations:
        st.subheader("Debugging Iterations")
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
