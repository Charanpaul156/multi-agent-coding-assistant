"""Streamlit frontend entrypoint for Multi-Agent Coding Assistant.

AI-powered repository-aware coding workspace and dashboard.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import requests
import streamlit as st

from frontend.workflow_ui import (
    WorkflowProgressState,
    check_backend_health,
    format_elapsed_time,
    format_progress_timeline,
    get_repository_info,
    get_timeline_stages,
    iter_sse_events,
    map_workflow_event,
    poll_workflow_job,
    record_recent_run,
    sanitize_display_text,
    sanitize_error_message,
    summarize_prompt,
)

logger = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Multi-Agent Coding Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for developer-tool aesthetic
st.markdown(
    """
    <style>
    /* ==========================================================================
       Multi-Agent Coding Assistant - Light White/Blue-Gray Visual Theme
       ========================================================================== */

    /* Main application layout & background */
    .stApp {
        background-color: #F4F7FB !important;
        color: #172033 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        background-color: #F4F7FB !important;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Secondary / sidebar background */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #EEF2F7 !important;
        border-right: 1px solid #DCE3EC !important;
    }

    /* Typography & Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #172033 !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em;
    }

    .main-header {
        padding: 0.2rem 0 0.8rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #DCE3EC;
    }

    .header-title {
        font-size: 1.85rem;
        font-weight: 700 !important;
        color: #172033 !important;
        margin: 0;
        padding: 0;
    }

    .header-subtitle {
        font-size: 0.95rem;
        color: #64748B !important;
        margin-top: 0.25rem;
        margin-bottom: 0.4rem;
    }

    /* General text & captions */
    p, span, div {
        color: inherit;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748B !important;
        font-size: 0.85rem;
    }

    /* Primary text on labels */
    label[data-testid="stWidgetLabel"] p {
        color: #172033 !important;
        font-weight: 500 !important;
        font-size: 0.88rem;
    }

    /* Dividers */
    hr {
        border-color: #DCE3EC !important;
        margin: 1.2rem 0 !important;
    }

    /* Cards & major panels */
    .card-box {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE3EC !important;
        border-radius: 6px !important;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE3EC !important;
        border-radius: 6px !important;
        padding: 0.85rem !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE3EC !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        box-shadow: none !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    [data-testid="stMetricValue"] {
        color: #172033 !important;
        font-size: 1.45rem !important;
        font-weight: 600 !important;
    }

    /* Expanders */
    details[data-testid="stExpander"],
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE3EC !important;
        border-radius: 6px !important;
        margin-bottom: 0.75rem !important;
    }

    details[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary {
        background-color: #FFFFFF !important;
        color: #172033 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    details[data-testid="stExpander"] summary:hover,
    div[data-testid="stExpander"] summary:hover {
        background-color: #F8FAFC !important;
        color: #172033 !important;
    }

    div[data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        border-top: 1px solid #DCE3EC !important;
        padding: 0.85rem !important;
    }

    /* Input Fields (Text, Textarea, NumberInput) */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background-color: #F8FAFC !important;
        border: 1px solid #DCE3EC !important;
        color: #172033 !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #3B82F6 !important;
        outline: none !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
        background-color: #FFFFFF !important;
    }

    /* Secondary / Standard Buttons */
    div.stButton > button[kind="secondary"],
    div.stButton > button:not([kind="primary"]) {
        background-color: #FFFFFF !important;
        color: #172033 !important;
        border: 1px solid #DCE3EC !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }

    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button:not([kind="primary"]):hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        color: #172033 !important;
    }

    /* Primary Action Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: 1px solid #2563EB !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #2563EB !important;
        border-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }

    /* Download Buttons */
    div[data-testid="stDownloadButton"] > button {
        background-color: #FFFFFF !important;
        color: #172033 !important;
        border: 1px solid #DCE3EC !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
    }

    /* Tabs */
    div[data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid #DCE3EC !important;
        gap: 0.25rem !important;
    }

    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #64748B !important;
        font-weight: 500 !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #172033 !important;
        background-color: #EEF2F7 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #3B82F6 !important;
        border-bottom: 2px solid #3B82F6 !important;
        font-weight: 600 !important;
        background-color: transparent !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #3B82F6 !important;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }

    .status-online {
        background-color: #F0FDF4 !important;
        color: #16A34A !important;
        border: 1px solid #BBF7D0 !important;
    }

    .status-offline {
        background-color: #FEF2F2 !important;
        color: #DC2626 !important;
        border: 1px solid #FECACA !important;
    }

    .status-repo {
        background-color: #EFF6FF !important;
        color: #3B82F6 !important;
        border: 1px solid #BFDBFE !important;
    }

    .status-warning {
        background-color: #FFFBEB !important;
        color: #D97706 !important;
        border: 1px solid #FDE68A !important;
    }

    /* Code Blocks */
    [data-testid="stCodeBlock"] {
        border: 1px solid #DCE3EC !important;
        border-radius: 6px !important;
        background-color: #FFFFFF !important;
    }

    code, pre {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace !important;
    }

    /* Technical details & Code container */
    .technical-details {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.85rem;
        color: #64748B;
    }

    .code-container {
        overflow-x: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _normalize_newlines_for_display(text: str) -> str:
    """Normalize literal newline sequences for display."""
    if not text:
        return ""
    if "\\n" in text or "\\r" in text:
        text = text.replace("\\r\\n", "\r\n")
        text = text.replace("\\n", "\n")
        text = text.replace("\\r", "\r")
    text = re.sub(r"\r?\n", "\n", text)
    return text


# Initialize session state variables safely
if "backend_health" not in st.session_state:
    st.session_state.backend_health = None
if "repo_root" not in st.session_state:
    st.session_state.repo_root = os.getcwd()
if "max_iterations" not in st.session_state:
    st.session_state.max_iterations = 3
if "apply_repo_changes" not in st.session_state:
    st.session_state.apply_repo_changes = False
if "workflow_prompt" not in st.session_state:
    st.session_state.workflow_prompt = ""

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
if "recent_runs" not in st.session_state:
    st.session_state.recent_runs = []

if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "workflow_code" not in st.session_state:
    st.session_state.workflow_code = ""

if "plan" not in st.session_state:
    st.session_state.plan = None
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""
if "execution" not in st.session_state:
    st.session_state.execution = None
if "review" not in st.session_state:
    st.session_state.review = None
if "tests_report" not in st.session_state:
    st.session_state.tests_report = None
if "debug_report" not in st.session_state:
    st.session_state.debug_report = None

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None
if "rag_results" not in st.session_state:
    st.session_state.rag_results = []

if "modify_repository_result" not in st.session_state:
    st.session_state.modify_repository_result = None
if "modify_repository_proposal" not in st.session_state:
    st.session_state.modify_repository_proposal = None


# Quick backend connectivity check (cached or light)
def _get_backend_status() -> dict[str, Any]:
    if st.session_state.backend_health is None:
        st.session_state.backend_health = check_backend_health(BACKEND_URL, timeout=0.4)
    return st.session_state.backend_health


backend_status = _get_backend_status()
repo_info = get_repository_info(st.session_state.repo_root)


# Sidebar controls & system health
with st.sidebar:
    st.markdown("### System Status")
    if backend_status.get("connected"):
        st.success(f"Backend: Connected ({BACKEND_URL})")
    else:
        st.error(f"Backend: Offline ({BACKEND_URL})")

    if st.button("Check Backend Health"):
        health = check_backend_health(BACKEND_URL, timeout=2.0)
        st.session_state.backend_health = health
        if health.get("connected"):
            st.success(f"Status: {health.get('status')}")
        else:
            st.error(f"Error: {health.get('error')}")

    st.markdown("---")
    st.markdown("### Workspace Overview")
    st.caption(f"Root: `{st.session_state.repo_root}`")
    if repo_info.get("branch"):
        st.caption(f"Branch: `{repo_info['branch']}`")
    st.caption(f"Max Iterations: `{st.session_state.max_iterations}`")
    st.caption("Assistant Version: `v0.3.0`")


# ===========================================================================
# Top Application Navigation Tabs
# ===========================================================================

tab_workspace, tab_repo_ops, tab_rag, tab_standalone = st.tabs([
    "Coding Workspace",
    "Repository Changes & Approval",
    "Repository RAG",
    "Standalone Agent Tools",
])


# ===========================================================================
# TAB 1: Main AI Coding Workspace (Step 1 Requirements)
# ===========================================================================
with tab_workspace:
    # -----------------------------------------------------------------------
    # 1. Header / Application Identity
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div class="main-header">
            <h1 class="header-title">Multi-Agent Coding Assistant</h1>
            <p class="header-subtitle">AI-powered repository-aware coding workspace</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hdr_c1, hdr_c2, hdr_c3 = st.columns([2, 3, 1])
    with hdr_c1:
        if backend_status.get("connected"):
            st.markdown(
                '<span class="status-badge status-online">● Backend: Online (HTTP 200)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-offline">● Backend: Disconnected</span>',
                unsafe_allow_html=True,
            )

    with hdr_c2:
        if repo_info.get("valid"):
            branch_str = f" ({repo_info['branch']})" if repo_info.get("branch") else ""
            repo_name = repo_info.get("name") or "Repository"
            st.markdown(
                f'<span class="status-badge status-repo">📁 {repo_name}{branch_str}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-offline">📁 Repository: Not configured</span>',
                unsafe_allow_html=True,
            )

    with hdr_c3:
        if st.button("Refresh Status", key="btn_refresh_status"):
            st.session_state.backend_health = check_backend_health(BACKEND_URL, timeout=1.5)
            st.rerun()

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 2. Project / Repository Panel
    # -----------------------------------------------------------------------
    with st.expander("Project & Repository Settings", expanded=False):
        c_repo1, c_repo2 = st.columns([3, 1])
        with c_repo1:
            input_repo = st.text_input(
                "Repository root directory",
                value=st.session_state.repo_root,
                placeholder=r"C:\path\to\repository",
                help="Absolute path to target local git repository",
                key="input_repo_root",
            )
            if input_repo != st.session_state.repo_root:
                st.session_state.repo_root = input_repo
                repo_info = get_repository_info(input_repo)

        with c_repo2:
            input_iters = st.number_input(
                "Max self-correction iterations",
                min_value=1,
                max_value=5,
                value=st.session_state.max_iterations,
                help="Maximum debugging and self-correction iterations (backend default: 3)",
                key="input_max_iters",
            )
            st.session_state.max_iterations = input_iters

        c_opt1, c_opt2 = st.columns([2, 2])
        with c_opt1:
            apply_changes = st.checkbox(
                "Apply repository changes directly (default: False / preview mode)",
                value=st.session_state.apply_repo_changes,
                help="When disabled, all repository modifications require explicit human approval",
                key="chk_apply_repo_changes",
            )
            st.session_state.apply_repo_changes = apply_changes

        with c_opt2:
            if repo_info.get("valid"):
                branch_msg = f" | Branch: **{repo_info['branch']}**" if repo_info.get("branch") else ""
                st.caption(f"Status: **Verified directory**{branch_msg}")
            elif input_repo.strip():
                st.caption(f"Status: **{repo_info.get('error', 'Invalid repository path')}**")
            else:
                st.caption("Status: Standalone mode (no repository root set)")

    # -----------------------------------------------------------------------
    # 3. AI Coding Request Workspace
    # -----------------------------------------------------------------------
    st.markdown("### AI Coding Request")
    st.caption("Describe the feature, module, or bug fix for the autonomous agent workflow.")

    # Helper prompt presets
    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        if st.button("Example: Calculator", key="btn_example_calc", use_container_width=True):
            st.session_state.workflow_prompt = (
                "Create a Python calculator module with add, subtract, multiply, "
                "and divide functions including zero-division error handling."
            )
            st.rerun()
    with c_p2:
        if st.button("Example: REST Client", key="btn_example_client", use_container_width=True):
            st.session_state.workflow_prompt = (
                "Build a Python HTTP REST client with exponential backoff retry logic, "
                "response status code verification, and custom exception handling."
            )
            st.rerun()
    with c_p3:
        if st.button("Example: Student System", key="btn_example_student", use_container_width=True):
            st.session_state.workflow_prompt = (
                "Implement a Student Management System with student enrollment, "
                "course registration, grade recording, and GPA calculation."
            )
            st.rerun()

    col_prompt_lbl, col_lang_sel = st.columns([3, 2])
    with col_prompt_lbl:
        st.caption("Enter your task requirements in natural language")
    with col_lang_sel:
        selected_language = st.selectbox(
            "Programming Language",
            options=["Auto Detect", "Python", "Java", "JavaScript", "TypeScript", "C++", "C"],
            index=0,
            key="workflow_selected_language",
            help="Choose explicit programming language or let assistant auto-detect from prompt",
        )

    active_prompt_val = st.session_state.get("workflow_prompt", "")
    workflow_request_prompt = st.text_area(
        "Coding prompt",
        value=active_prompt_val,
        height=140,
        placeholder="Enter your coding request here (e.g., 'Build a robust RateLimiter class with sliding window algorithm')...",
        key="main_workflow_prompt_input",
    )
    st.session_state.workflow_prompt = workflow_request_prompt

    col_btn_async, col_btn_sync, col_btn_clear = st.columns([2, 2, 1])
    with col_btn_async:
        run_live_clicked = st.button(
            "Run Live Streaming Workflow",
            type="primary",
            use_container_width=True,
            key="btn_run_live_stream",
        )
    with col_btn_sync:
        run_standard_clicked = st.button(
            "Run Standard Workflow",
            use_container_width=True,
            key="btn_run_standard",
        )
    with col_btn_clear:
        if st.button("Clear", use_container_width=True, key="btn_clear_request"):
            st.session_state.workflow_prompt = ""
            st.rerun()

    # Placeholders for live execution dashboard
    status_box = st.empty()
    metrics_box = st.empty()
    activity_box = st.empty()
    timeline_box = st.empty()

    # -----------------------------------------------------------------------
    # Handle Live Streaming Workflow execution (Async SSE)
    # -----------------------------------------------------------------------
    if run_live_clicked:
        clean_prompt = workflow_request_prompt.strip()
        if not clean_prompt:
            st.warning("Please enter a coding prompt before running the workflow.")
        else:
            st.session_state.async_workflow_result = None
            st.session_state.async_event_log = []
            progress_state = WorkflowProgressState()
            st.session_state.async_progress_state = progress_state

            status_box.info("Submitting asynchronous workflow job to backend...")

            payload_data: dict[str, Any] = {"prompt": clean_prompt}
            if selected_language and selected_language != "Auto Detect":
                payload_data["language"] = selected_language.lower()
            if st.session_state.repo_root and os.path.exists(st.session_state.repo_root):
                payload_data["repository_root"] = st.session_state.repo_root
            if st.session_state.apply_repo_changes:
                payload_data["apply_repository_changes"] = True

            try:
                submit_resp = requests.post(
                    f"{BACKEND_URL}/workflow/jobs",
                    json=payload_data,
                    timeout=15,
                )
                if submit_resp.status_code != 202:
                    status_box.error(
                        f"Job submission failed ({submit_resp.status_code}): "
                        f"{sanitize_error_message(submit_resp.text)}"
                    )
                else:
                    submit_data = submit_resp.json()
                    job_id = submit_data.get("job_id", "")
                    st.session_state.async_job_id = job_id
                    progress_state.job_id = job_id
                    progress_state.status = submit_data.get("status", "queued")
                    progress_state.start_time = time.monotonic()
                    status_box.success(f"Job accepted. ID: `{job_id}`")

                    def _render_live_dashboard_components():
                        # Metrics row
                        m1, m2, m3, m4, m5 = metrics_box.columns(5)
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
                        elapsed_str = format_elapsed_time(progress_state.elapsed_seconds)
                        m5.metric("Elapsed", elapsed_str)

                        # Visual timeline
                        timeline_box.markdown(format_progress_timeline(progress_state))

                    _render_live_dashboard_components()

                    # Stream SSE events
                    stream_url = f"{BACKEND_URL}/workflow/jobs/{job_id}/stream"
                    status_box.info(f"Connecting to live SSE stream for job `{job_id}` ...")

                    sse_succeeded = False
                    try:
                        with requests.get(stream_url, stream=True, timeout=300) as sse_resp:
                            if sse_resp.status_code == 200:
                                sse_succeeded = True
                                status_box.info("Live SSE connected. Tracking real-time agent execution...")
                                for parsed_ev in iter_sse_events(sse_resp.iter_lines(decode_unicode=True)):
                                    ev_type = parsed_ev.event_type
                                    p_load = (
                                        parsed_ev.data.get("payload", {})
                                        if isinstance(parsed_ev.data, dict)
                                        else {}
                                    )
                                    log_msg = map_workflow_event(ev_type, p_load, progress_state)
                                    timestamp_str = time.strftime("%H:%M:%S")
                                    st.session_state.async_event_log.append(f"[{timestamp_str}] {log_msg}")

                                    _render_live_dashboard_components()
                                    activity_box.caption(f"Latest activity: **{log_msg}**")

                                    if progress_state.is_terminal:
                                        break
                    except Exception as sse_exc:
                        logger.warning("Live SSE stream interrupted: %s", sse_exc)

                    # Fallback polling if SSE ended before terminal state
                    if not progress_state.is_terminal:
                        status_box.warning("Streaming disconnected. Polling background job status...")

                        def _poll_cb(p_data: dict[str, Any]):
                            st_name = p_data.get("status", "running")
                            progress_state.status = st_name
                            _render_live_dashboard_components()

                        poll_res = poll_workflow_job(
                            BACKEND_URL,
                            job_id,
                            max_wait_seconds=180.0,
                            poll_interval=1.5,
                            on_poll_callback=_poll_cb,
                        )
                        if poll_res:
                            p_status = poll_res.get("status")
                            if p_status == "completed":
                                progress_state.status = "completed"
                                progress_state.is_terminal = True
                            elif p_status in ("failed", "cancelled"):
                                progress_state.status = "failed"
                                progress_state.is_terminal = True
                                progress_state.error = sanitize_error_message(
                                    poll_res.get("error", "Job failed")
                                )

                    # Fetch final complete job details
                    try:
                        final_resp = requests.get(f"{BACKEND_URL}/workflow/jobs/{job_id}", timeout=15)
                        if final_resp.status_code == 200:
                            final_data = final_resp.json()
                            st.session_state.async_workflow_status = final_data.get("status")
                            st.session_state.async_workflow_result = final_data.get("result")
                    except Exception as f_exc:
                        logger.error("Failed to retrieve final job payload: %s", f_exc)

                    _render_live_dashboard_components()

                    # Record recent run in session state
                    run_record = {
                        "job_id": job_id,
                        "status": progress_state.status.title(),
                        "prompt_summary": summarize_prompt(clean_prompt),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "tests_result": (
                            f"{progress_state.tests_passed} pass / {progress_state.tests_failed} fail"
                            if progress_state.tests_passed is not None
                            else "N/A"
                        ),
                        "reviewer_score": (
                            f"{progress_state.reviewer_score}/100"
                            if progress_state.reviewer_score is not None
                            else "N/A"
                        ),
                        "duration": format_elapsed_time(progress_state.elapsed_seconds),
                    }
                    st.session_state.recent_runs = record_recent_run(
                        st.session_state.recent_runs,
                        run_record,
                    )

                    if progress_state.status == "completed":
                        status_box.success("Workflow completed successfully!")
                    else:
                        status_box.error(
                            f"Workflow failed: {progress_state.error or 'Execution failed'}"
                        )
            except Exception as exc:
                st.error(f"Execution error: {sanitize_error_message(str(exc))}")

    # -----------------------------------------------------------------------
    # Handle Standard Workflow execution (Synchronous /run-workflow)
    # -----------------------------------------------------------------------
    if run_standard_clicked:
        clean_prompt = workflow_request_prompt.strip()
        if not clean_prompt:
            st.warning("Please enter a coding prompt before running the workflow.")
        else:
            payload_data: dict[str, Any] = {"prompt": clean_prompt}
            if selected_language and selected_language != "Auto Detect":
                payload_data["language"] = selected_language.lower()
            if st.session_state.repo_root and os.path.exists(st.session_state.repo_root):
                payload_data["repository_root"] = st.session_state.repo_root
            if st.session_state.apply_repo_changes:
                payload_data["apply_repository_changes"] = True

            start_t = time.monotonic()
            try:
                with st.spinner("Running full multi-agent workflow synchronously..."):
                    resp = requests.post(
                        f"{BACKEND_URL}/run-workflow",
                        json=payload_data,
                        timeout=300,
                    )
                elapsed_s = time.monotonic() - start_t

                if resp.status_code != 200:
                    st.error(f"Workflow request failed ({resp.status_code}): {sanitize_error_message(resp.text)}")
                else:
                    wf_data = resp.json()
                    wf_obj = wf_data.get("workflow", {})
                    st.session_state.workflow = wf_obj
                    is_success = wf_data.get("success", False)

                    # Extract test and review data
                    t_exec = wf_obj.get("test_execution") or {}
                    p_cnt = t_exec.get("passed")
                    f_cnt = t_exec.get("failed")
                    tests_str = f"{p_cnt} pass / {f_cnt} fail" if p_cnt is not None else "N/A"

                    rev = wf_obj.get("review") or {}
                    r_score = rev.get("overall_score")
                    rev_str = f"{r_score}/100" if r_score is not None else "N/A"

                    run_record = {
                        "job_id": f"sync_{int(time.time())}",
                        "status": "Completed" if is_success else "Failed",
                        "prompt_summary": summarize_prompt(clean_prompt),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "tests_result": tests_str,
                        "reviewer_score": rev_str,
                        "duration": format_elapsed_time(elapsed_s),
                    }
                    st.session_state.recent_runs = record_recent_run(
                        st.session_state.recent_runs,
                        run_record,
                    )

                    if is_success:
                        st.success("Workflow completed successfully!")
                    else:
                        st.warning("Workflow finished with warnings or issues.")
            except Exception as exc:
                st.error(f"Workflow failed: {sanitize_error_message(str(exc))}")

    # -----------------------------------------------------------------------
    # 4 & 5. Live Execution Dashboard & Activity Timeline (Render when active)
    # -----------------------------------------------------------------------
    if st.session_state.async_progress_state and not run_live_clicked:
        p_state = st.session_state.async_progress_state
        st.markdown("### Execution Dashboard")

        # Compact technical details bar
        c_tech1, c_tech2, c_tech3 = st.columns([2, 1, 1])
        with c_tech1:
            st.caption(f"Job ID: `{p_state.job_id or 'N/A'}`")
        with c_tech2:
            st.caption(f"Status: **{p_state.status.title()}**")
        with c_tech3:
            st.caption(f"Elapsed: **{format_elapsed_time(p_state.elapsed_seconds)}**")

        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
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
        m5.metric("Elapsed Time", format_elapsed_time(p_state.elapsed_seconds))

        st.markdown("#### Agent Activity Timeline")
        st.markdown(format_progress_timeline(p_state))

    if st.session_state.async_event_log:
        with st.expander("Live Workflow Event Log", expanded=False):
            for entry in st.session_state.async_event_log:
                st.write(sanitize_display_text(entry))

    # -----------------------------------------------------------------------
    # 6. Final Result Area (Organized in Structured Tabs)
    # -----------------------------------------------------------------------
    active_result = st.session_state.async_workflow_result or st.session_state.workflow
    if active_result:
        st.markdown("---")
        st.markdown("### Workflow Results")

        res_tab1, res_tab2, res_tab3, res_tab4, res_tab5 = st.tabs([
            "Summary",
            "Generated Code",
            "Tests",
            "Review",
            "Documentation",
        ])

        # TAB 1: Summary
        with res_tab1:
            st.markdown("#### Execution Summary")
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Workflow Status", str(active_result.get("workflow_status", "unknown")).upper())
            s_col2.metric("Execution Time (ms)", f"{active_result.get('execution_time_ms', 0):.1f}")
            s_col3.metric("Success", "Yes" if active_result.get("success", False) else "No")

            if active_result.get("error"):
                st.error(sanitize_error_message(str(active_result.get("error"))))
            if active_result.get("test_error"):
                st.warning(f"Test Issue: {active_result.get('test_error')}")

            # Planning breakdown
            plan_obj = active_result.get("planning")
            if plan_obj:
                with st.expander("Implementation Plan Overview", expanded=True):
                    st.write(f"**Problem Summary:** {plan_obj.get('problem_summary', '')}")
                    st.write(f"**Project Type:** {plan_obj.get('project_type', '')}")
                    if plan_obj.get("requirements"):
                        st.markdown("**Key Requirements:**")
                        for req in plan_obj.get("requirements", []):
                            st.write(f"- {req}")
                    if plan_obj.get("modules"):
                        st.markdown("**Modules & Architecture:**")
                        for mod in plan_obj.get("modules", []):
                            st.write(f"- {mod}")

            # Debugging iterations if any
            iters = active_result.get("iterations") or []
            if iters:
                with st.expander(f"Self-Correction History ({len(iters)} iteration(s))", expanded=False):
                    for it in iters:
                        it_num = it.get("iteration_number", "?")
                        st.markdown(f"**Iteration {it_num}**")
                        dbg = it.get("debug_report")
                        if dbg:
                            st.write(f"- Defect: {dbg.get('error_type', 'N/A')}")
                            st.write(f"- Cause: {dbg.get('root_cause', 'N/A')}")

        # TAB 2: Generated Code
        with res_tab2:
            st.markdown("#### Generated Python Code")
            code_text = active_result.get("generated_code") or ""
            if code_text:
                norm_code = _normalize_newlines_for_display(code_text)
                st.code(norm_code, language="python")
                st.download_button(
                    label="Download generated_code.py",
                    data=norm_code,
                    file_name="generated_code.py",
                    mime="text/x-python",
                    key="dl_main_code",
                )
                st.caption(f"Line count: {len(norm_code.splitlines())} lines | Ready for deployment.")
            else:
                st.info("No code generated in this run.")

        # TAB 3: Tests
        with res_tab3:
            st.markdown("#### Test Execution & Test Suite")
            t_exec = active_result.get("test_execution")
            if t_exec:
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("Tests Passed", t_exec.get("passed", 0))
                tc2.metric("Tests Failed", t_exec.get("failed", 0))
                tc3.metric("Exit Code", t_exec.get("exit_code", 0))

                if t_exec.get("stdout"):
                    with st.expander("Test Stdout", expanded=False):
                        st.code(t_exec.get("stdout"), language="text")
                if t_exec.get("stderr"):
                    with st.expander("Test Stderr", expanded=False):
                        st.code(t_exec.get("stderr"), language="text")

            tests_text = active_result.get("generated_tests") or ""
            if tests_text:
                norm_tests = _normalize_newlines_for_display(tests_text)
                st.markdown("##### Generated Pytest Suite")
                st.code(norm_tests, language="python")
                st.download_button(
                    label="Download tests_generated.py",
                    data=norm_tests,
                    file_name="tests_generated.py",
                    mime="text/x-python",
                    key="dl_main_tests",
                )
            else:
                st.info("No test code generated.")

        # TAB 4: Review
        with res_tab4:
            st.markdown("#### Code Reviewer Report")
            rev_obj = active_result.get("review")
            if rev_obj:
                score_val = rev_obj.get("overall_score", "N/A")
                st.metric("Reviewer Score", f"{score_val} / 100")

                if rev_obj.get("strengths"):
                    with st.expander("Key Strengths", expanded=True):
                        for s in rev_obj.get("strengths", []):
                            st.write(f"- {s}")
                if rev_obj.get("weaknesses"):
                    with st.expander("Weaknesses & Issues", expanded=False):
                        for w in rev_obj.get("weaknesses", []):
                            st.write(f"- {w}")
                if rev_obj.get("logic_issues"):
                    with st.expander("Logic Concerns", expanded=False):
                        for li in rev_obj.get("logic_issues", []):
                            st.write(f"- {li}")
                if rev_obj.get("recommendations"):
                    with st.expander("Recommendations", expanded=True):
                        for r in rev_obj.get("recommendations", []):
                            st.write(f"- {r}")
                if rev_obj.get("final_summary"):
                    st.markdown("**Reviewer Summary:**")
                    st.write(rev_obj.get("final_summary"))
            else:
                st.info("No review report available.")

        # TAB 5: Documentation
        with res_tab5:
            st.markdown("#### Generated Documentation")
            doc_obj = active_result.get("documentation")
            if doc_obj and doc_obj.get("markdown_documentation"):
                st.markdown(doc_obj.get("markdown_documentation"))
            else:
                st.info("No documentation generated for this run.")

    # -----------------------------------------------------------------------
    # 7. Recent Workflow Runs (Session Local)
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Recent Workflow Runs")
    if not st.session_state.recent_runs:
        st.caption("No workflow runs recorded in this session yet.")
    else:
        for idx, run in enumerate(st.session_state.recent_runs):
            with st.container(border=True):
                r_c1, r_c2, r_c3, r_c4, r_c5, r_c6 = st.columns([2, 1, 3, 1, 1, 1])
                r_c1.write(f"**`{run.get('job_id', 'N/A')}`**")
                r_c2.write(run.get("status", "N/A"))
                r_c3.write(run.get("prompt_summary", ""))
                r_c4.write(run.get("tests_result", "N/A"))
                r_c5.write(run.get("reviewer_score", "N/A"))
                r_c6.write(run.get("duration", "N/A"))


# ===========================================================================
# TAB 2: Repository Operations & Human Approval Gate (Preserved)
# ===========================================================================
with tab_repo_ops:
    st.subheader("Modify Repository")
    st.caption(
        "Repository-aware code modification. Changes are generated, validated, "
        "and displayed as a diff preview before being applied. Human approval "
        "is strictly required before writing changes."
    )

    mod_repo_path = st.text_input(
        "Repository",
        value=st.session_state.repo_root,
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
                with st.spinner("Retrieving context -> planning -> generating changes -> validating..."):
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
                        st.success("Proposed changes generated successfully")
                    else:
                        st.warning(data.get("error", "Modification did not complete"))
            except Exception as exc:
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
            st.warning("Proposal was rejected. No files were written to the repository.")
        elif approval_status in ("approved", "applied"):
            st.success("Changes have been applied to the repository.")

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
                    if change.get("new_content"):
                        st.code(
                            _normalize_newlines_for_display(change.get("new_content", "")),
                            language="python",
                        )

        diff_entries = result.get("diff", []) or []
        if diff_entries:
            st.subheader("Diff Preview")
            for entry in diff_entries:
                with st.expander(f"{entry.get('operation', '')}: {entry.get('file_path', '')}", expanded=False):
                    st.code(
                        _normalize_newlines_for_display(entry.get("diff_text", "")),
                        language="diff",
                    )

    _render_modify_repository_result(st.session_state.modify_repository_result)

    st.markdown("---")
    st.markdown("### Human Approval Gate")
    st.caption("Repository modifications require explicit human approval. NO files are written until approved.")

    res = st.session_state.modify_repository_result
    token = res.get("approval_token") if res else None
    repo = (
        st.session_state.modify_repository_proposal.get("repository")
        if st.session_state.modify_repository_proposal
        else None
    )

    col_app1, col_app2 = st.columns(2)
    with col_app1:
        if st.button("Approve Changes", type="primary", key="btn_approve_changes"):
            if not token:
                st.error("No active proposal to approve. Generate changes first.")
            else:
                try:
                    with st.spinner("Applying approved changes to repository..."):
                        resp = requests.post(
                            f"{BACKEND_URL}/modify-repository/approve",
                            json={"approval_token": token, "repository": repo},
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
                except Exception as exc:
                    st.error(f"Failed to approve changes: {exc}")

    with col_app2:
        if st.button("Reject Changes", key="btn_reject_changes"):
            if not token:
                st.error("No active proposal to reject. Generate changes first.")
            else:
                try:
                    with st.spinner("Rejecting proposed changes..."):
                        resp = requests.post(
                            f"{BACKEND_URL}/modify-repository/reject",
                            json={"approval_token": token, "repository": repo},
                            timeout=60,
                        )
                    if resp.status_code != 200:
                        st.error(f"Rejection failed: {resp.status_code} - {resp.text}")
                    else:
                        data = resp.json()
                        st.session_state.modify_repository_result = data
                        st.warning("Proposed changes rejected. Filesystem remains untouched.")
                except Exception as exc:
                    st.error(f"Failed to reject changes: {exc}")


# ===========================================================================
# TAB 3: Repository RAG (Preserved)
# ===========================================================================
with tab_rag:
    st.subheader("Repository RAG")
    st.caption("Index a repository and search it to retrieve relevant code context.")

    repo_path = st.text_input(
        "Repository path for RAG",
        value=st.session_state.repo_root,
        placeholder=r"C:\path\to\repository",
        key="rag_repo_path_input",
    )

    col_rag1, col_rag2 = st.columns(2)
    with col_rag1:
        if st.button("Index Repository", key="btn_index_repo"):
            if not repo_path.strip():
                st.error("Repository path must not be empty")
            else:
                try:
                    with st.spinner("Indexing repository into vector store..."):
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
                            st.success(f"Indexed {data.get('file_count', 0)} files / {data.get('chunk_count', 0)} chunks")
                        else:
                            st.warning(data.get("error", "Indexing incomplete"))
                except Exception as exc:
                    st.error(f"Failed to index repository: {exc}")

    with col_rag2:
        if st.button("Show RAG Status", key="btn_show_rag_status"):
            try:
                with st.spinner("Fetching RAG status..."):
                    resp = requests.get(f"{BACKEND_URL}/rag/status", timeout=15)
                if resp.status_code != 200:
                    st.error(f"Status failed: {resp.status_code} - {resp.text}")
                else:
                    data = resp.json()
                    st.session_state.rag_status = data
                    st.write(f"Chunks indexed: {data.get('chunk_count', 0)}")
                    st.write(f"Repositories: {', '.join(data.get('repositories', []) or []) or 'none'}")
            except Exception as exc:
                st.error(f"Failed to fetch RAG status: {exc}")

    st.markdown("### Search Repository")
    search_query = st.text_area(
        "Search query",
        height=80,
        placeholder="Find where authentication is implemented",
        key="rag_search_query_input",
    )

    if st.button("Search Repository", key="btn_search_repo"):
        if not search_query.strip():
            st.error("Search query must not be empty")
        else:
            try:
                with st.spinner("Searching repository chunks..."):
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
            except Exception as exc:
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


# ===========================================================================
# TAB 4: Standalone Agent Tools (Planning, Code, Review, Tests, Debug) (Preserved)
# ===========================================================================
with tab_standalone:
    st.subheader("Planning")
    plan_prompt = st.text_area(
        "Planning request",
        height=100,
        placeholder="Build a Banking Management System",
        key="standalone_plan_prompt",
    )
    if st.button("Generate Plan", key="btn_standalone_plan"):
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
            except Exception as exc:
                st.error(f"Failed to generate plan: {exc}")

    if st.session_state.plan:
        with st.expander("Generated Plan Details", expanded=True):
            p = st.session_state.plan
            st.write(f"**Problem:** {p.get('problem_summary', '')}")
            st.write(f"**Type:** {p.get('project_type', '')}")
            st.markdown("**Requirements:**")
            st.write("\n".join(p.get("requirements", []) or []))

    st.markdown("---")
    st.subheader("Generate Code")
    col_gen_lbl, col_gen_lang = st.columns([3, 2])
    with col_gen_lang:
        standalone_gen_lang = st.selectbox(
            "Language",
            options=["Auto Detect", "Python", "Java", "JavaScript", "TypeScript", "C++", "C"],
            index=0,
            key="standalone_gen_lang",
        )
    gen_prompt = st.text_area(
        "Programming request",
        height=100,
        placeholder="Create a calculator in Java",
        key="standalone_gen_code_prompt",
    )
    if st.button("Generate Code", key="btn_standalone_gen_code"):
        if not gen_prompt.strip():
            st.error("Prompt must not be empty")
        else:
            try:
                with st.spinner("Generating code..."):
                    gen_payload: dict[str, Any] = {"prompt": gen_prompt}
                    if standalone_gen_lang and standalone_gen_lang != "Auto Detect":
                        gen_payload["language"] = standalone_gen_lang.lower()
                    resp = requests.post(
                        f"{BACKEND_URL}/generate-code",
                        json=gen_payload,
                        timeout=120,
                    )
                if resp.status_code != 200:
                    st.error(f"Request failed: {resp.status_code} - {resp.text}")
                else:
                    data = resp.json()
                    st.session_state.generated_code = _normalize_newlines_for_display(
                        data.get("generated_code", "")
                    )
                    st.success("Code generated")
            except Exception as exc:
                st.error(f"Failed to generate code: {exc}")

    if st.session_state.generated_code:
        st.code(st.session_state.generated_code, language="python")

        col_st_act1, col_st_act2, col_st_act3 = st.columns(3)
        with col_st_act1:
            if st.button("Execute Code", key="btn_standalone_exec"):
                try:
                    with st.spinner("Executing code in sandbox..."):
                        resp = requests.post(
                            f"{BACKEND_URL}/execute-code",
                            json={"generated_code": st.session_state.generated_code},
                            timeout=30,
                        )
                    if resp.status_code == 200:
                        st.session_state.execution = resp.json()
                        st.success("Execution completed")
                except Exception as exc:
                    st.error(f"Execution error: {exc}")

        with col_st_act2:
            if st.button("Review Code", key="btn_standalone_rev"):
                try:
                    with st.spinner("Reviewing code quality..."):
                        resp = requests.post(
                            f"{BACKEND_URL}/review-code",
                            json={"generated_code": st.session_state.generated_code},
                            timeout=120,
                        )
                    if resp.status_code == 200:
                        st.session_state.review = resp.json().get("review")
                        st.success("Review complete")
                except Exception as exc:
                    st.error(f"Review error: {exc}")

        with col_st_act3:
            if st.button("Generate Tests", key="btn_standalone_tests"):
                try:
                    with st.spinner("Generating test suite..."):
                        resp = requests.post(
                            f"{BACKEND_URL}/generate-tests",
                            json={"generated_code": st.session_state.generated_code},
                            timeout=120,
                        )
                    if resp.status_code == 200:
                        st.session_state.tests_report = resp.json().get("report")
                        st.success("Tests generated")
                except Exception as exc:
                    st.error(f"Test generation error: {exc}")

        if st.session_state.execution:
            with st.expander("Execution Output", expanded=True):
                st.write(f"Exit Code: {st.session_state.execution.get('exit_code')}")
                if st.session_state.execution.get("stdout"):
                    st.code(st.session_state.execution.get("stdout"), language="text")
                if st.session_state.execution.get("stderr"):
                    st.code(st.session_state.execution.get("stderr"), language="text")

        if st.session_state.review:
            with st.expander("Review Report", expanded=True):
                st.metric("Reviewer Score", st.session_state.review.get("overall_score", "N/A"))
                st.write(st.session_state.review.get("final_summary", ""))

        if st.session_state.tests_report:
            with st.expander("Generated Tests Suite", expanded=True):
                t_code = _normalize_newlines_for_display(
                    st.session_state.tests_report.get("generated_test_code", "")
                )
                st.code(t_code, language="python")
