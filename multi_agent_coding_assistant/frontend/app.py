"""Streamlit frontend entrypoint.

UI for generating Python code via POST /generate-code.
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

    # Clipboard copy in Streamlit is not reliably supported without using
    # custom components. Provide an always-available copy target instead.
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


