"""Streamlit frontend entrypoint.

UI for generating Python code via POST /generate-code.
"""

from __future__ import annotations

import re

import requests
import streamlit as st

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

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


