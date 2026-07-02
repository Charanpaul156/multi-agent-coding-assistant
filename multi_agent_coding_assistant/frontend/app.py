"""Streamlit frontend entrypoint.

Scaffolding only.
"""

import requests
import streamlit as st

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Multi-Agent Coding Assistant", layout="wide")

st.title("Multi-Agent Coding Assistant")

if st.button("Check Backend Health"):
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        st.success(f"Backend status: {resp.json()}")
    except Exception as exc:  # pragma: no cover
        st.error(f"Failed to reach backend: {exc}")

st.info("UI scaffolding only. Agent workflows are not implemented yet.")

