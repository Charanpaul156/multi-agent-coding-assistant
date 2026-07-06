# TODO

- [x] Inspect current frontend/app.py logic for generate-code POST requests and backend health check.
- [x] Fix BUG 1: ensure only ONE POST request to /generate-code occurs, inside the spinner.
- [x] Fix BUG 2: update “Check Backend Health” to use resp = requests.get(.../health) and handle success/connection/non-200 with Streamlit messages.
- [ ] Verify:
  - [ ] Only ONE POST request is made.
  - [ ] Health check works (success + connection failure + non-200).
  - [ ] Code generation still works and st.code + download remain functional.
  - [ ] No import errors / code style issues.

