# Multi-Agent Coding Assistant

Production-quality portfolio project foundation.

## Project overview
This repository scaffolds a **modular multi-agent AI software engineering team**.
- Backend: **FastAPI** (API surface + dependency scaffolding)
- Frontend: **Streamlit** (UI scaffold)
- Agents: placeholders for future **CrewAI** agents (Planner/Coder/Reviewer/Test/Debug/Docs/RAG)
- RAG/Vector DB: prepared for future **LangChain + ChromaDB** integration

At this stage, **no agent workflows or AI business logic are implemented**.

## Folder structure
Clean architecture mapping is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

High-level folders:
- `backend/`: FastAPI app (API, application, domain, infrastructure)
- `frontend/`: Streamlit UI
- `agents/`: CrewAI agent placeholders (not implemented)
- `tools/`: tool abstractions/registry placeholder
- `rag/`: RAG placeholder modules (not implemented)
- `config/`: typed configuration loading
- `tests/`: test scaffolding
- `docs/`: additional documentation

## Environment setup
1. Copy environment example:
   - `backend/.env.example` → `backend/.env`
2. Set required keys for your eventual LLM provider (e.g., `OPENAI_API_KEY`).

This project uses `python-dotenv` + `pydantic-settings` via `config/settings.py`.

## Installation
```bash
python -m venv .venv
.
pip install -r requirements.txt
```

## Running the backend
```bash
uvicorn backend.main:app --reload
```
Health endpoint:
- `GET http://localhost:8000/health`

## Running the frontend
In a separate terminal:
```bash
streamlit run frontend/app.py
```

## Future roadmap (not implemented yet)
- CrewAI team wiring (Planner/Coder/Reviewer/Test/Debug/Docs/RAG agents)
- LLM provider integration in `backend/infrastructure/llm_client.py`
- Vector store (ChromaDB) integration in `backend/infrastructure/vector_store.py`
- RAG pipeline using `rag/` modules (LangChain + ChromaDB)
- Use-case orchestration layer in `backend/application/`
- Additional API endpoints and end-to-end tests

## License
Add your license as desired.


