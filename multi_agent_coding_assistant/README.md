# Multi-Agent Coding Assistant

Modular multi-agent software engineering assistant with repository-aware RAG, orchestration, and validation workflows.

## Project overview
This repository implements a **modular multi-agent AI software engineering team**.
- Backend: **FastAPI** (API surface + dependency scaffolding)
- Frontend: **Streamlit** (UI scaffold)
- Agents: specialized Planner/Coder/Reviewer/Test/Debug/Docs/Researcher components
- Repository RAG: dependency-injected retrieval, embeddings, vector-store abstraction, and evidence-backed research reports

RAG retrieval and orchestration are implemented; provider integrations and capabilities depend on the configured services.

## Folder structure
Clean architecture mapping is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

High-level folders:
- `backend/`: FastAPI app (API, application, domain, infrastructure)
- `frontend/`: Streamlit UI
- `agents/`: specialized agents and orchestration-facing interfaces
- `tools/`: tool abstractions/registry placeholder
- `rag/`: repository loading, chunking, embedding, vector search, and retrieval ranking
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
- Use-case orchestration layer in `backend/application/`
- Additional API endpoints and end-to-end tests

## License
Add your license as desired.


