# Architecture

Clean folder architecture with separation of concerns.

## High-level components
- `backend/`: FastAPI backend (API surface, orchestration wiring, dependencies)
- `frontend/`: Streamlit UI (chat / request submission)
- `agents/`: CrewAI agent definitions (placeholders for now)
- `tools/`: reusable tool wrappers (placeholders for future external tools)
- `rag/`: retrieval modules (placeholders for future LangChain + Chroma)
- `config/`: configuration loading and validation
- `tests/`: test scaffolding
- `docs/`: documentation

## Clean architecture mapping
- **API layer**: `backend/api/*`
- **Application layer**: `backend/application/*` (use-cases; placeholder)
- **Domain layer**: `backend/domain/*` (entities / DTOs; placeholder)
- **Infrastructure**: `backend/infrastructure/*` (persistence, external services; placeholder)

This repo currently contains scaffolding only; business logic and workflows are intentionally omitted.

