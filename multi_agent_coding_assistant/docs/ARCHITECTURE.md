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

## Repository RAG flow

Repository research uses the dependency-injected pipeline in `rag/`:

1. `RepositoryLoader` reads allowed repository files and records repository-relative paths.
2. `CodeChunker` creates source chunks with file paths, line ranges, language, and chunk indexes.
3. `Embedder` creates vectors; `VectorStore` performs similarity search through its stable abstraction.
4. `RepositoryRetriever` embeds each query once, requests a bounded candidate pool, removes exact normalized duplicates per file, and reranks candidates using vector distance, query-term overlap, path-term overlap, and a small source-type adjustment. Each returned chunk carries a score and human-readable ranking reasons.
5. `RagAgent` builds source references from retrieved paths and line metadata and asks the configured LLM to synthesize only from those snippets.

The ranking pass does not make additional embedding calls. The default retriever top-k remains configurable, while the candidate pool is capped at 50. Existing `RetrievedChunk`, `SourceReference`, and `ResearchReport` fields remain available; added ranking and evidence fields have defaults for caller compatibility.

### Evidence and failure behavior

Source paths, line ranges, language, chunk indexes, snippets, relevance scores, and ranking reasons are preserved where available. Missing or malformed source paths/content are omitted rather than replaced with invented references. Reports identify evidence as `available`, `insufficient`, or `unavailable`; optional warnings, assumptions, and recommendations keep limitations and suggested actions separate from indexed evidence.

Empty results do not trigger LLM synthesis. Retrieval and answer-provider failures are logged and converted into a report that preserves any valid sources already retrieved. The orchestrator's existing summary, answer, source, dependency, insight, and confidence fields are unchanged; downstream agents can continue receiving context through the existing report contract.

