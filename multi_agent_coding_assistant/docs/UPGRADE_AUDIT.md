# Codebase Audit & Platform Upgrade Specification
**Project:** Multi-Agent AI Software Engineering Platform  
**Target Architecture:** Autonomous Agentic Engineering Platform with Next.js Developer Workspace  
**Status:** Audit Complete  
**Date:** March 2026  

---

## 1. Executive Summary

This audit establishes the baseline architecture, existing capabilities, architectural weaknesses, missing functionality, and transformation roadmap for upgrading the Multi-Agent Coding Assistant into a production-grade AI Software Engineering Platform.

The platform already possesses a solid Clean Architecture foundation (Domain, Application, Infrastructure, API) with strict separation of concerns, 466 passing unit and integration tests, SSE streaming, in-memory job management, deterministic change validation, diff generation, and a human-in-the-loop approval gate.

### Non-Negotiable Upgrading Rules & Constraints
1. **Planner Agent Immutability:** The Planner Agent (`agents/planner_agent.py`, `GeneratePlanUseCase`, prompt contracts, and `ImplementationPlan` DTO) must remain strictly unmodified.
2. **Preserve Working Capabilities:** All existing REST endpoints, SSE streams, job queues, and approval gates must retain backwards compatibility.
3. **Streamlit Preservation:** The Streamlit frontend (`frontend/app.py`, `frontend/workflow_ui.py`) remains intact while the new Next.js enterprise frontend is developed and validated.
4. **Human Approval Gate:** Repository file modification must never bypass user confirmation and cryptographic change validation.
5. **Real Implementations Only:** No mocked or placeholder agents; all agents must perform grounded analysis and generation.

---

## 2. Inventory of Existing Components & Architecture

### 2.1 Agent Inventory
| Agent | Implementation File | Status | Primary Responsibilities | Upgrade Scope |
| :--- | :--- | :--- | :--- | :--- |
| **Planner Agent** | `agents/planner_agent.py` | **Production Baseline** | Generates structured `ImplementationPlan` from prompt. Read-only. | **STRICTLY FROZEN (DO NOT TOUCH)** |
| **Coder Agent** | `agents/coder_agent.py` | Functional | Code generation (`generate_code`) & repo modifications (`generate_changes`). | Enhance multi-file AST context, framework conventions, structured summaries, and resilient JSON generation. |
| **Reviewer Agent** | `agents/code_reviewer_agent.py` | Functional | PR-style code review (`ReviewReport`) analyzing code and test outputs. | Add line-level severity annotations (Critical/High/Med/Low), architectural rule checks, duplicate code detection. |
| **Debugger Agent** | `agents/debugger_agent.py` | Functional | Diagnoses failures (`DebugReport`) and outputs `corrected_code` or `correct_changes`. | Add error taxonomy classification, localized delta repairs, bounded iterative debugging loops. |
| **Test Generator Agent** | `agents/test_generator_agent.py` | Functional | Generates pytest test cases (`TestGenerationReport`). | Expand beyond single file, generate integration tests, edge cases, pytest fixtures, and mock setups. |
| **Documentation Agent** | `agents/documentation_agent.py` | Functional | Generates markdown docs, signatures, and examples (`DocumentationReport`). | Add repository README generation, API specs (OpenAPI/docstrings), architecture documentation, change logs. |
| **RAG / Researcher Agent** | `agents/rag_agent.py` | **Placeholder (`NotImplementedError`)** | Intended for codebase research and semantic query answering. | **Fully Implement**: Codebase Q&A, dependency graph inspection, semantic search synthesis, AST-aware retrieval. |
| **Validator Agent** | *Missing as an agent* | Deterministic infra only | `ChangeValidator` in infrastructure validates paths, allowed roots, and Python AST. | **Implement Validator Agent**: Add static analysis (flake8/ruff/mypy readiness), import verification, config syntax check, structured reports. |
| **Test Executor** | `backend/tools/test_executor.py` | Tool / Use-Case | Isolated subprocess pytest execution with timeout and output capture. | Add structured failure parsing (failed test names, traces, skips), duration metrics, and resource guards. |
| **Python Executor** | `backend/tools/python_executor.py` | Tool / Use-Case | Temporary-file subprocess execution with timeout. | Add execution metrics, memory/duration limits, and isolation boundaries. |

### 2.2 Orchestration & Communication Layer
- **Synchronous Workflow:** `backend/application/workflow_use_cases.py` (`RunWorkflowUseCase`).
  - Sequence: Planner -> Coder -> Test Generator -> Python Executor -> Test Executor -> Reviewer -> Debugger (loop up to `max_iterations`) -> Documentation.
- **Repository Modification Workflow:** `backend/application/modify_repository_use_cases.py` (`ModifyRepositoryUseCase`).
  - Sequence: RAG retrieval -> Planner -> Coder (`generate_changes`) -> ChangeValidator -> DiffGenerator -> (ApprovalStore if preview) -> ChangeApplier -> Test Generator -> Test Executor -> Reviewer -> Debugger (`correct_changes`).
- **Asynchronous Jobs & SSE:**
  - `JobManager` (`backend/application/job_manager.py`) manages threaded background execution states (`queued`, `running`, `completed`, `failed`).
  - `EventBus` (`backend/infrastructure/event_bus.py`) publishes `WorkflowEvent` objects to in-memory subscriber queues.
  - SSE endpoint (`/workflow-jobs/{job_id}/events`) streams realtime updates to clients.

### 2.3 RAG & Repository Indexing
- `rag/file_loader.py` (`RepositoryLoader`): Scans directory trees, filters binary files and excluded patterns (`.git`, `.venv`, `__pycache__`, `.env*`).
- `rag/chunker.py` (`CodeChunker`): Performs line-window chunking with line overlaps.
- `rag/embedder.py` & `rag/embedders/sentence_transformer.py`: Generates 384-dimensional embeddings via `all-MiniLM-L6-v2`.
- `rag/vector_store.py` & `rag/vector_stores/chroma.py`: ChromaDB persistent storage.
- `rag/retriever.py` (`RepositoryRetriever`): Cosine similarity search returning `RetrievedChunk`.
- `rag/context.py`: Formats retrieved chunks into Markdown snippets.

### 2.4 FastAPI Endpoints Inventory
- **System:** `GET /health`
- **Execution & Tools:** `POST /execute-code`
- **Single-Agent Invocation:**
  - `POST /generate-plan` (Planner)
  - `POST /generate-code` (Coder)
  - `POST /review-code` (Reviewer)
  - `POST /debug-code` (Debugger)
  - `POST /generate-tests` (Test Generator)
  - `POST /generate-documentation` (Documentation)
- **RAG & Knowledge:**
  - `POST /rag/index`
  - `POST /rag/search`
  - `GET /rag/status`
- **Repository Modification & Approval Gate:**
  - `POST /modify-repository` (proposes changes, generates diff, creates approval token)
  - `POST /modify-repository/approve` (applies changes using token)
  - `POST /modify-repository/reject` (marks token rejected)
- **Workflows & Background Jobs:**
  - `POST /run-workflow` (synchronous execution)
  - `POST /workflow-jobs` (spawns asynchronous background workflow)
  - `GET /workflow-jobs/{job_id}` (retrieves job status)
  - `GET /workflow-jobs/{job_id}/events` (SSE realtime streaming)

---

## 3. Gap Analysis & Deficiencies

### A. Existing Functionality
- Robust domain models for changes (`ChangeSet`, `FileChange`, `DiffEntry`, `ValidationReport`).
- Secure file application with backups and automatic rollback on failure (`ChangeApplier`).
- Safety controls: `allowed_repository_roots` path jail, protection of sensitive files (`.env`, secrets, git).
- Clean, decoupled test suite with 466 tests covering all use-cases and tools.

### B. Missing Functionality
1. **Validator Agent:** Currently absent. Changes only pass basic Python AST syntax parsing without linting, type validation, import resolution, or config verification.
2. **Researcher / Codebase Q&A Agent:** `agents/rag_agent.py` is an uninvoked placeholder. Developers cannot ask free-form codebase architecture or logic questions grounded in the indexed vector store.
3. **Dynamic Mode Selection in Orchestration:** The current orchestrator forces a fixed linear sequence. It lacks support for specialized developer workflows such as:
   - *Codebase Chat Mode* (Read-only Q&A)
   - *Feature Development Mode* (Plan -> Code -> Validate -> Diff -> Approve)
   - *Bug Fix Mode* (Debug -> Targeted Correction -> Verify)
   - *Code Optimization Mode* (Review -> Refactor -> Benchmark)
   - *Security Audit Mode* (Deep vulnerability and policy inspection)
   - *Test Generation Mode* (Generate and verify unit/integration tests)
   - *Documentation Mode* (Comprehensive doc updates)
4. **Git Workspace Integration:** No capability to inspect active branches, view commits, generate smart commit messages, or draft Pull Request descriptions safely.
5. **Modern Professional Web UI:** The existing Streamlit interface, while functional, lacks:
   - High-performance multi-file Monaco code editor.
   - Resizable 3-panel IDE layout (Explorer/Navigation, Workspace/Chat/Plan, Live Monitoring/Inspector).
   - Side-by-side split visual diff inspection with one-click approval/rejection.
   - Comprehensive test runner dashboard with per-test traces and timing.
   - Real-time dark mode design crafted specifically for software engineers.

### C. Technical Weaknesses
1. **RAG Chunking Limitations:** Naive line-based chunking can sever class/function definitions across chunks, degrading embedding quality.
2. **Single-File Testing Focus:** Test generator and test executor assume a single `generated_code.py` file, lacking robust multi-file test harnesses for full-repo modifications.
3. **Subprocess Sandboxing:** Subprocess execution relies solely on execution timeouts; it needs strict memory limits and execution isolation guarantees.
4. **Error Localization in Reviewer/Debugger:** Findings are often high-level strings without exact file/line anchors, making automated targeted fixes more prone to full-file hallucinations.

### D. Integration Dependencies
- Python 3.10+ with `fastapi`, `pydantic`, `pydantic-settings`, `chromadb`, `sentence-transformers`, `google-genai`, `groq`, `pytest`.
- Node.js v24.19.0 / npm 11.17.0 for the Next.js frontend ecosystem.
- Host Git CLI for safe repository inspection.

### E. Potential Breaking Changes & Safeguards
- **Risk:** Altering endpoint schemas could break existing consumers or tests.
  - *Safeguard:* Retain all existing request/response schemas; add new fields or endpoints with backward-compatible defaults.
- **Risk:** Modifying orchestrator execution might disrupt SSE event structures.
  - *Safeguard:* Preserve existing `WorkflowEvent` types while extending with structured metadata.
- **Risk:** Prematurely removing the Streamlit frontend.
  - *Safeguard:* Keep Streamlit untouched and fully runnable side-by-side until the Next.js platform is verified end-to-end.

---

## 4. Upgrade Strategy & Execution Milestones

### Milestone 1: Agent Intelligence Upgrade (Excluding Planner)
1. **Coder Agent:** Multi-file generation intelligence, framework awareness, structured modification summaries, AST sanity checks.
2. **Researcher / RAG Agent:** Implement real `RagAgent` for codebase Q&A, semantic search ranking, import dependency extraction, source file/location citations.
3. **Validator Agent:** Create dedicated `ValidatorAgent` and use-case for multi-level validation: syntax checking, type analysis, import integrity, and config validation.
4. **Test Generator & Test Executor:** Multi-file pytest support, test scenario categorization (unit, integration, edge, boundary), structured test report with skipped/failed details, execution duration, and resource monitoring.
5. **Reviewer Agent:** Add severity ratings (Critical, Major, Minor, Info), file/line references, anti-pattern detection, security vulnerability checks.
6. **Debugger Agent:** Enhanced error categorization, root-cause localization, minimal targeted patches, bounded retry integration.
7. **Documentation Agent:** Architecture overview, API reference generation, README updater, changelog generation.

### Milestone 2: Advanced Orchestration & Workflow Modes
1. Standardize inter-agent message envelopes and agent status telemetry.
2. Implement dynamic mode-driven orchestration (Feature Dev, Bug Fix, Code Review, Security Audit, Test Suite Generation, Docs).
3. Introduce Git integration service (safe read-only inspection, branch creation, commit message & PR generation).

### Milestone 3: Professional Frontend Transformation (Next.js 15 + React 19 + TypeScript)
1. Initialize Next.js app with Tailwind CSS, shadcn/ui components, Framer Motion, Zustand, and TanStack React Query.
2. Build 3-panel developer workspace with collapsible/resizable layout.
3. Integrate Monaco Editor for multi-file editing, syntax highlighting, and diff viewing.
4. Implement live SSE agent monitoring with step progress, timestamps, and real execution duration.
5. Implement interactive Visual Diff & Approval modal with side-by-side comparison.
6. Implement Testing Dashboard and Repository Intelligence explorer.
7. Connect to FastAPI backend endpoints with complete error resilience.

### Milestone 4: Security Hardening, Verification & Regression Testing
1. Path jail integrity and anti-traversal validation.
2. Prompt injection defense on retrieved repository files.
3. Full backend pytest regression (ensuring 100% of the 466 existing tests pass plus new agent tests).
4. Full Next.js frontend build, typecheck, and end-to-end user workflow verification.
