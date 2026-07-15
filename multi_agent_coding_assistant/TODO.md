# TODO - Planner Agent Implementation (Planner only)

## Step 1: Create dataclasses
- [ ] Add `backend/application/planning_use_cases.py` with framework-independent dataclasses:
  - [ ] `ImplementationPlan`
  - [ ] `GeneratePlanRequest`
  - [ ] `GeneratePlanResult`

## Step 2: Implement PlannerAgent (no code generation)
- [ ] Update `agents/planner_agent.py`:
  - [ ] Implement `PlannerAgent.create_plan(prompt: str) -> ImplementationPlan`
  - [ ] Use existing `LLMClient`
  - [ ] Enforce strict JSON-only planning output
  - [ ] Retry once on malformed JSON with an auto-correction prompt
  - [ ] Raise `PlannerAgentError` if still invalid

## Step 3: Create GeneratePlanUseCase
- [ ] Implement `GeneratePlanUseCase` in `backend/application/planning_use_cases.py`
  - [ ] Validate non-empty prompt
  - [ ] Logging: Planning started/completed/failed

## Step 4: Register DI
- [x] Update `backend/api/deps.py` to register:
  - [x] `get_planner_agent`
  - [x] `get_generate_plan_use_case`

## Step 5: Add POST /generate-plan
- [x] Update `backend/api/routes.py` to add endpoint and Pydantic request/response models.

## Step 6: Update Streamlit UI
- [x] Update `frontend/app.py`:
  - [x] Add a separate "Planning" section with st.expander blocks
  - [x] Keep existing "Generate Python code" and "Execute Code" sections unchanged and clearly separated

## Step 7: Verification
- [ ] Run FastAPI import/start checks
- [ ] Run Streamlit import/start checks
- [ ] Manual test `/generate-plan` with 3 prompts
- [ ] Confirm `/generate-code` and `/execute-code` still work
- [ ] Confirm PlannerAgent never returns source code (validate JSON schema)

