# Task: Repository-Aware Code Modification

## Implementation Order (approved)

- [x] 1. Domain change models (`backend/domain/change_models.py`)
- [x] 2. Diff generator (`backend/infrastructure/diff_generator.py`)
- [x] 3. Change validator (`backend/infrastructure/change_validation.py`)
- [x] 4. Change applier + rollback (`backend/infrastructure/change_applier.py`)
- [x] 5. CoderAgent.generate_changes() (additive)
- [x] 6. DebuggerAgent.correct_changes() (additive)
- [x] 7. ModifyRepositoryUseCase + DTOs (`backend/application/modify_repository_use_cases.py`)
- [x] 8. DI wiring (`backend/api/deps.py`)
- [x] 9. POST /modify-repository (`backend/api/routes.py`)
- [x] 10. Streamlit "Modify Repository" section (`frontend/app.py`)
- [x] 11. Tests (models, validation, applier, rollback, diff, use-case, API)
- [ ] 12. Full verification (compileall, pytest, pip check)

## Decisions (approved)
- Full-file `new_content`, no patches in v1.
- Only `create` + `modify`; NO `delete` in v1.
- Keep `generate_code()` unchanged; add `generate_changes()`.
- Dedicated ChangeValidator + ChangeApplier; agents never write files.
- Reuse RagConfig allowed-root / excluded-file / binary security.
- Unified diffs from old/new content for human review.
- Mandatory dry-run; explicit Apply action; no auto-apply.
- Re-verify original hash immediately before apply; reject on mismatch.
- Auto-rollback on any mid-transaction failure.
- No shell/subprocess for file modification. No Git. No new framework.
- Preserve all existing endpoints and 71+ passing tests.

## Verification
- [ ] `python -m compileall -q .`
- [ ] `python -m pytest -q`
- [ ] `python -m pip check`
- [ ] Manual: POST /modify-repository through Swagger
- [ ] Manual: Streamlit "Modify Repository" section
