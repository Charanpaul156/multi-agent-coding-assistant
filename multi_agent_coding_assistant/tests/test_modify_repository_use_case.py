"""Tests for the ModifyRepositoryUseCase (application layer).

Uses fakes for the retriever, planner, coder, validator, applier, test
generation, test execution, review, and debugger. All file operations happen
in temporary directories; the real project is never touched.
"""

from __future__ import annotations

import pytest

from agents.planner_agent import ImplementationPlan
from backend.application.modify_repository_use_cases import (
    ModifyRepositoryRequest,
    ModifyRepositoryUseCase,
    ModifyRepositoryStatus,
)
from backend.domain.change_models import (
    ApplicationResult,
    ChangeOperation,
    ChangeSet,
    ChangeValidationResult,
    FileChange,
    ValidationReport,
)
from rag.config import RagConfig


@pytest.fixture
def repo_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    return root


@pytest.fixture
def config(repo_root):
    return RagConfig(allowed_repository_roots=(str(repo_root),))


class FakeRetriever:
    def __init__(self):
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        return type("R", (), {"results": []})


class FakePlanner:
    def __init__(self, plan=None):
        self.plan = plan or ImplementationPlan(
            problem_summary="p",
            project_type="t",
            requirements=["r"],
            modules=["m"],
            functions=["f"],
            classes=[],
            external_libraries=[],
            database_needed=False,
            api_needed=[],
            algorithm="none",
            edge_cases=[],
            estimated_complexity="low",
            future_improvements=[],
        )
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        return type("R", (), {"plan": self.plan})


class FakeCoder:
    def __init__(self, change_set):
        self.change_set = change_set
        self.calls = 0

    def generate_changes(self, prompt, *, retrieved_context=None, plan_summary=None):
        self.calls += 1
        return self.change_set


class FakeValidator:
    def __init__(self, report):
        self.report = report

    def validate(self, change_set):
        return self.report


class FakeApplier:
    def __init__(self, result):
        self.result = result
        self.last_dry_run = None

    def apply(self, change_set, *, dry_run=False):
        self.last_dry_run = dry_run
        return self.result


class FakeTestGen:
    def __init__(self, code="import pytest\n"):
        self.code = code

    def execute(self, request):
        return type("R", (), {"report": type("Rep", (), {"generated_test_code": self.code})()})


class FakeTestExec:
    def __init__(self, success=True, exit_code=0):
        self.success = success
        self.exit_code = exit_code

    def execute(self, request):
        return type(
            "R",
            (),
            {
                "success": self.success,
                "exit_code": self.exit_code,
                "stdout": "",
                "stderr": "",
                "passed": 1,
                "failed": 0 if self.success else 1,
            },
        )()


class FakeReview:
    def __init__(self, score=90):
        self.score = score

    def execute(self, request):
        return type("R", (), {"report": type("Rep", (), {"overall_score": self.score, "final_summary": "ok"})()})


class FakeDebugger:
    def __init__(self, corrected):
        self.corrected = corrected

    def correct_changes(self, change_set, *, feedback, retrieved_context=None):
        return self.corrected


def _valid_report():
    r = ChangeValidationResult(True, "a.py", "create")
    return ValidationReport(valid=True, results=[r])


def _inval_report():
    r = ChangeValidationResult(False, "a.py", "create", ["unsafe path"])
    return ValidationReport(valid=False, results=[r])


def _create():
    return FileChange(
        file_path="a.py",
        operation=ChangeOperation.CREATE,
        new_content="x = 1\n",
    )


def _build_use_case(
    *,
    coder_set=None,
    validator_report=None,
    applier_result=None,
    test_gen=None,
    test_exec=None,
    review=None,
    debugger=None,
    config=None,
):
    return ModifyRepositoryUseCase(
        config=config,
        retriever_use_case=FakeRetriever(),
        plan_use_case=FakePlanner(),
        coder_agent=FakeCoder(coder_set or ChangeSet(changes=[_create()])),
        validator=FakeValidator(validator_report or _valid_report()),
        applier=FakeApplier(applier_result or ApplicationResult(success=True)),
        test_generation_use_case=test_gen or FakeTestGen(),
        test_execution_use_case=test_exec or FakeTestExec(),
        review_use_case=review or FakeReview(),
        debugger_agent=debugger,
        max_iterations=3,
    )


def _request(config, *, dry_run=False):
    return ModifyRepositoryRequest(
        repository_path=str(config.allowed_repository_roots[0]),
        request="add feature",
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_empty_repository_path_rejected(config):
    uc = _build_use_case(config=config)
    with pytest.raises(ValueError):
        uc.execute(ModifyRepositoryRequest(repository_path="", request="x"))


def test_empty_request_rejected(config):
    uc = _build_use_case(config=config)
    with pytest.raises(ValueError):
        uc.execute(ModifyRepositoryRequest(repository_path=str(config.allowed_repository_roots[0]), request=""))


def test_path_outside_allowed_root_rejected(config):
    uc = _build_use_case(config=config)
    # Use a path that is not under the allowed root.
    with pytest.raises(ValueError):
        uc.execute(ModifyRepositoryRequest(repository_path="C:/somewhere/else", request="x"))


def test_no_allowed_roots_disables(config):
    cfg = RagConfig(allowed_repository_roots=())
    uc = _build_use_case(config=cfg)
    with pytest.raises(ValueError):
        uc.execute(ModifyRepositoryRequest(repository_path="C:/x", request="x"))


# ---------------------------------------------------------------------------
# Successful dry run
# ---------------------------------------------------------------------------


def test_dry_run_returns_proposed(config):
    uc = _build_use_case(config=config)
    result = uc.execute(_request(config, dry_run=True))
    assert result.success is True
    assert result.status == ModifyRepositoryStatus.PROPOSED
    assert result.dry_run is True
    assert result.change_set is not None
    assert len(result.diffs) == 1
    # No application result in dry run.
    assert result.application is None


def test_dry_run_does_not_apply(config):
    applier = FakeApplier(ApplicationResult(success=True))
    uc = _build_use_case(config=config, applier_result=applier.result)
    # dry_run path avoids applier entirely.
    result = uc.execute(_request(config, dry_run=True))
    assert result.success is True
    assert result.application is None


# ---------------------------------------------------------------------------
# Validation failure
# ---------------------------------------------------------------------------


def test_validation_failure(config):
    uc = _build_use_case(
        config=config,
        validator_report=_inval_report(),
    )
    result = uc.execute(_request(config, dry_run=False))
    assert result.success is False
    assert result.status == ModifyRepositoryStatus.VALIDATION_FAILED
    assert result.validation is not None
    assert result.validation.valid is False


# ---------------------------------------------------------------------------
# Successful apply
# ---------------------------------------------------------------------------


def test_successful_apply(config):
    uc = _build_use_case(config=config)
    result = uc.execute(_request(config, dry_run=False))
    assert result.success is True
    assert result.status == ModifyRepositoryStatus.APPLIED
    assert result.application is not None
    assert result.application.success is True
    assert len(result.iterations) == 1


# ---------------------------------------------------------------------------
# Application failure
# ---------------------------------------------------------------------------


def test_application_failure(config):
    uc = _build_use_case(
        config=config,
        applier_result=ApplicationResult(success=False, error="apply boom"),
    )
    result = uc.execute(_request(config, dry_run=False))
    assert result.success is False
    assert result.status == ModifyRepositoryStatus.APPLICATION_FAILED
    assert "apply boom" in (result.error or "")


# ---------------------------------------------------------------------------
# Test failure -> debugger -> correction
# ---------------------------------------------------------------------------


def test_test_failure_triggers_debugger(config):
    corrected = ChangeSet(changes=[_create()])
    debugger = FakeDebugger(corrected)
    uc = _build_use_case(
        config=config,
        test_exec=FakeTestExec(success=False, exit_code=1),
        debugger=debugger,
    )
    result = uc.execute(_request(config, dry_run=False))
    # Debugger produces corrected changes; re-apply succeeds.
    assert result.success is True
    assert result.status == ModifyRepositoryStatus.APPLIED


def test_max_iterations_reached(config):
    corrected = ChangeSet(changes=[_create()])
    configured = config
    uc = ModifyRepositoryUseCase(
        config=configured,
        retriever_use_case=FakeRetriever(),
        plan_use_case=FakePlanner(),
        coder_agent=FakeCoder(ChangeSet(changes=[_create()])),
        validator=FakeValidator(_valid_report()),
        applier=FakeApplier(ApplicationResult(success=True)),
        test_generation_use_case=FakeTestGen(),
        test_execution_use_case=FakeTestExec(success=False, exit_code=1),
        review_use_case=FakeReview(score=90),
        debugger_agent=FakeDebugger(corrected),
        max_iterations=1,
    )
    result = uc.execute(_request(configured, dry_run=False))
    assert result.success is False
    assert result.status == ModifyRepositoryStatus.MAX_ITERATIONS_REACHED


def test_debugger_not_available(config):
    uc = _build_use_case(
        config=config,
        test_exec=FakeTestExec(success=False, exit_code=1),
        debugger=None,
    )
    result = uc.execute(_request(config, dry_run=False))
    assert result.success is False
    assert result.status == ModifyRepositoryStatus.DEBUGGER_FAILED


def test_review_failure_triggers_debugger(config):
    corrected = ChangeSet(changes=[_create()])
    uc = _build_use_case(
        config=config,
        review=FakeReview(score=30),
        debugger=FakeDebugger(corrected),
    )
    result = uc.execute(_request(config, dry_run=False))
    assert result.success is True
    assert result.status == ModifyRepositoryStatus.APPLIED


# ---------------------------------------------------------------------------
# Coder failure
# ---------------------------------------------------------------------------


def test_coder_failure_returns_validation_failed(config):
    class BoomCoder:
        def generate_changes(self, prompt, *, retrieved_context=None, plan_summary=None):
            raise RuntimeError("coder boom")

    uc = ModifyRepositoryUseCase(
        config=config,
        retriever_use_case=FakeRetriever(),
        plan_use_case=FakePlanner(),
        coder_agent=BoomCoder(),
        validator=FakeValidator(_valid_report()),
        applier=FakeApplier(ApplicationResult(success=True)),
    )
    result = uc.execute(_request(config, dry_run=True))
    assert result.success is False
    assert "change generation failed" in (result.error or "")
