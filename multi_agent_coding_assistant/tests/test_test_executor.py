"""Unit tests for the TestExecutor and ExecuteTestsUseCase.

Uses monkeypatched ``subprocess.run`` so no real pytest is required.
"""

from __future__ import annotations

import subprocess

import pytest

from backend.tools.test_executor import (
    ExecuteTestsUseCase,
    TestExecutionRequest,
    TestExecutionResponse,
    TestExecutor,
    TestExecutorTimeoutError,
)


class _FakeCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_passing_tests(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        # The temp dir cwd exists; the test filename is in args.
        assert "test_generated.py" in args
        assert kwargs["cwd"] is not None
        return _FakeCompleted(stdout="3 passed in 0.5s", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    executor = TestExecutor(timeout_seconds=10.0)
    resp = executor.execute(
        TestExecutionRequest(
            generated_code="def add(a, b):\n    return a + b\n",
            generated_tests="def test_add():\n    assert add(1, 2) == 3\n",
        )
    )

    assert isinstance(resp, TestExecutionResponse)
    assert resp.success is True
    assert resp.exit_code == 0
    assert resp.passed == 3


def test_failing_tests(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        return _FakeCompleted(
            stdout="1 passed, 1 failed in 0.5s", returncode=1
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    executor = TestExecutor(timeout_seconds=10.0)
    resp = executor.execute(
        TestExecutionRequest(
            generated_code="def add(a, b):\n    return a + b\n",
            generated_tests="def test_add():\n    assert add(1, 2) == 3\n",
        )
    )

    assert resp.success is False
    assert resp.exit_code == 1
    assert resp.passed == 1
    assert resp.failed == 1


def test_syntax_error(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        return _FakeCompleted(
            stdout="1 error in 0.5s", returncode=1
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    executor = TestExecutor(timeout_seconds=10.0)
    resp = executor.execute(
        TestExecutionRequest(
            generated_code="def broken(:\n",
            generated_tests="def test_broken():\n    pass\n",
        )
    )

    assert resp.success is False
    assert resp.exit_code == 1


def test_timeout(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    executor = TestExecutor(timeout_seconds=1.0)
    with pytest.raises(TestExecutorTimeoutError):
        executor.execute(
            TestExecutionRequest(
                generated_code="pass\n",
                generated_tests="def test_x():\n    pass\n",
            )
        )


def test_pytest_unavailable(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        raise FileNotFoundError("pytest not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    executor = TestExecutor(timeout_seconds=10.0)
    resp = executor.execute(
        TestExecutionRequest(
            generated_code="pass\n",
            generated_tests="def test_x():\n    pass\n",
        )
    )

    assert resp.success is False
    assert "unavailable" in resp.stderr
    assert resp.passed is None
    assert resp.failed is None


def test_cleanup(monkeypatch, tmp_path) -> None:
    created_dirs = []

    original_mkdtemp = __import__("tempfile").mkdtemp
    def fake_mkdtemp(*args, **kwargs):
        d = tmp_path / "isolated"
        d.mkdir(exist_ok=True)
        created_dirs.append(str(d))
        return str(d)

    def fake_run(args, **kwargs):
        return _FakeCompleted(stdout="1 passed", returncode=0)

    monkeypatch.setattr(__import__("tempfile"), "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(subprocess, "run", fake_run)

    executor = TestExecutor(timeout_seconds=10.0)
    executor.execute(
        TestExecutionRequest(
            generated_code="pass\n",
            generated_tests="def test_x():\n    pass\n",
        )
    )

    # The isolated directory should be removed (files unlinked + rmdir).
    assert not (tmp_path / "isolated").exists()


def test_parse_counts_none_when_unparseable() -> None:
    executor = TestExecutor(timeout_seconds=10.0)
    passed, failed = executor._parse_counts("no counts here")
    assert passed is None
    assert failed is None


def test_use_case_valid_request(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        return _FakeCompleted(stdout="2 passed", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    use_case = ExecuteTestsUseCase(executor=TestExecutor(timeout_seconds=10.0))
    resp = use_case.execute(
        TestExecutionRequest(
            generated_code="pass\n",
            generated_tests="def test_a():\n    pass\n",
        )
    )
    assert resp.success is True


def test_use_case_empty_tests_raises() -> None:
    use_case = ExecuteTestsUseCase(executor=TestExecutor(timeout_seconds=10.0))
    with pytest.raises(ValueError):
        use_case.execute(
            TestExecutionRequest(generated_code="pass\n", generated_tests="   ")
        )


def test_use_case_wrong_request_type() -> None:
    use_case = ExecuteTestsUseCase(executor=TestExecutor(timeout_seconds=10.0))
    with pytest.raises(TypeError):
        use_case.execute("not a request")  # type: ignore[arg-type]
