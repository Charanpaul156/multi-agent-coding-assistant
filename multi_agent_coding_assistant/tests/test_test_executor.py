"""Unit tests for the TestExecutor and ExecuteTestsUseCase.

Uses monkeypatched ``subprocess.run`` so no real pytest is required.
"""

from __future__ import annotations

from pathlib import Path
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


def test_math_utils_import_reproduction(monkeypatch) -> None:
    written_files = {}

    def fake_run(args, **kwargs):
        cwd = kwargs["cwd"]
        from pathlib import Path
        for p in Path(cwd).glob("**/*"):
            if p.is_file():
                written_files[p.name] = p.read_text(encoding="utf-8")
        assert "math_utils.py" in written_files
        assert "from math_utils import add, divide" in written_files["test_generated.py"]
        return _FakeCompleted(stdout="2 passed in 0.1s", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    executor = TestExecutor(timeout_seconds=10.0)
    resp = executor.execute(
        TestExecutionRequest(
            generated_code="def add(a, b):\n    return a + b\ndef divide(a, b):\n    return a / b\n",
            generated_tests="from math_utils import add, divide\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            files={"math_utils.py": "def add(a, b):\n    return a + b\ndef divide(a, b):\n    return a / b\n"},
        )
    )

    assert resp.success is True
    assert resp.passed == 2


def test_multiple_changed_files(monkeypatch) -> None:
    written_files = {}

    def fake_run(args, **kwargs):
        cwd = kwargs["cwd"]
        from pathlib import Path
        for p in Path(cwd).glob("**/*"):
            if p.is_file():
                written_files[p.name] = p.read_text(encoding="utf-8")
        assert "math_utils.py" in written_files
        assert "string_utils.py" in written_files
        return _FakeCompleted(stdout="2 passed in 0.1s", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    executor = TestExecutor(timeout_seconds=10.0)
    resp = executor.execute(
        TestExecutionRequest(
            generated_tests="from math_utils import add\nfrom string_utils import cap\n",
            files={
                "math_utils.py": "def add(a, b):\n    return a + b\n",
                "string_utils.py": "def cap(s):\n    return s.upper()\n",
            },
        )
    )

    assert resp.success is True
    assert resp.passed == 2


def test_path_traversal_prevention(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        return _FakeCompleted(stdout="1 passed", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    executor = TestExecutor(timeout_seconds=10.0)
    with pytest.raises(ValueError, match="Path traversal"):
        executor.execute(
            TestExecutionRequest(
                generated_tests="def test_x(): pass\n",
                files={"../../etc/passwd": "malicious content"},
            )
        )


def test_standalone_fallback_import_injected_and_executes_real_pytest() -> None:
    """Verify fallback import allows standalone tests to execute without NameError."""
    executor = TestExecutor(timeout_seconds=10.0)
    resp = executor.execute(
        TestExecutionRequest(
            generated_code="def multiply(a, b):\n    return a * b\n",
            generated_tests="def test_multiply():\n    assert multiply(2, 3) == 6\n",
        )
    )

    assert resp.success is True
    assert resp.exit_code == 0
    assert resp.passed == 1


def test_standalone_fallback_import_injected_content(monkeypatch) -> None:
    """Verify fallback import is prepended to test_generated.py when import is missing."""
    written_test_content = None

    def fake_run(args, **kwargs):
        nonlocal written_test_content
        test_file = Path(kwargs["cwd"]) / "test_generated.py"
        written_test_content = test_file.read_text(encoding="utf-8")
        return _FakeCompleted(stdout="1 passed", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    executor = TestExecutor(timeout_seconds=10.0)
    resp = executor.execute(
        TestExecutionRequest(
            generated_code="def multiply(a, b):\n    return a * b\n",
            generated_tests="def test_multiply():\n    assert multiply(2, 3) == 6\n",
        )
    )

    assert resp.success is True
    assert written_test_content is not None
    assert "from generated_code import *" in written_test_content
    assert "def test_multiply():" in written_test_content


def test_standalone_no_duplicate_import_when_already_imported(monkeypatch) -> None:
    """Verify TestExecutor does NOT add duplicate fallback import if already present."""
    written_test_content = None

    def fake_run(args, **kwargs):
        nonlocal written_test_content
        test_file = Path(kwargs["cwd"]) / "test_generated.py"
        written_test_content = test_file.read_text(encoding="utf-8")
        return _FakeCompleted(stdout="1 passed", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    executor = TestExecutor(timeout_seconds=10.0)
    resp = executor.execute(
        TestExecutionRequest(
            generated_code="def multiply(a, b):\n    return a * b\n",
            generated_tests=(
                "from generated_code import multiply\n\n"
                "def test_multiply():\n"
                "    assert multiply(2, 3) == 6\n"
            ),
        )
    )

    assert resp.success is True
    assert written_test_content is not None
    assert "from generated_code import *" not in written_test_content
    assert written_test_content.count("generated_code") == 1


def test_multi_file_does_not_inject_generated_code_fallback(monkeypatch) -> None:
    """Verify multi-file execution behavior remains unchanged without injecting fallback."""
    written_test_content = None

    def fake_run(args, **kwargs):
        nonlocal written_test_content
        test_file = Path(kwargs["cwd"]) / "test_generated.py"
        written_test_content = test_file.read_text(encoding="utf-8")
        return _FakeCompleted(stdout="1 passed", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    executor = TestExecutor(timeout_seconds=10.0)
    resp = executor.execute(
        TestExecutionRequest(
            generated_tests="def test_something():\n    assert True\n",
            files={"calc.py": "def multiply(a, b): return a * b\n"},
        )
    )

    assert resp.success is True
    assert written_test_content is not None
    assert "from generated_code import *" not in written_test_content

