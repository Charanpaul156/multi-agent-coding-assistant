"""Test execution tool.

Responsibilities:
- Execute generated pytest tests in a SAFE, isolated, temporary directory.
- Write the generated application code to ``generated_code.py``.
- Write the generated test code to ``test_generated.py``.
- Run the fixed ``python -m pytest -q test_generated.py`` command in the
  temporary directory as the working directory (so the test file can import
  the generated application module naturally, e.g. ``from generated_code
  import add``).
- Capture stdout/stderr/exit code/timing.
- Best-effort parse of ``passed``/``failed`` counts (never invent counts).
- Guaranteed cleanup of the temporary directory in ``finally``.

SECURITY:
- Uses a fixed, application-configured command (``python -m pytest -q``).
- Never accepts a command from the API/user.
- No network access, no shell commands, no arbitrary subprocess input.
- Hard timeout enforced via subprocess ``timeout``.
- Temporary directory is always removed in ``finally``.

This module is framework-agnostic and intended for dependency injection.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestExecutionRequest:
    """Request DTO for executing generated pytest tests.

    ``generated_code`` is the application source; ``generated_tests`` is the
    pytest test source. Both are required.
    """

    __test__ = False

    generated_code: str
    generated_tests: str


@dataclass(frozen=True)
class TestExecutionResponse:
    """Response DTO for pytest test execution.

    ``passed``/``failed`` are best-effort counts parsed from pytest output.
    They are ``None`` when the output cannot be parsed reliably.
    """

    __test__ = False

    success: bool
    stdout: str
    stderr: str
    execution_time_ms: float
    exit_code: int
    passed: Optional[int] = None
    failed: Optional[int] = None


class TestExecutorError(RuntimeError):
    """Base error for test executor failures."""

    __test__ = False


class TestExecutorTimeoutError(TestExecutorError):
    """Raised when test execution exceeds the configured timeout."""

    __test__ = False


class TestExecutor:
    """Execute a generated pytest suite in an isolated temporary directory."""

    __test__ = False

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        pytest_command: Optional[list[str]] = None,
    ) -> None:
        self._timeout_seconds = float(timeout_seconds)
        # Fixed command; never sourced from the API/user.
        self._pytest_command = pytest_command or [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
        ]

    def execute(self, request: TestExecutionRequest) -> TestExecutionResponse:
        """Execute the generated tests and return captured output."""

        if not isinstance(request, TestExecutionRequest):
            raise TypeError("request must be a TestExecutionRequest")

        if not isinstance(request.generated_code, str) or not isinstance(
            request.generated_tests, str
        ):
            raise TypeError("generated_code and generated_tests must be strings")
        if not request.generated_code.strip():
            raise ValueError("generated_code must not be empty")
        if not request.generated_tests.strip():
            raise ValueError("generated_tests must not be empty")

        tmp_dir: Optional[Path] = None
        start = time.perf_counter()
        try:
            # Create an isolated temporary directory.
            tmp_dir = Path(tempfile.mkdtemp(prefix="wfa_test_"))
            app_file = tmp_dir / "generated_code.py"
            test_file = tmp_dir / "test_generated.py"

            app_file.write_text(request.generated_code, encoding="utf-8")
            test_file.write_text(request.generated_tests, encoding="utf-8")

            logger.info("TestExecutor: running pytest in %s", tmp_dir)
            completed = subprocess.run(
                [*self._pytest_command, str(test_file.name)],
                cwd=str(tmp_dir),
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                stdin=subprocess.DEVNULL,
            )

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            exit_code = int(completed.returncode)
            elapsed_ms = (time.perf_counter() - start) * 1000

            passed, failed = self._parse_counts(stdout)
            logger.info("TestExecutor: finished (exit=%d)", exit_code)

            return TestExecutionResponse(
                success=exit_code == 0,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=elapsed_ms,
                exit_code=exit_code,
                passed=passed,
                failed=failed,
            )

        except subprocess.TimeoutExpired as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("TestExecutor: timeout")
            raise TestExecutorTimeoutError("Test execution timed out") from exc

        except FileNotFoundError as exc:
            # pytest (or python) not available.
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("TestExecutor: system command not found")
            return TestExecutionResponse(
                success=False,
                stdout="",
                stderr=f"test runner unavailable: {exc}",
                execution_time_ms=elapsed_ms,
                exit_code=-1,
                passed=None,
                failed=None,
            )

        except Exception as exc:  # pragma: no cover
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("TestExecutor: failed")
            return TestExecutionResponse(
                success=False,
                stdout="",
                stderr=str(exc),
                execution_time_ms=elapsed_ms,
                exit_code=-1,
                passed=None,
                failed=None,
            )

        finally:
            # Guaranteed cleanup of the temporary directory.
            if tmp_dir is not None:
                try:
                    for f in tmp_dir.iterdir():
                        try:
                            f.unlink(missing_ok=True)
                        except Exception:  # pragma: no cover
                            logger.warning("Failed to remove %s", f)
                    tmp_dir.rmdir()
                except Exception:  # pragma: no cover
                    logger.warning("Failed to cleanup temp dir: %s", tmp_dir)

    @staticmethod
    def _parse_counts(stdout: str) -> tuple[Optional[int], Optional[int]]:
        """Best-effort parse of ``X passed`` / ``Y failed`` from pytest output.

        Returns (None, None) if the output cannot be parsed reliably.
        """
        if not stdout:
            return None, None

        passed: Optional[int] = None
        failed: Optional[int] = None

        # Patterns like "3 passed, 1 failed" or "2 passed".
        full_match = re.search(
            r"(?P<passed>\d+)\s+passed\s*,\s*(?P<failed>\d+)\s+failed",
            stdout,
        )
        if full_match:
            passed = int(full_match.group("passed"))
            failed = int(full_match.group("failed"))
            return passed, failed

        passed_only = re.search(r"(?P<passed>\d+)\s+passed", stdout)
        if passed_only:
            passed = int(passed_only.group("passed"))

        failed_only = re.search(r"(?P<failed>\d+)\s+failed", stdout)
        if failed_only:
            failed = int(failed_only.group("failed"))

        return passed, failed


class ExecuteTestsUseCase:
    """Framework-agnostic use-case for executing generated pytest tests."""

    def __init__(self, *, executor: TestExecutor) -> None:
        self._executor = executor

    def execute(self, request: TestExecutionRequest) -> TestExecutionResponse:
        if not isinstance(request, TestExecutionRequest):
            raise TypeError("request must be a TestExecutionRequest")

        if not request.generated_code or not request.generated_code.strip():
            raise ValueError("generated_code must not be empty")
        if not request.generated_tests or not request.generated_tests.strip():
            raise ValueError("generated_tests must not be empty")

        return self._executor.execute(request)
