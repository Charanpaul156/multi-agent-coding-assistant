"""Python code execution tool.

Responsibilities:
- Execute generated Python code safely (local dev safeguards).
- Persist code to a temporary file.
- Execute via subprocess with timeout.
- Capture stdout/stderr/exit code.
- Cleanup temporary files.

This module is framework-agnostic and intended for dependency injection.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionRequest:
    """Request DTO for executing generated Python code."""

    generated_code: str


@dataclass(frozen=True)
class ExecutionResponse:
    """Response DTO for Python code execution."""

    success: bool
    stdout: str
    stderr: str
    execution_time_ms: float
    exit_code: int


class PythonExecutorError(RuntimeError):
    """Base error for executor failures."""


class PythonExecutorTimeoutError(PythonExecutorError):
    """Raised when code execution exceeds the configured timeout."""


class PythonExecutor:
    """Execute Python code in a subprocess via a temporary file."""

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self._timeout_seconds = float(timeout_seconds)

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        """Execute code and return captured output."""

        if not isinstance(request.generated_code, str):
            raise TypeError("generated_code must be a string")

        code = request.generated_code
        if not code.strip():
            raise ValueError("generated_code must not be empty")

        tmp_path: Optional[Path] = None
        start = time.perf_counter()
        try:
            logger.info("Execution started")

            # Use system temp directory. File exists only transiently.
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(code)
                tmp_file.flush()

            completed = subprocess.run(
                [sys.executable, str(tmp_path)],
                input=None,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                # Disable stdin by ensuring we don't pass any input.
                stdin=subprocess.DEVNULL,
            )

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            exit_code = int(completed.returncode)
            elapsed_ms = (time.perf_counter() - start) * 1000

            logger.info("Execution finished")
            logger.info("Execution time: %.2f ms", elapsed_ms)

            return ExecutionResponse(
                success=exit_code == 0,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=elapsed_ms,
                exit_code=exit_code,
            )

        except subprocess.TimeoutExpired as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("Execution timeout")
            logger.info("Execution time: %.2f ms", elapsed_ms)
            # Signal timeout to the use-case/endpoint.
            raise TimeoutError("Execution timed out") from exc


        except FileNotFoundError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("Execution failed")
            logger.info("Execution time: %.2f ms", elapsed_ms)
            return ExecutionResponse(
                success=False,
                stdout="",
                stderr=str(exc),
                execution_time_ms=elapsed_ms,
                exit_code=-1,
            )

        except Exception as exc:  # pragma: no cover
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("Execution failed")
            logger.info("Execution time: %.2f ms", elapsed_ms)
            return ExecutionResponse(
                success=False,
                stdout="",
                stderr=str(exc),
                execution_time_ms=elapsed_ms,
                exit_code=-1,
            )

        finally:
            # Cleanup temp file.
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:  # pragma: no cover
                    logger.warning("Failed to delete temp file: %s", tmp_path)


class ExecuteCodeUseCase:
    """Framework-agnostic use-case for executing generated Python code."""

    def __init__(self, *, executor: PythonExecutor) -> None:
        self._executor = executor

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        if not isinstance(request, ExecutionRequest):
            raise TypeError("request must be an ExecutionRequest")

        # Validate non-empty code here to keep behavior consistent.
        if not request.generated_code or not request.generated_code.strip():
            raise ValueError("generated_code must not be empty")

        return self._executor.execute(request)

