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

import ast
import logging
import re
import shutil
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
    pytest test source. ``files`` is an optional mapping of relative file paths
    to file contents for multi-file or named module execution.
    """

    __test__ = False

    generated_code: str = ""
    generated_tests: str = ""
    files: Optional[dict[str, str]] = None


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

        if request.files is not None:
            if not isinstance(request.files, dict):
                raise TypeError("files must be a dictionary if provided")
            for k, v in request.files.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise TypeError("keys and values in files must be strings")

        has_files = bool(request.files)
        has_code = bool(request.generated_code.strip())

        if not has_files and not has_code:
            raise ValueError("Either generated_code or files must be provided and non-empty")
        if not request.generated_tests.strip():
            raise ValueError("generated_tests must not be empty")

        tmp_dir: Optional[Path] = None
        start = time.perf_counter()
        try:
            # Create an isolated temporary directory.
            tmp_dir = Path(tempfile.mkdtemp(prefix="wfa_test_"))

            files_to_write: dict[str, str] = {}

            # 1. Add explicit files if provided.
            if request.files:
                for rel_path, content in request.files.items():
                    files_to_write[rel_path] = content

            # 2. Parse embedded sections from generated_code if files not explicitly provided.
            parsed_sections: dict[str, str] = {}
            if not request.files and has_code:
                parsed_sections = self._parse_code_sections(request.generated_code)
                if parsed_sections:
                    files_to_write.update(parsed_sections)

            # 3. If target module file is still missing, infer module from test imports.
            if has_code:
                imported_modules = self._extract_imported_modules(request.generated_tests)
                for mod in imported_modules:
                    mod_filename = f"{mod}.py"
                    if mod_filename not in files_to_write and mod not in ("generated_code", "pytest"):
                        files_to_write[mod_filename] = request.generated_code

                # Legacy/Fallback: always ensure generated_code.py is present if not already added.
                if "generated_code.py" not in files_to_write:
                    files_to_write["generated_code.py"] = request.generated_code

            # Write all files safely preventing path traversal.
            for rel_path, content in files_to_write.items():
                self._safe_write_file(tmp_dir, rel_path, content)

            test_content = request.generated_tests
            # Standalone generated-code execution: if generated_code.py is the application module
            # and the test suite does not import it, prepend a safe fallback import.
            is_standalone = (not request.files) and (not parsed_sections) and has_code
            if is_standalone and not self._has_generated_code_import(test_content):
                test_content = self._inject_generated_code_fallback(test_content)

            test_file = tmp_dir / "test_generated.py"
            test_file.write_text(test_content, encoding="utf-8")

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

        except ValueError:
            raise

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
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:  # pragma: no cover
                    logger.warning("Failed to cleanup temp dir: %s", tmp_dir)

    @staticmethod
    def _safe_write_file(tmp_dir: Path, rel_path_str: str, content: str) -> Path:
        """Write a file into tmp_dir, preventing path traversal."""
        clean_path_str = rel_path_str.replace("\\", "/").lstrip("/")
        target_path = (tmp_dir / clean_path_str).resolve()
        base_dir = tmp_dir.resolve()

        try:
            target_path.relative_to(base_dir)
        except ValueError:
            raise ValueError(f"Path traversal attempt detected in file path: {rel_path_str}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")

        # Create __init__.py files in nested directories inside tmp_dir
        curr = target_path.parent
        while curr != base_dir and base_dir in curr.parents:
            init_file = curr / "__init__.py"
            if not init_file.exists():
                init_file.write_text("", encoding="utf-8")
            curr = curr.parent

        return target_path

    @staticmethod
    def _parse_code_sections(code: str) -> dict[str, str]:
        """Extract sections marked like '# === path/to/file.py (operation) ==='."""
        sections: dict[str, str] = {}
        pattern = re.compile(r"^# ===\s*(.*?)(?:\s*\([^)]*\))?\s*===\s*$", re.MULTILINE)
        matches = list(pattern.finditer(code))
        if not matches:
            return sections

        for i, match in enumerate(matches):
            raw_path = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
            section_content = code[start:end].strip()
            if raw_path and (raw_path.endswith(".py") or "." in raw_path):
                sections[raw_path] = section_content

        return sections

    @staticmethod
    def _extract_imported_modules(test_code: str) -> list[str]:
        """Extract top-level custom imported module names from test source."""
        modules: list[str] = []
        try:
            tree = ast.parse(test_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top_mod = alias.name.split(".")[0]
                        if top_mod not in modules:
                            modules.append(top_mod)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top_mod = node.module.split(".")[0]
                        if top_mod not in modules:
                            modules.append(top_mod)
        except Exception:
            for match in re.finditer(r"^\s*(?:from|import)\s+([a-zA-Z0-9_]+)", test_code, re.MULTILINE):
                mod = match.group(1)
                if mod not in modules:
                    modules.append(mod)

        stdlib_names = getattr(sys, "stdlib_module_names", set())
        excluded = stdlib_names | {"pytest", "unittest", "conftest"}
        return [m for m in modules if m not in excluded]

    @staticmethod
    def _has_generated_code_import(test_code: str) -> bool:
        """Check if test code imports or references the generated_code module."""
        try:
            tree = ast.parse(test_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top_mod = alias.name.split(".")[0]
                        if top_mod == "generated_code":
                            return True
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split(".")[0] == "generated_code":
                        return True
        except Exception:
            if re.search(
                r"^\s*(?:from\s+generated_code(?:\.|\s)|import\s+generated_code(?:\.|\s|,|$))",
                test_code,
                re.MULTILINE,
            ):
                return True
        return False

    @staticmethod
    def _inject_generated_code_fallback(test_code: str) -> str:
        """Prepend 'from generated_code import *' safely respecting __future__ imports."""
        import_stmt = "from generated_code import *\n"
        try:
            tree = ast.parse(test_code)
            last_future_lineno = 0
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    end_line = getattr(node, "end_lineno", None) or node.lineno
                    last_future_lineno = max(last_future_lineno, end_line)
                elif isinstance(node, ast.Expr) and isinstance(
                    getattr(node, "value", None), ast.Constant
                ) and isinstance(node.value.value, str):
                    # Module docstring
                    pass
                else:
                    break

            if last_future_lineno > 0:
                lines = test_code.splitlines(keepends=True)
                return (
                    "".join(lines[:last_future_lineno])
                    + "\n"
                    + import_stmt
                    + "".join(lines[last_future_lineno:])
                )
        except Exception:
            pass

        return import_stmt + "\n" + test_code

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
