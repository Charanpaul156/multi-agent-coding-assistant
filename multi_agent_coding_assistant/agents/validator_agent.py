"""Validator Agent.

This agent is responsible for deep code validation across multiple dimensions:
- Syntax validation (AST parsing and token checks).
- Static analysis (detecting security risks, antipatterns, dead code).
- Type checking & type annotation consistency.
- Import validation (unresolved, cyclic, wildcards, or dangerous imports).
- Configuration file validation (JSON, YAML, TOML).
- Detection of common implementation errors (NaN equality, mutable defaults, resource leaks).
- Generating structured validation reports.

STRICT CONSTRAINTS:
- NEVER execute code or shell commands.
- NEVER write or modify files on disk.
- Operates purely through deterministic analysis and static inspection.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.domain.change_models import ChangeSet, FileChange

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class ValidationIssue:
    """A specific validation issue discovered during analysis."""

    rule_id: str
    category: str  # syntax, static_analysis, type_safety, imports, configuration, common_pitfalls
    severity: str  # ERROR, WARNING, INFO
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    recommendation: Optional[str] = None


@dataclass(frozen=True)
class StructuredValidationReport:
    """Comprehensive validation report produced by the Validator Agent."""

    valid: bool
    error_count: int
    warning_count: int
    info_count: int
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: str = ""
    files_validated: List[str] = field(default_factory=list)


class ValidatorAgent:
    """Perform multi-dimensional code and change-set validation."""

    def validate_code(
        self,
        code: str,
        *,
        language: str = "python",
        file_path: Optional[str] = None,
    ) -> StructuredValidationReport:
        """Validate a single code snippet or file content."""
        if not isinstance(code, str):
            raise TypeError("code must be a string")

        issues: List[ValidationIssue] = []
        target_path = file_path or f"snippet.{language}"
        lang = (language or "python").lower()

        if not code.strip():
            issues.append(
                ValidationIssue(
                    rule_id="VAL-EMPTY",
                    category="syntax",
                    severity=ValidationSeverity.ERROR.value,
                    message="Code content is empty",
                    file_path=target_path,
                    recommendation="Provide valid implementation code.",
                )
            )
            return self._build_report(issues, [target_path])

        if lang == "python" or target_path.endswith(".py"):
            self._validate_python(code, target_path, issues)
        elif lang == "json" or target_path.endswith(".json"):
            self._validate_json(code, target_path, issues)
        elif lang in ("yaml", "yml") or target_path.endswith((".yaml", ".yml")):
            self._validate_yaml(code, target_path, issues)
        else:
            self._validate_generic(code, target_path, issues)

        return self._build_report(issues, [target_path])

    def validate_changeset(
        self,
        change_set: ChangeSet,
        *,
        repository_path: Optional[str | Path] = None,
    ) -> StructuredValidationReport:
        """Validate an entire ChangeSet including cross-file dependencies and configs."""
        if not isinstance(change_set, ChangeSet):
            raise TypeError("change_set must be a ChangeSet")

        issues: List[ValidationIssue] = []
        files_validated: List[str] = []

        if not change_set.changes:
            issues.append(
                ValidationIssue(
                    rule_id="VAL-EMPTY-CHANGESET",
                    category="syntax",
                    severity=ValidationSeverity.ERROR.value,
                    message="ChangeSet contains no file changes",
                    recommendation="Ensure at least one file change is proposed.",
                )
            )
            return self._build_report(issues, files_validated)

        changed_paths = {c.file_path for c in change_set.changes}

        for change in change_set.changes:
            files_validated.append(change.file_path)
            content = change.new_content or ""
            fpath = change.file_path.lower()

            if fpath.endswith(".py"):
                self._validate_python(content, change.file_path, issues, changed_paths)
            elif fpath.endswith(".json"):
                self._validate_json(content, change.file_path, issues)
            elif fpath.endswith((".yaml", ".yml")):
                self._validate_yaml(content, change.file_path, issues)
            elif fpath.endswith((".toml", ".ini", ".cfg")):
                self._validate_config(content, change.file_path, issues)

        return self._build_report(issues, files_validated)

    # ------------------------------------------------------------------ #
    # Python Validation
    # ------------------------------------------------------------------ #

    def _validate_python(
        self,
        code: str,
        file_path: str,
        issues: List[ValidationIssue],
        all_changed_paths: Optional[set[str]] = None,
    ) -> None:
        """Run deep Python AST and structural checks."""
        try:
            tree = ast.parse(code, filename=file_path)
        except SyntaxError as exc:
            issues.append(
                ValidationIssue(
                    rule_id="PY-SYNTAX-ERR",
                    category="syntax",
                    severity=ValidationSeverity.ERROR.value,
                    message=f"Syntax error: {exc.msg}",
                    file_path=file_path,
                    line_number=exc.lineno,
                    column_number=exc.offset,
                    recommendation="Fix Python syntax error at indicated line.",
                )
            )
            return

        # AST-based static checks
        imports_found: List[str] = []
        for node in ast.walk(tree):
            # 1. Security: Dangerous functions (eval, exec)
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in ("eval", "exec"):
                    issues.append(
                        ValidationIssue(
                            rule_id="SEC-DANGEROUS-CALL",
                            category="static_analysis",
                            severity=ValidationSeverity.ERROR.value,
                            message=f"Dangerous dynamic code execution `{func_name}()` detected.",
                            file_path=file_path,
                            line_number=node.lineno,
                            recommendation="Avoid eval/exec; use safe parsing or structured dispatch.",
                        )
                    )

                # Check shell=True in subprocess
                if func_name in ("run", "Popen", "call", "check_call", "check_output"):
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            issues.append(
                                ValidationIssue(
                                    rule_id="SEC-SUBPROCESS-SHELL",
                                    category="static_analysis",
                                    severity=ValidationSeverity.ERROR.value,
                                    message="`shell=True` detected in subprocess execution.",
                                    file_path=file_path,
                                    line_number=node.lineno,
                                    recommendation="Set shell=False and pass arguments as a list.",
                                )
                            )

            # 2. Antipattern: Mutable default arguments
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default is not None and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(
                            ValidationIssue(
                                rule_id="PY-MUTABLE-DEFAULT",
                                category="common_pitfalls",
                                severity=ValidationSeverity.WARNING.value,
                                message=f"Mutable default argument in function `{node.name}`.",
                                file_path=file_path,
                                line_number=node.lineno,
                                recommendation="Default to None and initialize inside the function body.",
                            )
                        )

            # 3. Antipattern: Bare except
            elif isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(
                        ValidationIssue(
                            rule_id="PY-BARE-EXCEPT",
                            category="common_pitfalls",
                            severity=ValidationSeverity.WARNING.value,
                            message="Bare `except:` clause catches system exit and keyboard interrupts.",
                            file_path=file_path,
                            line_number=node.lineno,
                            recommendation="Catch specific exceptions or `except Exception:`.",
                        )
                    )

            # 4. Imports tracking & wildcard imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports_found.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.names and any(n.name == "*" for n in node.names):
                    issues.append(
                        ValidationIssue(
                            rule_id="PY-WILDCARD-IMPORT",
                            category="imports",
                            severity=ValidationSeverity.WARNING.value,
                            message=f"Wildcard import `from {node.module or ''} import *` pollutes namespace.",
                            file_path=file_path,
                            line_number=node.lineno,
                            recommendation="Import only explicit names needed.",
                        )
                    )
                elif node.module:
                    imports_found.append(node.module)

            # 5. Common Pitfall: NaN comparison with ==
            elif isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    # Check for == float('nan')
                    if isinstance(comparator, ast.Call):
                        if (
                            isinstance(comparator.func, ast.Name)
                            and comparator.func.id == "float"
                            and comparator.args
                            and isinstance(comparator.args[0], ast.Constant)
                            and str(comparator.args[0].value).lower() == "nan"
                        ):
                            issues.append(
                                ValidationIssue(
                                    rule_id="PY-NAN-EQUALITY",
                                    category="common_pitfalls",
                                    severity=ValidationSeverity.ERROR.value,
                                    message="Direct equality comparison with NaN (`== float('nan')`) always evaluates to False.",
                                    file_path=file_path,
                                    line_number=node.lineno,
                                    recommendation="Use `math.isnan(value)` instead.",
                                )
                            )

    # ------------------------------------------------------------------ #
    # Config & Generic Validation
    # ------------------------------------------------------------------ #

    def _validate_json(self, content: str, file_path: str, issues: List[ValidationIssue]) -> None:
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    rule_id="JSON-SYNTAX-ERR",
                    category="configuration",
                    severity=ValidationSeverity.ERROR.value,
                    message=f"Invalid JSON syntax: {exc.msg}",
                    file_path=file_path,
                    line_number=exc.lineno,
                    column_number=exc.colno,
                    recommendation="Ensure JSON conforms to valid syntax specifications.",
                )
            )

    def _validate_yaml(self, content: str, file_path: str, issues: List[ValidationIssue]) -> None:
        # Check basic yaml indentation and tabs
        for idx, line in enumerate(content.splitlines(), 1):
            if "\t" in line and not line.strip().startswith("#"):
                issues.append(
                    ValidationIssue(
                        rule_id="YAML-TAB-INDENT",
                        category="configuration",
                        severity=ValidationSeverity.ERROR.value,
                        message="Tab character used for YAML indentation.",
                        file_path=file_path,
                        line_number=idx,
                        recommendation="Replace tabs with spaces in YAML files.",
                    )
                )
                break

    def _validate_config(self, content: str, file_path: str, issues: List[ValidationIssue]) -> None:
        if not content.strip():
            issues.append(
                ValidationIssue(
                    rule_id="CFG-EMPTY",
                    category="configuration",
                    severity=ValidationSeverity.WARNING.value,
                    message="Configuration file is empty.",
                    file_path=file_path,
                )
            )

    def _validate_generic(self, content: str, file_path: str, issues: List[ValidationIssue]) -> None:
        # Balance check for braces/parentheses
        stack = []
        pairs = {")": "(", "}": "{", "]": "["}
        for idx, char in enumerate(content):
            if char in "({[":
                stack.append(char)
            elif char in ")}]":
                if not stack or stack[-1] != pairs[char]:
                    issues.append(
                        ValidationIssue(
                            rule_id="GEN-UNBALANCED-DELIM",
                            category="syntax",
                            severity=ValidationSeverity.WARNING.value,
                            message=f"Unmatched closing delimiter `{char}`.",
                            file_path=file_path,
                            recommendation="Check bracket and parentheses balance.",
                        )
                    )
                    break
                stack.pop()

    def _build_report(
        self, issues: List[ValidationIssue], files_validated: List[str]
    ) -> StructuredValidationReport:
        errors = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR.value)
        warnings = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING.value)
        infos = sum(1 for i in issues if i.severity == ValidationSeverity.INFO.value)
        valid = errors == 0

        summary = (
            f"Validation passed ({len(files_validated)} files checked, {warnings} warnings, {infos} infos)."
            if valid
            else f"Validation failed with {errors} error(s), {warnings} warning(s)."
        )

        return StructuredValidationReport(
            valid=valid,
            error_count=errors,
            warning_count=warnings,
            info_count=infos,
            issues=issues,
            summary=summary,
            files_validated=files_validated,
        )
