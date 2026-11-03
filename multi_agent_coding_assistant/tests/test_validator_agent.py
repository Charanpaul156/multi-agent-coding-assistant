"""Unit tests for the Validator Agent."""

from __future__ import annotations

import pytest

from agents.validator_agent import (
    StructuredValidationReport,
    ValidationIssue,
    ValidationSeverity,
    ValidatorAgent,
)
from backend.domain.change_models import ChangeOperation, ChangeSet, FileChange


def test_validator_python_syntax_valid() -> None:
    agent = ValidatorAgent()
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    report = agent.validate_code(code, language="python")

    assert report.valid is True
    assert report.error_count == 0
    assert len(report.files_validated) == 1


def test_validator_python_syntax_error() -> None:
    agent = ValidatorAgent()
    code = "def broken(:\n    pass"
    report = agent.validate_code(code, language="python")

    assert report.valid is False
    assert report.error_count >= 1
    assert any(i.rule_id == "PY-SYNTAX-ERR" for i in report.issues)


def test_validator_detects_dangerous_calls() -> None:
    agent = ValidatorAgent()
    code = "import subprocess\ndef run_bad():\n    eval('1 + 1')\n    subprocess.run('ls', shell=True)\n"
    report = agent.validate_code(code, language="python")

    assert report.valid is False
    assert any(i.rule_id == "SEC-DANGEROUS-CALL" for i in report.issues)
    assert any(i.rule_id == "SEC-SUBPROCESS-SHELL" for i in report.issues)


def test_validator_detects_mutable_default() -> None:
    agent = ValidatorAgent()
    code = "def append_item(x, target=[]):\n    target.append(x)\n    return target\n"
    report = agent.validate_code(code, language="python")

    assert report.valid is True  # Warning does not invalidate by default
    assert report.warning_count >= 1
    assert any(i.rule_id == "PY-MUTABLE-DEFAULT" for i in report.issues)


def test_validator_detects_nan_equality() -> None:
    agent = ValidatorAgent()
    code = "def check(x):\n    if x == float('nan'):\n        return True\n    return False\n"
    report = agent.validate_code(code, language="python")

    assert report.valid is False
    assert any(i.rule_id == "PY-NAN-EQUALITY" for i in report.issues)


def test_validator_json_syntax() -> None:
    agent = ValidatorAgent()
    valid_json = '{"key": "value", "list": [1, 2, 3]}'
    invalid_json = '{"key": "value", "unclosed}'

    assert agent.validate_code(valid_json, language="json").valid is True
    invalid_report = agent.validate_code(invalid_json, language="json")
    assert invalid_report.valid is False
    assert any(i.rule_id == "JSON-SYNTAX-ERR" for i in invalid_report.issues)


def test_validator_changeset() -> None:
    agent = ValidatorAgent()
    change_set = ChangeSet(
        changes=[
            FileChange(
                file_path="src/main.py",
                operation=ChangeOperation.CREATE,
                new_content="def hello():\n    return 'world'\n",
            ),
            FileChange(
                file_path="config.json",
                operation=ChangeOperation.CREATE,
                new_content='{"app": "assistant"}',
            ),
        ],
        summary="Add main and config",
    )

    report = agent.validate_changeset(change_set)
    assert report.valid is True
    assert len(report.files_validated) == 2
