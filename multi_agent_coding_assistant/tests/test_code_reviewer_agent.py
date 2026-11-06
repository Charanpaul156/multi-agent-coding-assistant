"""Unit tests for the ReviewerAgent.

These tests mock the LLMClient so no Gemini API key or network access is
required.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from agents.code_reviewer_agent import (
    ReviewFinding,
    ReviewReport,
    ReviewerAgent,
    ReviewerAgentError,
)
from backend.infrastructure.llm_client import LLMResponse


def _valid_review_dict() -> Dict[str, Any]:
    return {
        "overall_score": 98,
        "strengths": ["Clean code", "Proper docstring"],
        "weaknesses": ["None"],
        "pep8_issues": [],
        "performance_suggestions": [],
        "security_concerns": [],
        "logic_issues": [],
        "maintainability": ["High"],
        "error_handling": ["Good"],
        "recommendations": ["Add type hints"],
        "final_summary": "Great implementation.",
    }


class _FakeLLMClient:
    """Configurable fake LLM client for deterministic testing."""

    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = 0
        self.last_prompt = None
        self.last_system_prompt = None

    def generate(self, prompt, *, system_prompt=None):
        self.calls += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise AssertionError("No more LLM responses configured")
        return LLMResponse(text=self.responses.pop(0))


def _build_agent(client) -> ReviewerAgent:
    return ReviewerAgent(client)  # type: ignore[arg-type]


def test_review_code_normalizes_none_in_logic_issues_to_empty() -> None:
    data = _valid_review_dict()
    data["logic_issues"] = [
        "None. No issues found.",
        "None. The function correctly implements multiplication.",
    ]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    report = agent.review_code("def multiply(a, b): return a * b")

    assert isinstance(report, ReviewReport)
    assert report.logic_issues == []


def test_review_code_normalizes_none_in_security_concerns_to_empty() -> None:
    data = _valid_review_dict()
    data["security_concerns"] = [
        "None. No security concerns.",
        "None. The function is a pure mathematical operation.",
    ]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    report = agent.review_code("def multiply(a, b): return a * b")

    assert isinstance(report, ReviewReport)
    assert report.security_concerns == []


def test_review_code_normalizes_no_issues_and_na() -> None:
    data = _valid_review_dict()
    data["logic_issues"] = ["No issues", "N/A", "no logic issues"]
    data["security_concerns"] = ["No security concerns", "not applicable", "None."]
    data["pep8_issues"] = ["No PEP8 issues", "None observed"]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    report = agent.review_code("def multiply(a, b): return a * b")

    assert report.logic_issues == []
    assert report.security_concerns == []
    assert report.pep8_issues == []


def test_review_code_preserves_genuine_issues() -> None:
    data = _valid_review_dict()
    data["logic_issues"] = [
        "Division by zero is not handled.",
        "Does not handle empty list input.",
        "No input validation for negative numbers.",
    ]
    data["security_concerns"] = [
        "User input is passed directly to subprocess.",
        "Hardcoded credential detected.",
    ]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    report = agent.review_code("def divide(a, b): return a / b")

    assert len(report.logic_issues) == 3
    assert "Division by zero is not handled." in report.logic_issues
    assert "Does not handle empty list input." in report.logic_issues
    assert "No input validation for negative numbers." in report.logic_issues
    assert len(report.security_concerns) == 2
    assert "User input is passed directly to subprocess." in report.security_concerns


def test_review_code_invalid_list_types_raise_error() -> None:
    data = _valid_review_dict()
    data["logic_issues"] = "not a list"
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    with pytest.raises(ReviewerAgentError, match="Expected list"):
        agent.review_code("def add(a, b): return a + b")

    data2 = _valid_review_dict()
    data2["security_concerns"] = [123, 456]
    client2 = _FakeLLMClient(responses=[json.dumps(data2)])
    agent2 = _build_agent(client2)

    with pytest.raises(ReviewerAgentError, match="Expected list"):
        agent2.review_code("def add(a, b): return a + b")


def test_reviewer_prompt_instructs_empty_array_for_no_issues() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_review_dict())])
    agent = _build_agent(client)

    agent.review_code("def multiply(a, b): return a * b")

    assert client.last_system_prompt is not None
    assert "empty array []" in client.last_system_prompt
    assert "None" in client.last_system_prompt
    assert "No issues" in client.last_system_prompt


def test_structured_finding_accepts_verified_evidence_and_canonical_severity() -> None:
    code = "def fetch(user_id):\n    query = f'SELECT * FROM users WHERE id={user_id}'\n    return db.execute(query)\n"
    data = _valid_review_dict()
    data["findings"] = [
        {
            "finding_id": "db-query",
            "title": "Unsafe SQL construction",
            "description": "User-controlled data is interpolated into SQL.",
            "category": "security",
            "severity": "high",
            "status": "confirmed",
            "file_path": "src/auth.py",
            "line_number": 3,
            "line_start": 3,
            "line_end": 3,
            "evidence": "db.execute(query)",
            "impact": "May allow query manipulation.",
            "recommended_fix": "Use a parameterized query.",
            "confidence": 0.94,
        }
    ]
    agent = _build_agent(_FakeLLMClient(responses=[json.dumps(data)]))

    report = agent.review_code(
        code,
        source_files={"src/auth.py": code},
    )

    finding = report.findings[0]
    assert finding.severity == "High"
    assert finding.status == "confirmed"
    assert finding.file_path == "src/auth.py"
    assert finding.line_number == finding.line_start == finding.line_end == 3
    assert finding.evidence == "db.execute(query)"
    assert finding.impact == "May allow query manipulation."
    assert finding.suggestion == "Use a parameterized query."
    assert finding.confidence == 0.94


def test_fabricated_paths_lines_and_evidence_are_removed() -> None:
    code = "def safe():\n    return True\n"
    data = _valid_review_dict()
    data["findings"] = [
        {
            "category": "security",
            "severity": "Critical",
            "message": "Possible credential exposure.",
            "status": "confirmed",
            "file_path": "secrets.py",
            "line_start": 500,
            "line_end": 500,
            "evidence": "API_KEY = 'not present'",
            "confidence": 1.0,
        }
    ]
    agent = _build_agent(_FakeLLMClient(responses=[json.dumps(data)]))

    finding = agent.review_code(code, source_files={"src/safe.py": code}).findings[0]

    assert finding.file_path is None
    assert finding.line_number is None
    assert finding.line_start is None
    assert finding.line_end is None
    assert finding.evidence is None
    assert finding.status == "potential"
    assert finding.confidence <= 0.79


@pytest.mark.parametrize("source_files", [None, {}])
def test_unmapped_file_path_is_preserved_as_potential(source_files) -> None:
    code = "def query(value):\n    return db.execute(value)\n"
    data = _valid_review_dict()
    data["findings"] = [
        {
            "category": "security",
            "severity": "High",
            "message": "The value reaches query execution.",
            "status": "confirmed",
            "file_path": "src/missing.py",
            "line_number": 2,
            "evidence": "db.execute(value)",
        }
    ]
    report = _build_agent(_FakeLLMClient(responses=[json.dumps(data)])).review_code(
        code,
        source_files=source_files,
    )

    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.status == "potential"
    assert finding.file_path is None
    assert finding.line_number is None
    assert finding.evidence == "db.execute(value)"


def test_context_only_finding_without_changed_files_remains_potential() -> None:
    code = "def changed():\n    return True\n"
    context = (
        "File: src/contracts.py (lines 20-21)\n\n"
        "```\nclass Contract:\n    pass\n```"
    )
    data = _valid_review_dict()
    data["findings"] = [
        {
            "category": "architecture",
            "severity": "Medium",
            "message": "The implementation may conflict with the contract.",
            "status": "confirmed",
            "file_path": "src/contracts.py",
            "line_start": 20,
            "line_end": 21,
            "evidence": "class Contract:",
        }
    ]

    finding = _build_agent(_FakeLLMClient(responses=[json.dumps(data)])).review_code(
        code,
        repository_context=context,
    ).findings[0]

    assert finding.file_path == "src/contracts.py"
    assert finding.line_start == 20
    assert finding.line_end == 21
    assert finding.evidence == "class Contract:"
    assert finding.status == "potential"


def test_out_of_range_lines_are_removed_for_known_source_path() -> None:
    code = "def safe():\n    return True\n"
    data = _valid_review_dict()
    data["findings"] = [
        {
            "category": "logic",
            "severity": "Medium",
            "message": "Possible missing condition.",
            "file_path": "src/safe.py",
            "line_number": 20,
            "line_start": 20,
            "line_end": 20,
            "evidence": "return True",
        }
    ]

    finding = _build_agent(_FakeLLMClient(responses=[json.dumps(data)])).review_code(
        code,
        source_files={"src/safe.py": code},
    ).findings[0]

    assert finding.file_path == "src/safe.py"
    assert finding.line_number is None
    assert finding.line_start is None
    assert finding.line_end is None
    assert finding.status == "potential"
    assert finding.confidence <= 0.79


def test_unverified_security_claim_is_potential_and_duplicate_is_removed() -> None:
    data = _valid_review_dict()
    finding = {
        "category": "security",
        "severity": "INFO",
        "message": "Possible injection risk.",
        "status": "confirmed",
        "confidence": 0.99,
    }
    data["findings"] = [finding, dict(finding)]
    agent = _build_agent(_FakeLLMClient(responses=[json.dumps(data)]))

    report = agent.review_code("def execute(value):\n    return value\n")

    assert len(report.findings) == 1
    assert report.findings[0].severity == "Informational"
    assert report.findings[0].status == "potential"
    assert report.findings[0].confidence <= 0.79


def test_paraphrased_findings_with_same_evidence_are_deduplicated() -> None:
    code = "def execute(value):\n    return value\n"
    data = _valid_review_dict()
    data["findings"] = [
        {
            "category": "logic",
            "severity": "Medium",
            "message": "The function returns the supplied value.",
            "line_number": 2,
            "evidence": "return value",
        },
        {
            "category": "security",
            "severity": "Low",
            "message": "The input passes directly to the return value.",
            "line_number": 2,
            "evidence": "return value",
        },
    ]
    report = _build_agent(_FakeLLMClient(responses=[json.dumps(data)])).review_code(code)

    assert len(report.findings) == 1


def test_invalid_severity_is_not_reported() -> None:
    data = _valid_review_dict()
    data["findings"] = [
        {"category": "security", "severity": "Blocker", "message": "Unsupported level."}
    ]
    report = _build_agent(_FakeLLMClient(responses=[json.dumps(data)])).review_code(
        "def safe():\n    return True"
    )

    assert report.findings == []


def test_legacy_finding_and_report_construction_remains_compatible() -> None:
    finding = ReviewFinding(
        "logic", "MEDIUM", "Existing finding", "src/legacy.py", 4, "Existing fix"
    )
    report = ReviewReport(
        80, [], [], [], [], [], [], [], [], [], "summary", [finding], ["architecture"]
    )

    assert finding.file_path == "src/legacy.py"
    assert finding.line_number == 4
    assert finding.suggestion == "Existing fix"
    assert report.architectural_violations == ["architecture"]
    assert report.code_duplications == []


def test_legacy_category_output_synthesizes_findings_when_findings_key_missing() -> None:
    data = _valid_review_dict()
    data.pop("findings", None)
    data["security_concerns"] = ["Input reaches a shell command."]
    data["logic_issues"] = ["Missing empty-input condition."]

    report = _build_agent(_FakeLLMClient(responses=[json.dumps(data)])).review_code(
        "def run(value):\n    return value\n"
    )

    assert [finding.category for finding in report.findings] == ["security", "logic"]
    assert all(finding.status == "potential" for finding in report.findings)
    assert all(finding.evidence is None for finding in report.findings)


def test_reviewer_prompt_contains_evidence_rules_and_supplied_context() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_review_dict())])
    agent = _build_agent(client)
    code = "def read():\n    return 1\n"

    agent.review_code(
        code,
        source_files={"src/read.py": code},
        repository_context="File: src/contracts.py\nLines: 1-2\nContent: def read(): ...",
    )

    assert "Do not invent evidence" in client.last_system_prompt
    assert "Critical, High, Medium, Low, Informational" in client.last_system_prompt
    assert "src/read.py" in client.last_prompt
    assert "src/contracts.py" in client.last_prompt


def test_context_source_reference_must_match_retrieved_path_and_line_range() -> None:
    code = "def changed():\n    return True\n"
    context = (
        "The following are relevant sections from the repository:\n\n"
        "File: src/contracts.py (lines 20-21)\n\n"
        "```\nclass Contract:\n    pass\n```"
    )
    data = _valid_review_dict()
    data["findings"] = [
        {
            "category": "architecture",
            "severity": "Medium",
            "message": "Changed implementation conflicts with the contract.",
            "status": "confirmed",
            "file_path": "src/contracts.py",
            "line_start": 20,
            "line_end": 21,
            "evidence": "class Contract:",
            "confidence": 0.88,
        }
    ]
    agent = _build_agent(_FakeLLMClient(responses=[json.dumps(data)]))

    finding = agent.review_code(
        code,
        source_files={"src/changed.py": code},
        repository_context=context,
    ).findings[0]

    assert finding.file_path == "src/contracts.py"
    assert finding.line_start == 20
    assert finding.line_end == 21
    assert finding.evidence == "class Contract:"
    assert finding.status == "potential"


def test_json_retry_retains_supplied_source_and_repository_context() -> None:
    client = _FakeLLMClient(responses=["not json", json.dumps(_valid_review_dict())])
    agent = _build_agent(client)
    code = "def read():\n    return 1\n"

    agent.review_code(
        code,
        source_files={"src/read.py": code},
        repository_context="File: src/contracts.py (lines 1-1)\n\n```\nclass Contract\n```",
    )

    assert client.calls == 2
    assert "src/read.py" in client.last_prompt
    assert "src/contracts.py" in client.last_prompt
