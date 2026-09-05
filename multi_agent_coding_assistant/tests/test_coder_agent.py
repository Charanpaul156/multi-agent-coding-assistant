"""Unit tests for the CoderAgent.

These tests mock the LLMClient so no Gemini API key or network access is
required.
"""

from __future__ import annotations

import json

import pytest

from agents.coder_agent import CoderAgent, CoderAgentError
from agents.planner_agent import ImplementationPlan
from backend.domain.change_models import ChangeOperation, ChangeSet, FileChange
from backend.infrastructure.llm_client import LLMResponse


def _valid_changes_json() -> dict:
    return {
        "summary": "Implement requested repository changes.",
        "changes": [
            {
                "file_path": "backend/new_feature.py",
                "operation": "create",
                "new_content": "value = 1\n",
                "description": "add new feature module",
            },
            {
                "file_path": "backend/existing.py",
                "operation": "modify",
                "new_content": "value = 2\n",
                "original_hash": "abc123",
                "description": "update existing logic",
            },
        ],
    }


def _create_only_json() -> dict:
    return {
        "summary": "Create a new file.",
        "changes": [
            {
                "file_path": "backend/new_file.py",
                "operation": "create",
                "new_content": "print('hello')\n",
                "description": "add file",
            }
        ],
    }


def _modify_only_json() -> dict:
    return {
        "summary": "Modify an existing file.",
        "changes": [
            {
                "file_path": "backend/existing.py",
                "operation": "modify",
                "new_content": "value = 2\n",
                "original_hash": "fff111",
                "description": "update value",
            }
        ],
    }


class _FakeLLMClient:
    """A configurable fake LLM client for deterministic tests."""

    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = 0
        self.prompts: list[str] = []
        self.system_prompts: list[str | None] = []

    def generate(self, prompt, *, system_prompt=None):
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise AssertionError("No more LLM responses configured")
        return LLMResponse(text=self.responses.pop(0))


def _build_agent(client) -> CoderAgent:
    # Duck-typed: agent only needs .generate(prompt, system_prompt=...).
    return CoderAgent(client)  # type: ignore[arg-type]


def _plan() -> ImplementationPlan:
    return ImplementationPlan(
        problem_summary="Update repository feature",
        project_type="python-service",
        requirements=["create a new module", "update an existing module"],
        modules=["backend/new_feature.py", "backend/existing.py"],
        functions=["render_feature", "update_value"],
        classes=[],
        external_libraries=[],
        database_needed=False,
        api_needed=[],
        algorithm="simple",
        edge_cases=[],
        estimated_complexity="low",
        future_improvements=[],
    )


def test_generate_changes_valid_changeset() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_changes_json())])
    agent = _build_agent(client)

    changes = agent.generate_changes(
        "Add repository-aware changes",
        implementation_plan=_plan(),
        retrieved_context="repo context",
    )

    assert isinstance(changes, ChangeSet)
    assert changes.summary == "Implement requested repository changes."
    assert len(changes.changes) == 2
    assert changes.changes[0].operation == ChangeOperation.CREATE
    assert changes.changes[1].operation == ChangeOperation.MODIFY
    assert changes.changes[1].original_hash == "abc123"
    assert changes.changes[0].description == "add new feature module"
    assert "repo context" in client.prompts[0]
    assert "problem_summary" in client.prompts[0]
    assert client.calls == 1


def test_generate_changes_create_only() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_create_only_json())])
    agent = _build_agent(client)

    changes = agent.generate_changes("Create a file")

    assert isinstance(changes, ChangeSet)
    assert len(changes.changes) == 1
    assert changes.changes[0].operation == ChangeOperation.CREATE
    assert changes.changes[0].original_hash is None
    assert client.calls == 1


def test_generate_changes_modify_only() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_modify_only_json())])
    agent = _build_agent(client)

    changes = agent.generate_changes("Modify a file")

    assert isinstance(changes, ChangeSet)
    assert len(changes.changes) == 1
    assert changes.changes[0].operation == ChangeOperation.MODIFY
    assert changes.changes[0].original_hash == "fff111"
    assert client.calls == 1


def test_generate_changes_malformed_json_retries_once() -> None:
    client = _FakeLLMClient(
        responses=["not json at all", json.dumps(_create_only_json())]
    )
    agent = _build_agent(client)

    changes = agent.generate_changes("Create a file")

    assert isinstance(changes, ChangeSet)
    assert client.calls == 2


def test_generate_changes_malformed_json_fails_after_retry() -> None:
    client = _FakeLLMClient(responses=["not json at all", "still not json"])
    agent = _build_agent(client)

    with pytest.raises(CoderAgentError):
        agent.generate_changes("Create a file")
    assert client.calls == 2


def test_generate_changes_invalid_schema_fails_without_retry() -> None:
    data = _create_only_json()
    del data["changes"][0]["description"]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    with pytest.raises(CoderAgentError):
        agent.generate_changes("Create a file")
    assert client.calls == 1


def test_generate_changes_rejects_delete_operation() -> None:
    data = _create_only_json()
    data["changes"][0]["operation"] = "delete"
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    with pytest.raises(CoderAgentError):
        agent.generate_changes("Create a file")
    assert client.calls == 1


def test_generate_changes_rejects_empty_prompt() -> None:
    agent = _build_agent(_FakeLLMClient())
    with pytest.raises(ValueError):
        agent.generate_changes("   ")
