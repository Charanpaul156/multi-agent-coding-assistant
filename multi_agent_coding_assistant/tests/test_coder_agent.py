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
from backend.infrastructure.llm_client import LLMResponse, LLMStructuredOutputError


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
        self.kwargs_list: list[dict] = []

    def generate(self, prompt, *, system_prompt=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        self.kwargs_list.append(kwargs)
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise AssertionError("No more LLM responses configured")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item)


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


def test_generate_changes_markdown_fenced_json() -> None:
    fenced = "```json\n" + json.dumps(_create_only_json()) + "\n```"
    client = _FakeLLMClient(responses=[fenced])
    agent = _build_agent(client)

    changes = agent.generate_changes("Create a file")
    assert isinstance(changes, ChangeSet)
    assert len(changes.changes) == 1
    assert changes.changes[0].file_path == "backend/new_file.py"


def test_generate_changes_unterminated_string_retries_and_succeeds() -> None:
    # Unterminated string in the first response (simulating cut-off or missing quote)
    unterminated = '{"summary": "Test", "changes": [{"file_path": "a.py", "operation": "create", "new_content": "unterminated code...'
    client = _FakeLLMClient(responses=[unterminated, json.dumps(_create_only_json())])
    agent = _build_agent(client)

    changes = agent.generate_changes("Build something")
    assert isinstance(changes, ChangeSet)
    assert len(changes.changes) == 1
    assert client.calls == 2
    # Verify the retry prompt included the correction guidance
    assert "CORRECTION REQUIRED" in client.prompts[1]
    assert "correctly escaped quotes" in client.prompts[1]


def test_generate_changes_truncated_json_fails_after_max_retries() -> None:
    # Truncated responses on both attempts
    truncated_1 = '{"summary": "Test", "changes": [{"file_path": "a.py", "operation": "create", "new_content": "val = 1'
    truncated_2 = '{"summary": "Test", "changes": [{"file_path": "a.py"'
    client = _FakeLLMClient(responses=[truncated_1, truncated_2])
    agent = _build_agent(client)

    with pytest.raises(CoderAgentError) as exc_info:
        agent.generate_changes("Build something")
    assert "could not parse" in str(exc_info.value).lower()
    assert client.calls == 2


def test_generate_changes_sanitizes_secret_in_error_message() -> None:
    secret = "gsk_super_confidential_key_12345"
    client = _FakeLLMClient(exc=RuntimeError(f"Network error with key {secret}"))
    agent = _build_agent(client)

    with pytest.raises(CoderAgentError) as exc_info:
        agent.generate_changes("Build something")
    err_text = str(exc_info.value)
    assert secret not in err_text
    assert "[REDACTED]" in err_text


def test_generate_changes_rejects_unsafe_paths() -> None:
    for bad_path in ["/etc/passwd", "../secret.py", "a/../../b.py"]:
        data = _create_only_json()
        data["changes"][0]["file_path"] = bad_path
        client = _FakeLLMClient(responses=[json.dumps(data)])
        agent = _build_agent(client)

        with pytest.raises(CoderAgentError) as exc_info:
            agent.generate_changes("Unsafe request")
        assert "unsafe" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
        assert client.calls == 1


def test_generate_changes_large_changeset_handling() -> None:
    # A website request with multiple files (HTML, CSS, JS, python backend, readme)
    files = [
        ("index.html", "create", "<!DOCTYPE html><html><body><h1>Hello</h1></body></html>"),
        ("styles.css", "create", "body { margin: 0; font-family: sans-serif; }"),
        ("app.js", "create", "document.addEventListener('DOMContentLoaded', () => console.log('ready'));"),
        ("backend/server.py", "create", "from fastapi import FastAPI\napp = FastAPI()\n"),
        ("README.md", "create", "# Website Project\nCreated via repository coder.\n"),
    ]
    data = {
        "summary": "Build website files.",
        "changes": [
            {
                "file_path": path,
                "operation": op,
                "new_content": content,
                "description": f"add {path}",
            }
            for path, op, content in files
        ],
    }
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    changes = agent.generate_changes("Build a website")
    assert isinstance(changes, ChangeSet)
    assert len(changes.changes) == 5
    assert [c.file_path for c in changes.changes] == [f[0] for f in files]
    assert client.calls == 1


def test_generate_changes_rejects_pathological_oversized_changeset() -> None:
    # 51 changes exceeds the safe limit of 50
    changes = [
        {
            "file_path": f"file_{i}.py",
            "operation": "create",
            "new_content": f"val_{i} = {i}\n",
            "description": f"file {i}",
        }
        for i in range(51)
    ]
    data = {"summary": "Too many files", "changes": changes}
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    with pytest.raises(CoderAgentError) as exc_info:
        agent.generate_changes("Make 51 files")
    assert "safe limit" in str(exc_info.value).lower()


def test_generate_changes_requests_structured_output() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_create_only_json())])
    agent = _build_agent(client)

    agent.generate_changes("Create a file")
    assert len(client.kwargs_list) == 1
    assert client.kwargs_list[0].get("response_format") == "json"
    assert client.kwargs_list[0].get("max_tokens") == 8192


def test_generate_changes_retries_on_structured_output_failure() -> None:
    # First call: structured output generation fails (e.g. Groq json_validate_failed)
    # Second call: valid JSON ChangeSet
    client = _FakeLLMClient(
        responses=[
            LLMStructuredOutputError("Provider failed structured output (code: json_validate_failed)"),
            json.dumps(_create_only_json()),
        ]
    )
    agent = _build_agent(client)

    changes = agent.generate_changes("Create a file", max_attempts=2)
    assert isinstance(changes, ChangeSet)
    assert len(changes.changes) == 1
    assert client.calls == 2
    assert "[CORRECTION REQUIRED - STRUCTURED OUTPUT GENERATION FAILED]" in client.prompts[1]
    assert "output budget" in client.prompts[1]


def test_generate_changes_fails_after_max_retries_on_structured_output_failure() -> None:
    client = _FakeLLMClient(
        responses=[
            LLMStructuredOutputError("Provider failed structured output (code: json_validate_failed)"),
            LLMStructuredOutputError("Provider failed structured output (code: json_validate_failed)"),
        ]
    )
    agent = _build_agent(client)

    with pytest.raises(CoderAgentError) as exc_info:
        agent.generate_changes("Build a huge website", max_attempts=2)

    assert client.calls == 2
    err_msg = str(exc_info.value).lower()
    assert "output budget" in err_msg or "smaller batches" in err_msg


def test_generate_changes_structured_output_error_sanitizes_secrets() -> None:
    secret = "gsk_1234567890abcdef1234567890abcdef"
    client = _FakeLLMClient(
        responses=[
            LLMStructuredOutputError(f"Provider failed with key {secret}"),
            LLMStructuredOutputError(f"Provider failed with key {secret}"),
        ]
    )
    agent = _build_agent(client)

    with pytest.raises(CoderAgentError) as exc_info:
        agent.generate_changes("Build something", max_attempts=2)

    assert secret not in str(exc_info.value)
