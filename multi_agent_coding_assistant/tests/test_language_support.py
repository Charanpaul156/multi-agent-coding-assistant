"""Tests for language extraction, validation, system prompt generation, and CoderAgent language enforcement."""

import pytest

from agents.coder_agent import CoderAgent, CoderAgentError
from agents.language_support import (
    build_coder_system_prompt,
    extract_programming_language,
    get_language_display_name,
    normalize_language,
    validate_code_language,
)
from backend.application.use_cases import GenerateCodeRequest, GenerateCodeUseCase
from backend.infrastructure.llm_client import LLMResponse


class _FakeLLMClient:
    """Configurable fake LLM client for testing CoderAgent generation and retries."""

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


# ============================================================================ #
# 1. Language Extraction & Normalization Tests
# ============================================================================ #

def test_extract_programming_language_explicit_phrases():
    assert extract_programming_language("Create a calculator in Java") == "java"
    assert extract_programming_language("Build a REST API using Python") == "python"
    assert extract_programming_language("Write a utility with TypeScript") == "typescript"
    assert extract_programming_language("Implement quicksort written in C++") == "c++"
    assert extract_programming_language("Code a server in Go") == "go"
    assert extract_programming_language("Write a program in Rust") == "rust"


def test_extract_programming_language_prefix_phrases():
    assert extract_programming_language("Java calculator with add and subtract") == "java"
    assert extract_programming_language("Write a Hello World program in Java") == "java"
    assert extract_programming_language("Create a Python function to calculate factorial") == "python"
    assert extract_programming_language("Write a C++ class for vector math") == "c++"


def test_extract_programming_language_react_component():
    assert extract_programming_language("Write a React component") == "javascript"
    assert extract_programming_language("Write a React component with TypeScript") == "typescript"


def test_extract_programming_language_undetermined_preserves_none():
    # When language cannot be determined, do not guess aggressively
    assert extract_programming_language("build calculator") is None
    assert extract_programming_language("create a simple algorithm") is None
    assert extract_programming_language("fix the bug in the data structure") is None
    assert extract_programming_language("") is None


def test_normalize_language():
    assert normalize_language("JAVA") == "java"
    assert normalize_language("py") == "python"
    assert normalize_language("cpp") == "c++"
    assert normalize_language("ts") == "typescript"
    assert normalize_language("Auto Detect") is None
    assert normalize_language("none") is None
    assert normalize_language(None) is None


# ============================================================================ #
# 2. Lightweight Language Validator Tests
# ============================================================================ #

def test_validate_code_language_java_accepts_valid_java():
    java_code = """
    public class Calculator {
        public static void main(String[] args) {
            System.out.println("Calculator started");
        }
    }
    """
    valid, reason = validate_code_language(java_code, "java")
    assert valid is True
    assert reason is None


def test_validate_code_language_java_rejects_python_indicators():
    # Failure example from user report: Python syntax when Java was requested
    python_in_java = """
    class Calculator:
        \"\"\"A simple calculator in python.\"\"\"
        def add(self, a, b):
            return a + b

    if __name__ == "__main__":
        op = input("Enter operation: ")
        print("Result:", op)
    """
    valid, reason = validate_code_language(python_in_java, "java")
    assert valid is False
    assert reason is not None
    assert "Python function syntax" in reason or "Python entry point" in reason or "Python input" in reason


def test_validate_code_language_python_accepts_valid_python():
    python_code = """
    def factorial(n: int) -> int:
        \"\"\"Calculate factorial.\"\"\"
        if n <= 1:
            return 1
        return n * factorial(n - 1)
    """
    valid, reason = validate_code_language(python_code, "python")
    assert valid is True
    assert reason is None


def test_validate_code_language_python_rejects_java_indicators():
    java_in_python = """
    public class Factorial {
        public static void main(String[] args) {
            System.out.println("Hello from Java");
        }
    }
    """
    valid, reason = validate_code_language(java_in_python, "python")
    assert valid is False
    assert reason is not None
    assert "Java main method" in reason or "Java print statement" in reason or "Java class" in reason


def test_validate_code_language_rejects_empty():
    valid, reason = validate_code_language("   ", "java")
    assert valid is False
    assert "empty" in reason.lower()


# ============================================================================ #
# 3. System Prompt & Conciseness Constraints Tests
# ============================================================================ #

def test_build_coder_system_prompt_enforces_language_and_conciseness():
    prompt_java = build_coder_system_prompt("java")
    assert "You MUST generate ONLY valid Java code." in prompt_java
    assert "NEVER substitute another programming language" in prompt_java
    assert "Do NOT output syntax, imports, or conventions belonging to any other language." in prompt_java
    assert "For simple tasks" in prompt_java
    assert "concise single-file implementation" in prompt_java
    assert "Do NOT add unnecessary abstractions" in prompt_java


# ============================================================================ #
# 4. CoderAgent Generation & Regeneration Tests
# ============================================================================ #

def test_coder_agent_java_request_success():
    valid_java = """```java
public class Calculator {
    public static void main(String[] args) {
        System.out.println(1 + 2);
    }
}
```"""
    client = _FakeLLMClient(responses=[valid_java])
    agent = CoderAgent(client)

    result = agent.generate_code("Create a calculator in Java")
    assert "public class Calculator" in result
    assert "System.out.println" in result
    assert client.calls == 1
    assert "Java" in client.system_prompts[0]


def test_coder_agent_java_request_regenerates_on_python_mismatch():
    # First response returns python code (failure case reported by user)
    python_code = """
def add(a, b):
    return a + b

if __name__ == "__main__":
    print(add(2, 3))
"""
    # Second response (after regeneration instruction) returns valid Java
    valid_java = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    public static void main(String[] args) {
        System.out.println(new Calculator().add(2, 3));
    }
}
"""
    client = _FakeLLMClient(responses=[python_code, valid_java])
    agent = CoderAgent(client)

    result = agent.generate_code("Create a calculator in Java")

    assert client.calls == 2
    assert "public class Calculator" in result
    # Check that retry prompt informed LLM of the language mismatch
    assert "[CRITICAL ERROR - LANGUAGE MISMATCH]" in client.prompts[1]
    assert "Java" in client.prompts[1]


def test_coder_agent_fails_after_max_regeneration_attempts():
    # LLM stubbornly returns Python code both times
    python_code_1 = "def add(a, b): return a + b\nif __name__ == '__main__': print(add(1, 2))"
    python_code_2 = "def subtract(a, b): return a - b\nif __name__ == '__main__': print(subtract(3, 1))"

    client = _FakeLLMClient(responses=[python_code_1, python_code_2])
    agent = CoderAgent(client)

    with pytest.raises(CoderAgentError) as exc_info:
        agent.generate_code("Create a calculator in Java", max_attempts=2)

    assert client.calls == 2
    assert "Failed to generate valid Java code after 2 attempts" in str(exc_info.value)
    assert "Python" in str(exc_info.value)


def test_coder_agent_python_request_regenerates_on_java_mismatch():
    java_code = "public class Hello { public static void main(String[] args) { System.out.println(\"hi\"); } }"
    python_code = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"

    client = _FakeLLMClient(responses=[java_code, python_code])
    agent = CoderAgent(client)

    result = agent.generate_code("Create a Python function to calculate factorial")

    assert client.calls == 2
    assert "def factorial" in result
    assert "[CRITICAL ERROR - LANGUAGE MISMATCH]" in client.prompts[1]


def test_coder_agent_backward_compatibility_unspecified_defaults_to_python():
    python_code = "def calculate(a, b):\n    return a + b\n"
    client = _FakeLLMClient(responses=[python_code])
    agent = CoderAgent(client)

    result = agent.generate_code("build calculator")
    assert client.calls == 1
    assert "def calculate" in result
    assert "Python" in client.system_prompts[0]


def test_coder_agent_explicit_language_parameter_overrides_prompt():
    valid_java = "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hi\"); } }"
    client = _FakeLLMClient(responses=[valid_java])
    agent = CoderAgent(client)

    # Prompt does not state language, but explicit language="java" is passed
    result = agent.generate_code("Write a Hello World program", language="java")
    assert client.calls == 1
    assert "public class HelloWorld" in result
    assert "Java" in client.system_prompts[0]


def test_generate_code_use_case_forwards_language():
    valid_java = "public class Calculator { public static void main(String[] args) { System.out.println(\"calc\"); } }"
    client = _FakeLLMClient(responses=[valid_java])
    agent = CoderAgent(client)
    use_case = GenerateCodeUseCase(agent)

    req = GenerateCodeRequest(prompt="Create a simple calculator", language="java")
    res = use_case.execute(req)

    assert "public class Calculator" in res.generated_code
    assert "Java" in client.system_prompts[0]


def test_api_generate_code_accepts_and_passes_language():
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.api.deps import get_generate_code_use_case
    from backend.application.use_cases import GenerateCodeResult

    received_requests = []

    class _MockGenUseCase:
        def execute(self, request):
            received_requests.append(request)
            return GenerateCodeResult(generated_code="public class Test {}")

    app.dependency_overrides[get_generate_code_use_case] = lambda: _MockGenUseCase()
    client = TestClient(app)

    try:
        resp = client.post("/generate-code", json={"prompt": "build calculator in java", "language": "java"})
        assert resp.status_code == 200
        assert len(received_requests) == 1
        assert received_requests[0].prompt == "build calculator in java"
        assert received_requests[0].language == "java"
    finally:
        app.dependency_overrides.clear()


def test_api_run_workflow_accepts_and_passes_language():
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.api.deps import get_run_workflow_use_case
    from backend.application.workflow_use_cases import WorkflowResult, WorkflowStatus

    received_requests = []

    class _MockWfUseCase:
        def execute(self, request):
            received_requests.append(request)
            return WorkflowResult(
                success=True,
                workflow_status=WorkflowStatus.COMPLETED,
                planning=None,
                generated_code="public class Test {}",
            )

    app.dependency_overrides[get_run_workflow_use_case] = lambda: _MockWfUseCase()
    client = TestClient(app)

    try:
        resp = client.post("/run-workflow", json={"prompt": "create a calculator in Java", "language": "java"})
        assert resp.status_code == 200
        assert len(received_requests) == 1
        assert received_requests[0].prompt == "create a calculator in Java"
        assert received_requests[0].language == "java"
    finally:
        app.dependency_overrides.clear()

