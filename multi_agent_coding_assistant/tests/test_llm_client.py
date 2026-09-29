"""Tests for LLMClient provider configuration, Groq integration, and security.

All tests use mocks/fakes - NO real API calls are made here.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import groq

from backend.infrastructure.llm_client import (
    LLMClient,
    LLMClientError,
    LLMConfigurationError,
    LLMResponse,
    LLMStructuredOutputError,
    LLMTransientError,
    _sanitize_error,
)


def _make_mock_groq_completion(content: str) -> MagicMock:
    """Helper to create a mock Groq ChatCompletion object."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


class TestLLMProviderSelection:
    """Tests for LLM provider selection and model configuration."""

    def test_provider_selection_gemini_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
        monkeypatch.delenv("GEMINI_MODEL", raising=False)

        client = LLMClient(backend_env_path=None, client=MagicMock())
        assert client.provider == "gemini"
        assert client.model == "gemini-2.5-flash"
        assert client.api_key_env == "GEMINI_API_KEY"

    def test_provider_selection_gemini_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

        client = LLMClient(provider="gemini", backend_env_path=None, client=MagicMock())
        assert client.provider == "gemini"

    def test_provider_selection_groq_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
        monkeypatch.delenv("GROQ_MODEL", raising=False)

        client = LLMClient(backend_env_path=None, client=MagicMock())
        assert client.provider == "groq"
        assert client.model == "openai/gpt-oss-120b"
        assert client.api_key_env == "GROQ_API_KEY"

    def test_provider_selection_groq_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")

        client = LLMClient(provider="groq", backend_env_path=None, client=MagicMock())
        assert client.provider == "groq"
        assert client.model == "openai/gpt-oss-120b"

    def test_missing_groq_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        with pytest.raises(LLMConfigurationError) as exc_info:
            LLMClient(backend_env_path=None)
        assert "GROQ_API_KEY" in str(exc_info.value)

    def test_missing_gemini_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(LLMConfigurationError) as exc_info:
            LLMClient(backend_env_path=None)
        assert "GEMINI_API_KEY" in str(exc_info.value)

    def test_unsupported_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "unsupported-provider")

        with pytest.raises(LLMConfigurationError) as exc_info:
            LLMClient(backend_env_path=None, client=MagicMock())
        assert "unsupported-provider" in str(exc_info.value).lower()

    def test_correct_groq_model_selection_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")

        client = LLMClient(backend_env_path=None, client=MagicMock())
        assert client.model == "llama-3.3-70b-versatile"

    def test_correct_groq_model_selection_param(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")

        client = LLMClient(model="custom/test-model", backend_env_path=None, client=MagicMock())
        assert client.model == "custom/test-model"


class TestGroqExecutionAndNormalization:
    """Tests for Groq response normalization and execution."""

    def test_groq_response_normalization_clean_text(self) -> None:
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = _make_mock_groq_completion(
            "public class HelloWorld { public static void main(String[] args) {} }"
        )

        client = LLMClient(provider="groq", client=mock_groq)
        response = client.generate("Generate Java Hello World")

        assert isinstance(response, LLMResponse)
        assert "public class HelloWorld" in response.text
        assert mock_groq.chat.completions.create.called
        call_kwargs = mock_groq.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-oss-120b"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_groq_response_normalization_strips_markdown_fences(self) -> None:
        mock_groq = MagicMock()
        fenced_code = "```java\npublic class Main {}\n```"
        mock_groq.chat.completions.create.return_value = _make_mock_groq_completion(fenced_code)

        client = LLMClient(provider="groq", client=mock_groq)
        response = client.generate("Generate Java code")

        assert response.text == "public class Main {}"

    def test_groq_system_prompt_handling(self) -> None:
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = _make_mock_groq_completion("result")

        client = LLMClient(provider="groq", client=mock_groq)
        client.generate("User request", system_prompt="System instructions")

        call_kwargs = mock_groq.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System instructions"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User request"

    def test_groq_temperature_forwarding(self) -> None:
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = _make_mock_groq_completion("result")

        client = LLMClient(provider="groq", temperature=0.7, client=mock_groq)
        client.generate("User request")

        call_kwargs = mock_groq.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7

    def test_groq_empty_response_raises_client_error(self) -> None:
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = _make_mock_groq_completion("   ")

        client = LLMClient(provider="groq", client=mock_groq)
        with pytest.raises(LLMClientError) as exc_info:
            client.generate("Generate something")
        assert "empty response" in str(exc_info.value).lower()

    def test_groq_empty_choices_raises_client_error(self) -> None:
        mock_groq = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = []
        mock_groq.chat.completions.create.return_value = mock_completion

        client = LLMClient(provider="groq", client=mock_groq)
        with pytest.raises(LLMClientError) as exc_info:
            client.generate("Generate something")
        assert "empty choices" in str(exc_info.value).lower()


class TestGroqErrorNormalization:
    """Tests for mapping Groq provider errors to application LLM errors."""

    def test_groq_auth_error_normalized(self) -> None:
        mock_groq = MagicMock()
        response_mock = MagicMock(status_code=401)
        err = groq.AuthenticationError(
            message="Invalid API Key",
            response=response_mock,
            body={"error": {"message": "Invalid API Key"}},
        )
        mock_groq.chat.completions.create.side_effect = err

        client = LLMClient(provider="groq", client=mock_groq)
        with pytest.raises(LLMClientError) as exc_info:
            client.generate("test prompt")
        assert "authentication failed" in str(exc_info.value).lower()

    def test_groq_rate_limit_error_normalized_transient(self) -> None:
        mock_groq = MagicMock()
        response_mock = MagicMock(status_code=429)
        err = groq.RateLimitError(
            message="Rate limit exceeded",
            response=response_mock,
            body={"error": {"message": "Rate limit exceeded"}},
        )
        mock_groq.chat.completions.create.side_effect = err

        client = LLMClient(provider="groq", client=mock_groq)
        # Because tenacity retries LLMTransientError 3 times and then reraises
        with pytest.raises(LLMTransientError) as exc_info:
            client.generate("test prompt")
        assert "rate limit" in str(exc_info.value).lower()
        assert mock_groq.chat.completions.create.call_count == 3

    def test_groq_timeout_error_normalized_transient(self) -> None:
        mock_groq = MagicMock()
        err = groq.APITimeoutError(request=MagicMock())
        mock_groq.chat.completions.create.side_effect = err

        client = LLMClient(provider="groq", client=mock_groq)
        with pytest.raises(LLMTransientError) as exc_info:
            client.generate("test prompt")
        assert "timeout" in str(exc_info.value).lower()
        assert mock_groq.chat.completions.create.call_count == 3

    def test_groq_provider_unavailable_normalized_transient(self) -> None:
        mock_groq = MagicMock()
        response_mock = MagicMock(status_code=503)
        err = groq.InternalServerError(
            message="Service Unavailable",
            response=response_mock,
            body={"error": {"message": "Service Unavailable"}},
        )
        mock_groq.chat.completions.create.side_effect = err

        client = LLMClient(provider="groq", client=mock_groq)
        with pytest.raises(LLMTransientError) as exc_info:
            client.generate("test prompt")
        assert "provider unavailable" in str(exc_info.value).lower()
        assert mock_groq.chat.completions.create.call_count == 3

    def test_groq_json_validate_failed_normalized_to_structured_output_error(self) -> None:
        mock_groq = MagicMock()
        response_mock = MagicMock(status_code=400)
        # Even if failed_generation contains "401" or "invalid api key", it must NOT be classified as auth error
        err = groq.BadRequestError(
            message="Failed to generate JSON",
            response=response_mock,
            body={
                "error": {
                    "message": "Failed to generate JSON",
                    "type": "invalid_request_error",
                    "code": "json_validate_failed",
                    "failed_generation": "const status = 401; // invalid_api_key in code",
                }
            },
        )
        mock_groq.chat.completions.create.side_effect = err

        client = LLMClient(provider="groq", client=mock_groq)
        with pytest.raises(LLMStructuredOutputError) as exc_info:
            client.generate("Generate website", response_format="json")

        assert issubclass(LLMStructuredOutputError, LLMClientError)
        err_msg = str(exc_info.value).lower()
        assert "json_validate_failed" in err_msg
        assert "authentication" not in err_msg
        assert "failed_generation" not in str(exc_info.value)
        # Verify it wasn't treated as transient error (not retried by tenacity)
        assert mock_groq.chat.completions.create.call_count == 1

    def test_groq_generic_400_is_client_error_not_auth_error(self) -> None:
        mock_groq = MagicMock()
        response_mock = MagicMock(status_code=400)
        err = groq.BadRequestError(
            message="Invalid parameter value",
            response=response_mock,
            body={"error": {"message": "Invalid parameter value"}},
        )
        mock_groq.chat.completions.create.side_effect = err

        client = LLMClient(provider="groq", client=mock_groq)
        with pytest.raises(LLMClientError) as exc_info:
            client.generate("test prompt")

        assert not isinstance(exc_info.value, LLMStructuredOutputError)
        assert "authentication" not in str(exc_info.value).lower()
        assert "invalid parameter value" in str(exc_info.value).lower()


class TestGeminiExecutionAndNormalization:
    """Tests for Gemini response normalization and execution using mocks."""

    def test_gemini_response_normalization_text_attr(self) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "```python\nprint('hello')\n```"
        mock_client.models.generate_content.return_value = mock_resp

        client = LLMClient(provider="gemini", client=mock_client)
        res = client.generate("test prompt")
        assert res.text == "print('hello')"

    def test_gemini_response_candidates_shape(self) -> None:
        mock_client = MagicMock()
        mock_part = MagicMock(text="def foo(): pass")
        mock_content = MagicMock(parts=[mock_part])
        mock_candidate = MagicMock(content=mock_content)
        mock_resp = MagicMock(spec=["candidates"])
        mock_resp.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_resp

        client = LLMClient(provider="gemini", client=mock_client)
        res = client.generate("test prompt")
        assert res.text == "def foo(): pass"

    def test_gemini_transient_error_retry(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("503 Service Unavailable")

        client = LLMClient(provider="gemini", client=mock_client)
        with pytest.raises(LLMTransientError) as exc_info:
            client.generate("test prompt")
        assert "503" in str(exc_info.value)
        assert mock_client.models.generate_content.call_count == 3


class TestSecurityAndKeyLeakPrevention:
    """Verify that credentials and API keys are NEVER exposed in errors, logs, or repr."""

    def test_api_key_redacted_in_sanitization(self) -> None:
        secret = "gsk_1234567890abcdef1234567890abcdef"
        msg = f"Failed to connect with Authorization: Bearer {secret} and key={secret}"
        sanitized = _sanitize_error(msg, secret)

        assert secret not in sanitized
        assert "[REDACTED]" in sanitized

    def test_groq_error_does_not_leak_key_in_exception(self) -> None:
        secret = "gsk_my_super_secret_groq_api_key_123456"
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = RuntimeError(
            f"Error with key {secret} and header Bearer {secret}"
        )

        client = LLMClient(provider="groq", client=mock_groq)
        client._api_key = secret

        with pytest.raises(LLMClientError) as exc_info:
            client.generate("Hello")

        error_message = str(exc_info.value)
        assert secret not in error_message
        assert "[REDACTED]" in error_message

    def test_gemini_error_does_not_leak_key_in_exception(self) -> None:
        secret = "AIzaSyDummySecretGeminiKey1234567890123"
        mock_gemini = MagicMock()
        mock_gemini.models.generate_content.side_effect = RuntimeError(
            f"Gemini error with {secret}"
        )

        client = LLMClient(provider="gemini", client=mock_gemini)
        client._api_key = secret

        with pytest.raises(LLMClientError) as exc_info:
            client.generate("Hello")

        error_message = str(exc_info.value)
        assert secret not in error_message
        assert "[REDACTED]" in error_message

    def test_client_repr_does_not_leak_key(self) -> None:
        client = LLMClient(provider="groq", client=MagicMock())
        client._api_key = "gsk_super_confidential_key"

        representation = repr(client)
        assert "gsk_super_confidential_key" not in representation
        assert "groq" in representation

