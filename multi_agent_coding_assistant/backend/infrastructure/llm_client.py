"""Reusable LLM client abstraction supporting multiple providers (Gemini, Groq).

This module must remain provider-agnostic and reusable.

It exposes methods that future agents (Planner/Reviewer/Tester/Debugger/
Documentation/RAG) can use without modifications.

Supported providers:
- Google Gemini (via `google-genai` SDK)
- Groq (via `groq` SDK)

Configuration is driven by environment variables:
- LLM_PROVIDER: "gemini" (default) or "groq"
- GEMINI_API_KEY / GEMINI_MODEL (default: "gemini-2.5-flash")
- GROQ_API_KEY / GROQ_MODEL (default: "openai/gpt-oss-120b")
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import google.genai as genai

try:
    import groq
except ImportError:  # pragma: no cover
    groq = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """A minimal, provider-agnostic response shape."""

    text: str


class LLMClientError(RuntimeError):
    """Base error for LLM client failures."""


class LLMConfigurationError(LLMClientError):
    """Raised when required configuration is missing or invalid."""


class LLMTransientError(LLMClientError):
    """Raised for transient/network/5xx-like failures eligible for retry."""


def _sanitize_error(msg: Any, secret: str | None = None) -> str:
    """Sanitize error messages to prevent leaking API keys or credentials."""
    text = str(msg)
    if secret and secret in text:
        text = text.replace(secret, "[REDACTED]")
    # Redact common key formats
    text = re.sub(r"gsk_[a-zA-Z0-9_-]{10,}", "[REDACTED]", text)
    text = re.sub(r"AIza[a-zA-Z0-9_-]{20,}", "[REDACTED]", text)
    text = re.sub(r"sk-[a-zA-Z0-9_-]{20,}", "[REDACTED]", text)
    # Redact Authorization header patterns
    text = re.sub(r"(Bearer\s+)[a-zA-Z0-9_.-]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    # Redact query parameters
    text = re.sub(r"(key=)[a-zA-Z0-9_.-]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    # Redact dictionary/json keys with api_key
    text = re.sub(r"('api_key':\s*')[^']+'", r"\1[REDACTED]'", text)
    text = re.sub(r'("api_key":\s*")[^"]+"', r'\1[REDACTED]"', text)
    return text


class LLMClient:
    """Generic, provider-configurable LLM client.

    Provider-agnostic public surface area:
    - `generate(prompt: str, *, system_prompt: Optional[str]) -> LLMResponse`

    Supports 'gemini' and 'groq' providers seamlessly.
    """

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        api_key_env: Optional[str] = None,
        model: Optional[str] = None,
        backend_env_path: str | os.PathLike[str] = "backend/.env",
        temperature: Optional[float] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._backend_env_path = (
            Path(backend_env_path) if backend_env_path is not None else None
        )
        if self._backend_env_path is not None and self._backend_env_path.exists():
            load_dotenv(dotenv_path=str(self._backend_env_path), override=False)

        raw_provider = (
            provider
            if provider is not None
            else os.getenv("LLM_PROVIDER", "gemini")
        )
        self._provider = (raw_provider or "gemini").strip().lower()

        if self._provider == "gemini":
            self._api_key_env = api_key_env or "GEMINI_API_KEY"
            self._model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        elif self._provider == "groq":
            self._api_key_env = api_key_env or "GROQ_API_KEY"
            self._model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        else:
            raise LLMConfigurationError(
                f"Unsupported LLM provider: {self._provider}. Supported providers: 'gemini', 'groq'"
            )

        self._temperature = temperature

        if client is not None:
            self._client = client
            self._api_key = "mock-key"
        else:
            self._api_key = self._load_api_key()
            if self._provider == "gemini":
                self._client = genai.Client(api_key=self._api_key)
            elif self._provider == "groq":
                if groq is None:
                    raise LLMConfigurationError(
                        "The 'groq' package is required when LLM_PROVIDER=groq. Please install groq."
                    )
                self._client = groq.Groq(api_key=self._api_key)

    @property
    def provider(self) -> str:
        """Return the active LLM provider name."""
        return self._provider

    @property
    def model(self) -> str:
        """Return the active model name."""
        return self._model

    @property
    def api_key_env(self) -> str:
        """Return the name of the API key environment variable in use."""
        return self._api_key_env

    def __repr__(self) -> str:
        return f"<LLMClient provider={self._provider!r} model={self._model!r}>"

    def _load_api_key(self) -> str:
        api_key = os.getenv(self._api_key_env)
        if not api_key or not api_key.strip():
            raise LLMConfigurationError(
                f"Missing required API key env var: {self._api_key_env}"
            )
        return api_key.strip()

    def _extract_text(self, response: Any) -> str:
        # google-genai returns a response object; try multiple shapes.
        if response is None:
            return ""

        # Common shape: response.text (string)
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text

        # Alternative: response.candidates[0].content.parts[0].text
        candidates = getattr(response, "candidates", None)
        if candidates and isinstance(candidates, list):
            try:
                return candidates[0].content.parts[0].text or ""
            except Exception:  # pragma: no cover
                pass

        # Last resort: stringify
        return str(response)

    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
    ) -> str:
        system_prefix = "" if not system_prompt else f"{system_prompt}\n\n"
        full_prompt = f"{system_prefix}{prompt}"

        logger.info("Gemini request started (model: %s)", self._model)
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "contents": full_prompt,
            }
            temp = temperature if temperature is not None else self._temperature
            if temp is not None:
                from google.genai import types

                kwargs["config"] = types.GenerateContentConfig(temperature=temp)
            resp = self._client.models.generate_content(**kwargs)
        except Exception as exc:
            msg = _sanitize_error(str(exc), self._api_key)
            lower_msg = msg.lower()
            if any(
                key in lower_msg
                for key in [
                    "timeout",
                    "temporar",
                    "connection",
                    "rate",
                    "429",
                    "503",
                    "500",
                    "502",
                    "504",
                ]
            ):
                logger.warning("Gemini transient error: %s", msg)
                raise LLMTransientError(msg) from None

            logger.error("Gemini request failed: %s", msg)
            raise LLMClientError(msg) from None

        return self._extract_text(resp)

    def _generate_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info("Groq request started (model: %s)", self._model)
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
            }
            temp = temperature if temperature is not None else self._temperature
            if temp is not None:
                kwargs["temperature"] = temp
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = _sanitize_error(str(exc), self._api_key)
            lower_msg = msg.lower()

            # Check for authentication / invalid API key
            if (
                (groq is not None and isinstance(exc, groq.AuthenticationError))
                or "invalid api key" in lower_msg
                or "invalid_api_key" in lower_msg
                or "401" in lower_msg
            ):
                logger.error("Groq authentication failed")
                raise LLMClientError(f"Authentication failed: {msg}") from None

            # Check for rate limit (429)
            if (
                (groq is not None and isinstance(exc, groq.RateLimitError))
                or "429" in lower_msg
                or "rate limit" in lower_msg
            ):
                logger.warning("Groq rate limit encountered (transient)")
                raise LLMTransientError(f"Rate limit exceeded: {msg}") from None

            # Check for timeout
            if (
                (groq is not None and isinstance(exc, groq.APITimeoutError))
                or "timeout" in lower_msg
                or "timed out" in lower_msg
            ):
                logger.warning("Groq timeout encountered (transient)")
                raise LLMTransientError(f"Request timeout: {msg}") from None

            # Check for provider unavailable / server errors
            if (
                (
                    groq is not None
                    and isinstance(exc, (groq.APIConnectionError, groq.InternalServerError))
                )
                or any(
                    code in lower_msg
                    for code in [
                        "500",
                        "502",
                        "503",
                        "504",
                        "connection",
                        "temporar",
                        "unavailable",
                    ]
                )
            ):
                logger.warning("Groq provider unavailable (transient): %s", msg)
                raise LLMTransientError(f"Provider unavailable: {msg}") from None

            logger.error("Groq request failed: %s", msg)
            raise LLMClientError(f"Groq request failed: {msg}") from None

        if not completion or not getattr(completion, "choices", None):
            raise LLMClientError("LLM returned empty choices")

        first_choice = completion.choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message else ""
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        return content

    @retry(
        retry=retry_if_exception_type(LLMTransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Generate text from a prompt using the configured LLM provider.

        This is the single, generic text-generation method used by agents.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        if self._provider == "gemini":
            text = self._generate_gemini(prompt, system_prompt, temperature)
        elif self._provider == "groq":
            text = self._generate_groq(prompt, system_prompt, temperature)
        else:
            raise LLMConfigurationError(f"Unknown provider: {self._provider}")

        text = (text or "").strip()
        if not text:
            raise LLMClientError("LLM returned empty response")

        # Remove obvious leading/trailing markdown fences in case upstream
        # prompt fails (agents may also do their own cleanup).
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()

        if not text:
            raise LLMClientError("LLM returned empty response")

        logger.info("LLM request finished")
        return LLMResponse(text=text)
