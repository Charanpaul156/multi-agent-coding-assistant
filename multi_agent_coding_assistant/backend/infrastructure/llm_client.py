"""Reusable LLM client abstraction for future agents.

This module must remain provider-agnostic and reusable.

It should expose methods that future agents (Planner/Reviewer/Tester/Debugger/
Documentation/RAG) can use without modifications.

Current implementation uses Google Gemini via the latest `google-genai` SDK,
but the public surface area intentionally stays generic.
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


class LLMClient:
    """Generic LLM client.

    Provider-agnostic public surface area:
    - `generate(prompt: str, system_prompt: Optional[str]) -> LLMResponse`

    Future agents should rely only on this public surface.


    Public methods intentionally accept plain text prompts and return plain
    text responses to keep downstream agents provider-agnostic.
    """

    def __init__(
        self,
        *,
        api_key_env: str = "GEMINI_API_KEY",
        model: str = "gemini-2.5-flash",
        backend_env_path: str | os.PathLike[str] = "backend/.env",
    ) -> None:
        self._api_key_env = api_key_env
        self._model = model
        self._backend_env_path = Path(backend_env_path)

        api_key = self._load_api_key()
        self._client = genai.Client(api_key=api_key)

    def _load_api_key(self) -> str:
        if self._backend_env_path.exists():
            load_dotenv(dotenv_path=str(self._backend_env_path), override=False)
        api_key = os.getenv(self._api_key_env)
        if not api_key:
            raise LLMConfigurationError(
                f"Missing required API key env var: {self._api_key_env}"
            )
        return api_key

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
    ) -> LLMResponse:
        """Generate text from a prompt.

        This is the single, generic text-generation method used by agents.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        system_prefix = "" if not system_prompt else f"{system_prompt}\n\n"
        full_prompt = f"{system_prefix}{prompt}"

        logger.info("LLM request started")
        try:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=full_prompt,
            )
        except Exception as exc:
            # Best-effort transient detection.
            msg = str(exc).lower()
            if any(
                key in msg
                for key in [
                    "timeout",
                    "temporar",
                    "connection",
                    "rate",
                    "503",
                    "500",
                    "502",
                    "504",
                ]
            ):
                logger.exception("LLM transient error")
                raise LLMTransientError(str(exc)) from exc

            logger.exception("LLM request failed")
            raise LLMClientError(str(exc)) from exc

        text = self._extract_text(resp).strip()
        if not text:
            raise LLMClientError("LLM returned empty response")

        # Remove obvious leading/trailing markdown fences in case upstream
        # prompt fails (agents may also do their own cleanup).
        text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()

        logger.info("LLM request finished")
        return LLMResponse(text=text)

