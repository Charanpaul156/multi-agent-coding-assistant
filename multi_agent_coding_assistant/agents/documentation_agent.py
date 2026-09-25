"""Documentation Agent.

This agent is responsible for analyzing Python source code and optional
retrieved context and producing a comprehensive, structured documentation
report (``DocumentationReport``).

RESPONSIBILITIES:
- Analyze Python source code.
- Understand functions, classes, signatures, and expected behavior.
- Incorporate optional retrieved repository context.
- Produce structured documentation including module description, function docs,
  class docs, usage examples, and clean markdown documentation.

STRICT CONSTRAINTS (must never be violated):
- NEVER write files to disk.
- NEVER execute code.
- NEVER run shell commands.
- NEVER access the filesystem.
- NEVER access ChromaDB directly.
- It communicates ONLY through the dependency-injected ``LLMClient``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from backend.infrastructure.llm_client import LLMClient

logger = logging.getLogger(__name__)


class DocumentationAgentError(RuntimeError):
    """Base error for documentation agent failures."""


@dataclass(frozen=True)
class DocumentationReport:
    """Structured documentation report produced by the DocumentationAgent.

    Contains structured breakdowns of module purpose, function and class
    documentation, practical usage examples, and complete formatted markdown.
    """

    summary: str
    module_description: str
    function_docs: Union[List[Any], Dict[str, Any]]
    class_docs: Union[List[Any], Dict[str, Any]]
    usage_examples: List[Any]
    markdown_documentation: str


class DocumentationAgent:
    """Produce a structured DocumentationReport from Python code and optional context.

    The agent only analyzes code and produces documentation. It never executes
    code, runs shell commands, accesses the filesystem, or touches vector stores.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    @property
    def llm_client(self) -> LLMClient:
        return self._llm_client

    def generate_documentation(
        self,
        code: str,
        *,
        retrieved_context: Optional[str] = None,
    ) -> DocumentationReport:
        """Generate structured documentation for the provided Python code.

        Args:
            code: The Python source code to analyze and document (required).
            retrieved_context: Optional context retrieved from the repository
                (e.g., via RAG) to aid documentation.

        Returns:
            A structured DocumentationReport.

        Raises:
            TypeError: If ``code`` is not a string or ``retrieved_context`` is
                not a string when provided.
            ValueError: If ``code`` is empty or only whitespace.
            DocumentationAgentError: If the LLM call fails or returns malformed
                or invalid documentation.
        """
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        if not code.strip():
            raise ValueError("code must be non-empty and non-whitespace")
        if retrieved_context is not None and not isinstance(retrieved_context, str):
            raise TypeError("retrieved_context must be a string if provided")

        system_prompt = (
            "You are an expert Python technical documentation specialist and "
            "senior software engineer.\n"
            "Your task is to analyze the provided Python source code and any "
            "retrieved context, and generate comprehensive, high-quality documentation.\n"
            "STRICT CONSTRAINTS:\n"
            "- Do NOT execute code.\n"
            "- Do NOT run shell commands.\n"
            "- Do NOT write or modify files on disk.\n"
            "- Do NOT access the filesystem or databases.\n"
            "- Return ONLY valid JSON matching the exact schema below.\n"
            "- Do NOT include any markdown code fences (like ```json) or explanations "
            "outside the JSON object.\n\n"
            "The JSON response must strictly conform to the following schema:\n"
            "{\n"
            '  "summary": "A concise one or two sentence summary of the code.",\n'
            '  "module_description": "A detailed description of the module, its purpose, architecture, and design.",\n'
            '  "function_docs": [\n'
            '    {\n'
            '      "name": "function_name",\n'
            '      "signature": "def function_name(...)",\n'
            '      "description": "what the function does",\n'
            '      "parameters": ["parameter descriptions"],\n'
            '      "returns": "return value description"\n'
            '    }\n'
            '  ],\n'
            '  "class_docs": [\n'
            '    {\n'
            '      "name": "ClassName",\n'
            '      "description": "what the class represents",\n'
            '      "methods": ["method descriptions"]\n'
            '    }\n'
            '  ],\n'
            '  "usage_examples": [\n'
            '    "Practical usage example demonstrating how to use the code."\n'
            '  ],\n'
            '  "markdown_documentation": "# Complete Markdown Documentation\\n..."\n'
            "}"
        )

        user_prompt = self._build_user_prompt(code, retrieved_context)

        logger.info("DocumentationAgent: generating documentation")
        raw_text = self._call_llm(user_prompt, system_prompt)

        try:
            data = self._parse_json(raw_text)
            return self._validate_and_build_report(data)
        except DocumentationAgentError as exc:
            # Only malformed JSON triggers a single retry attempt.
            # Missing fields or validation errors fail fast.
            if "malformed" not in str(exc).lower():
                raise
            logger.warning(
                "DocumentationAgent: failed first parse (%s); retrying once", exc
            )
            correction_prompt = (
                "The previous response was not valid JSON matching the required "
                "schema. Return ONLY corrected valid JSON matching the schema. "
                "Do not add explanations."
            )
            try:
                raw_text_retry = self._call_llm(correction_prompt, system_prompt)
                data_retry = self._parse_json(raw_text_retry)
                return self._validate_and_build_report(data_retry)
            except Exception as exc2:
                logger.exception("DocumentationAgent: retry failed")
                raise DocumentationAgentError(
                    "Invalid documentation response from LLM"
                ) from exc2

    def run(
        self,
        code: str,
        *,
        retrieved_context: Optional[str] = None,
    ) -> DocumentationReport:
        """Alias for generate_documentation to support existing callers."""
        return self.generate_documentation(
            code, retrieved_context=retrieved_context
        )

    def _build_user_prompt(
        self, code: str, retrieved_context: Optional[str] = None
    ) -> str:
        prompt_parts: List[str] = []
        if retrieved_context and retrieved_context.strip():
            prompt_parts.append(
                "### Retrieved Context:\n"
                f"{retrieved_context.strip()}"
            )
        prompt_parts.append(
            "### Python Code to Document:\n"
            "```python\n"
            f"{code.strip()}\n"
            "```"
        )
        prompt_parts.append(
            "Analyze the code above and return comprehensive documentation as a single JSON object."
        )
        return "\n\n".join(prompt_parts)

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        try:
            response = self._llm_client.generate(
                prompt, system_prompt=system_prompt
            )
            return (response.text or "").strip()
        except DocumentationAgentError:
            raise
        except Exception as exc:
            raise DocumentationAgentError(
                f"LLM failure during documentation generation: {exc}"
            ) from exc

    def _parse_json(self, text: str) -> Dict[str, Any]:
        cleaned = _strip_markdown_code_fences(text).strip()

        # Attempt direct JSON parse
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first JSON object using regex
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        raise DocumentationAgentError("Malformed JSON from LLM")

    def _validate_and_build_report(
        self, data: Dict[str, Any]
    ) -> DocumentationReport:
        required_fields = [
            "summary",
            "module_description",
            "function_docs",
            "class_docs",
            "usage_examples",
            "markdown_documentation",
        ]
        for field_name in required_fields:
            if field_name not in data:
                raise DocumentationAgentError(
                    f"Missing required field in documentation JSON: {field_name}"
                )

        if not isinstance(data["summary"], str) or isinstance(
            data["summary"], bool
        ):
            raise DocumentationAgentError("Expected string for summary")

        if not isinstance(data["module_description"], str) or isinstance(
            data["module_description"], bool
        ):
            raise DocumentationAgentError(
                "Expected string for module_description"
            )

        if not isinstance(data["markdown_documentation"], str) or isinstance(
            data["markdown_documentation"], bool
        ):
            raise DocumentationAgentError(
                "Expected string for markdown_documentation"
            )

        if not isinstance(data["function_docs"], (list, dict)):
            raise DocumentationAgentError(
                "Expected list or dict for function_docs"
            )

        if not isinstance(data["class_docs"], (list, dict)):
            raise DocumentationAgentError("Expected list or dict for class_docs")

        if not isinstance(data["usage_examples"], list):
            raise DocumentationAgentError("Expected list for usage_examples")

        return DocumentationReport(
            summary=data["summary"],
            module_description=data["module_description"],
            function_docs=data["function_docs"],
            class_docs=data["class_docs"],
            usage_examples=data["usage_examples"],
            markdown_documentation=data["markdown_documentation"],
        )


def _strip_markdown_code_fences(text: str) -> str:
    """Remove leading/trailing triple-backtick fences if present."""
    stripped = text.strip()

    fence_match = re.match(
        r"^```(?:json|markdown)?\s*([\s\S]*?)\s*```$",
        stripped,
        flags=re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1).strip()

    stripped = re.sub(
        r"^```(?:json|markdown)?\s*", "", stripped, flags=re.IGNORECASE
    )
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()
