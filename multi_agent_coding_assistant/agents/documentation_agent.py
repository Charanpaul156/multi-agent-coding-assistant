"""Documentation Agent.

This agent is responsible for analyzing Python source code and optional
retrieved context and producing comprehensive, structured documentation
reports (``DocumentationReport``), README files, API references, and architecture guides.

RESPONSIBILITIES:
- Analyze Python source code and repository structure.
- Understand functions, classes, signatures, and expected behavior.
- Generate structured module, function, and class documentation.
- Generate and update repository README files.
- Generate architectural and API documentation.
- Update documentation following approved code modifications.

STRICT CONSTRAINTS (must never be violated):
- NEVER write files to disk directly.
- NEVER execute code or run shell commands.
- NEVER access the filesystem or ChromaDB directly.
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
    """Structured documentation report produced by the DocumentationAgent."""

    summary: str
    module_description: str
    function_docs: Union[List[Any], Dict[str, Any]]
    class_docs: Union[List[Any], Dict[str, Any]]
    usage_examples: List[Any]
    markdown_documentation: str


class DocumentationAgent:
    """Produce structured documentation from code and context."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    @property
    def llm_client(self) -> LLMClient:
        return self._llm_client

    def run(
        self,
        code: str,
        *,
        retrieved_context: Optional[str] = None,
    ) -> DocumentationReport:
        """Execute documentation generation (backward-compatible alias)."""
        return self.generate_documentation(code, retrieved_context=retrieved_context)

    def generate_documentation(
        self,
        code: str,
        *,
        retrieved_context: Optional[str] = None,
    ) -> DocumentationReport:
        """Generate structured documentation for the provided Python code."""
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
            if "malformed" not in str(exc).lower():
                raise
            logger.warning(
                "DocumentationAgent: failed first parse (%s); retrying once", exc
            )
            correction_prompt = (
                "The previous response was not valid JSON matching the required "
                "schema. Return ONLY corrected valid JSON matching the schema. "
                "Do not add explanations or markdown code fences."
            )
            raw_text_retry = self._call_llm(correction_prompt, system_prompt)
            try:
                data_retry = self._parse_json(raw_text_retry)
                return self._validate_and_build_report(data_retry)
            except Exception as exc2:
                logger.exception("DocumentationAgent: retry failed")
                raise DocumentationAgentError(
                    "Invalid documentation response from LLM"
                ) from exc2

    def generate_readme(
        self,
        repository_name: str,
        description: str,
        files_summary: List[Dict[str, Any]],
        existing_readme: Optional[str] = None,
    ) -> str:
        """Generate a complete repository README in Markdown."""
        system_prompt = (
            "You are a Principal Software Engineer and Technical Writer.\n"
            "Create a clean, production-grade README.md for the repository.\n"
            "Include: Project Overview, Architecture, Key Modules, Setup & Installation, Usage, and Testing.\n"
            "Preserve existing documentation context where appropriate.\n"
            "Return ONLY the README markdown content without code fences surrounding the whole document."
        )
        prompt = (
            f"Repository: {repository_name}\n"
            f"Description: {description}\n"
            f"Indexed Modules: {json.dumps(files_summary, indent=2)}\n"
        )
        if existing_readme:
            prompt += f"\nExisting README:\n{existing_readme}\n"

        response = self._llm_client.generate(prompt, system_prompt=system_prompt)
        return _strip_markdown_code_fences(response.text or "")

    def generate_architecture_docs(
        self,
        structure: Dict[str, Any],
        tech_stack: List[str],
    ) -> str:
        """Generate architectural documentation for the codebase."""
        system_prompt = (
            "You are an Enterprise Software Architect.\n"
            "Generate an Architecture Specification in Markdown.\n"
            "Include: High-level System Design, Layer Separation (API, Application, Domain, Infrastructure), "
            "Data Flow, Integration Boundaries, and Design Tradeoffs."
        )
        prompt = f"Tech Stack: {', '.join(tech_stack)}\nStructure:\n{json.dumps(structure, indent=2)}"
        response = self._llm_client.generate(prompt, system_prompt=system_prompt)
        return _strip_markdown_code_fences(response.text or "")

    def update_documentation_for_changes(
        self,
        change_summary: str,
        current_documentation: str,
    ) -> str:
        """Update existing documentation to reflect newly approved changes."""
        system_prompt = (
            "You are an expert technical documentation maintainer.\n"
            "Update the existing documentation to accurately reflect the changes described in the summary.\n"
            "Preserve unrelated existing sections, formatting, and style.\n"
            "Return the updated markdown documentation."
        )
        prompt = (
            f"Change Summary:\n{change_summary}\n\n"
            f"Current Documentation:\n{current_documentation}"
        )
        response = self._llm_client.generate(prompt, system_prompt=system_prompt)
        return _strip_markdown_code_fences(response.text or "")

    def _build_user_prompt(
        self, code: str, retrieved_context: Optional[str]
    ) -> str:
        user_prompt = "Generate documentation for the following Python code:\n\n"
        user_prompt += "```python\n" + code + "\n```\n"

        if retrieved_context is not None and retrieved_context.strip():
            user_prompt += (
                "\n\nThe following repository context was retrieved to assist "
                "in understanding the code:\n\n"
                + retrieved_context.strip()
                + "\n"
            )

        return user_prompt

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

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

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
