"""Planner Agent.

This agent is responsible for turning a user programming request into a
structured *implementation plan*.

It must never generate source code.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List


from backend.infrastructure.llm_client import LLMClient

logger = logging.getLogger(__name__)


class PlannerAgentError(RuntimeError):
    """Base error for planner failures."""


@dataclass(frozen=True)
class ImplementationPlan:
    problem_summary: str
    project_type: str
    requirements: List[str]
    modules: List[str]
    functions: List[str]
    classes: List[str]
    external_libraries: List[str]
    database_needed: bool
    api_needed: List[str]
    algorithm: str
    edge_cases: List[str]
    estimated_complexity: str
    future_improvements: List[str]


@dataclass(frozen=True)
class GeneratePlanRequest:
    prompt: str


@dataclass(frozen=True)
class GeneratePlanResult:
    plan: ImplementationPlan


class PlannerAgent:
    """Create an ImplementationPlan from a natural language prompt."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def create_plan(
        self,
        prompt: str,
        *,
        retrieved_context: str | None = None,
    ) -> ImplementationPlan:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")

        # Optionally prepend repository context to the user prompt. Agents
        # never retrieve context internally; they only receive it as input.
        if retrieved_context:
            prompt = f"{retrieved_context}\n\n---\n\nUser request:\n{prompt}"

        system_prompt = (
            "You are a Senior Software Architect.\n"
            "Your responsibility is to create implementation plans only.\n"
            "Never generate source code.\n"
            "Never include code blocks.\n"
            "Never include markdown.\n"
            "Return ONLY structured planning information as valid JSON.\n"
            "The JSON MUST match the following schema:\n"
            "{\n"
            "  \"problem_summary\": string,\n"
            "  \"project_type\": string,\n"
            "  \"requirements\": string[],\n"
            "  \"modules\": string[],\n"
            "  \"functions\": string[],\n"
            "  \"classes\": string[],\n"
            "  \"external_libraries\": string[],\n"
            "  \"database_needed\": boolean,\n"
            "  \"api_needed\": string[],\n"
            "  \"algorithm\": string,\n"
            "  \"edge_cases\": string[],\n"
            "  \"estimated_complexity\": string,\n"
            "  \"future_improvements\": string[]\n"
            "}\n"
        )

        logger.info("PlannerAgent: planning")
        raw_text = self._call_llm(prompt, system_prompt)

        try:
            data = self._parse_json(raw_text)
            return self._validate_and_build_plan(data)
        except PlannerAgentError:
            raise
        except Exception as exc:
            # malformed JSON or unexpected response shape
            logger.warning("PlannerAgent: failed first parse (%s); retrying once", exc)
            correction_prompt = (
                "The previous response was not valid JSON matching the required schema. "
                "Return ONLY corrected valid JSON matching the schema. "
                "Do not add explanations."
            )
            raw_text_retry = self._call_llm(correction_prompt, system_prompt)
            try:
                data_retry = self._parse_json(raw_text_retry)
                return self._validate_and_build_plan(data_retry)
            except Exception as exc2:
                logger.exception("PlannerAgent: retry failed")
                raise PlannerAgentError("Invalid planning response from LLM") from exc2

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        response = self._llm_client.generate(prompt, system_prompt=system_prompt)
        return (response.text or "").strip()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        cleaned = _strip_markdown_code_fences(text).strip()

        # Attempt direct JSON parse.
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Last resort: extract the first JSON object.
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed

        raise PlannerAgentError("Malformed JSON from LLM")

    def _validate_and_build_plan(self, data: Dict[str, Any]) -> ImplementationPlan:
        required_fields = [
            "problem_summary",
            "project_type",
            "requirements",
            "modules",
            "functions",
            "classes",
            "external_libraries",
            "database_needed",
            "api_needed",
            "algorithm",
            "edge_cases",
            "estimated_complexity",
            "future_improvements",
        ]
        for field in required_fields:
            if field not in data:
                raise PlannerAgentError(f"Missing field in plan JSON: {field}")

        def as_str(value: Any) -> str:
            if not isinstance(value, str):
                raise PlannerAgentError("Expected string fields in plan")
            return value

        def as_str_list(value: Any) -> List[str]:
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise PlannerAgentError("Expected list[string] fields in plan")
            return value

        if not isinstance(data["database_needed"], bool):
            raise PlannerAgentError("Expected boolean for database_needed")

        return ImplementationPlan(
            problem_summary=as_str(data["problem_summary"]),
            project_type=as_str(data["project_type"]),
            requirements=as_str_list(data["requirements"]),
            modules=as_str_list(data["modules"]),
            functions=as_str_list(data["functions"]),
            classes=as_str_list(data["classes"]),
            external_libraries=as_str_list(data["external_libraries"]),
            database_needed=data["database_needed"],
            api_needed=as_str_list(data["api_needed"]),
            algorithm=as_str(data["algorithm"]),
            edge_cases=as_str_list(data["edge_cases"]),
            estimated_complexity=as_str(data["estimated_complexity"]),
            future_improvements=as_str_list(data["future_improvements"]),
        )


def _strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()

    fence_match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", stripped, flags=re.I)
    if fence_match:
        return fence_match.group(1)

    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


