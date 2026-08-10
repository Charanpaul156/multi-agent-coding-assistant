"""Application use-cases for planning.

Use-cases remain framework-independent.
Application layer must not depend on FastAPI or Pydantic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from agents.planner_agent import ImplementationPlan, PlannerAgent



logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratePlanRequest:

    """Request DTO for generating an implementation plan."""

    prompt: str
    retrieved_context: str | None = None


@dataclass(frozen=True)
class GeneratePlanResult:
    """Result DTO for a generated implementation plan."""

    plan: ImplementationPlan


class GeneratePlanUseCase:
    """Use-case: generate a structured implementation plan."""

    def __init__(self, planner_agent: PlannerAgent) -> None:
        self._planner_agent = planner_agent

    def execute(self, request: GeneratePlanRequest) -> GeneratePlanResult:
        if not isinstance(request.prompt, str):
            raise TypeError("prompt must be a string")
        if not request.prompt.strip():
            raise ValueError("prompt must be non-empty")

        logger.info("Planning started")
        try:
            plan = self._planner_agent.create_plan(
                request.prompt,
                retrieved_context=request.retrieved_context,
            )
            logger.info("Planning completed")
            return GeneratePlanResult(plan=plan)
        except Exception:
            logger.exception("Planning failed")
            raise

