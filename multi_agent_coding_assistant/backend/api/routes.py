"""API routes.

This module defines HTTP endpoints.
Business logic must remain in the application layer.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import (
    get_execute_code_use_case,
    get_generate_code_use_case,
    get_generate_plan_use_case,
)
from backend.application.use_cases import GenerateCodeRequest, GenerateCodeUseCase
from backend.application.planning_use_cases import GeneratePlanRequest, GeneratePlanUseCase

from backend.tools.python_executor import ExecutionRequest, ExecuteCodeUseCase

logger = logging.getLogger(__name__)

router = APIRouter()


class GenerateCodePayload(BaseModel):
    prompt: str = Field(..., min_length=1)


class GenerateCodeResponse(BaseModel):
    success: bool
    generated_code: str


class ExecuteCodePayload(BaseModel):
    generated_code: str


class ExecuteCodeApiResponse(BaseModel):
    success: bool
    stdout: str
    stderr: str
    execution_time_ms: float
    exit_code: int


@router.post(
    "/execute-code",
    response_model=ExecuteCodeApiResponse,
    tags=["ai"],
)
def execute_code(
    payload: ExecuteCodePayload,
    use_case: ExecuteCodeUseCase = Depends(get_execute_code_use_case),
) -> ExecuteCodeApiResponse:
    """Execute AI-generated Python code."""

    code = (payload.generated_code or "").strip()
    if not code:
        logger.warning("POST /execute-code: empty generated_code")
        raise HTTPException(status_code=400, detail="generated_code must not be empty")

    logger.info("POST /execute-code: request received")
    try:
        logger.info("POST /execute-code: use-case started")
        result = use_case.execute(ExecutionRequest(generated_code=code))
        logger.info("POST /execute-code: use-case finished")
        return ExecuteCodeApiResponse(
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr,
            execution_time_ms=result.execution_time_ms,
            exit_code=result.exit_code,
        )
    except ValueError as exc:
        logger.warning("POST /execute-code: validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except TimeoutError:
        logger.warning("POST /execute-code: execution timeout")
        raise HTTPException(status_code=408, detail="Execution timed out") from None
    except Exception as exc:
        logger.exception("POST /execute-code: error")
        raise HTTPException(status_code=500, detail="Unexpected execution failure") from exc


@router.get("/health", tags=["system"])
def health_check() -> dict:
    """Health endpoint."""

    return {"status": "ok"}


class GeneratePlanPayload(BaseModel):
    prompt: str = Field(..., min_length=1)



class ImplementationPlanModel(BaseModel):
    problem_summary: str
    project_type: str
    requirements: list[str]
    modules: list[str]
    functions: list[str]
    classes: list[str]
    external_libraries: list[str]
    database_needed: bool
    api_needed: list[str]
    algorithm: str
    edge_cases: list[str]
    estimated_complexity: str
    future_improvements: list[str]


class GeneratePlanApiResponse(BaseModel):
    success: bool
    plan: ImplementationPlanModel


@router.post(
    "/generate-plan",
    response_model=GeneratePlanApiResponse,
    tags=["ai"],
)

def generate_plan(
    payload: GeneratePlanPayload,
    use_case: GeneratePlanUseCase = Depends(get_generate_plan_use_case),
) -> GeneratePlanApiResponse:
    """Generate a structured implementation plan from a natural language prompt."""

    prompt = (payload.prompt or "").strip()
    if not prompt:
        logger.warning("POST /generate-plan: empty prompt")
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    logger.info("POST /generate-plan: request received")
    try:
        logger.info("POST /generate-plan: use-case started")
        result = use_case.execute(GeneratePlanRequest(prompt=prompt))
        logger.info("POST /generate-plan: use-case finished")

        plan = result.plan
        return GeneratePlanApiResponse(
            success=True,
            plan=ImplementationPlanModel(
                problem_summary=plan.problem_summary,
                project_type=plan.project_type,
                requirements=plan.requirements,
                modules=plan.modules,
                functions=plan.functions,
                classes=plan.classes,
                external_libraries=plan.external_libraries,
                database_needed=plan.database_needed,
                api_needed=plan.api_needed,
                algorithm=plan.algorithm,
                edge_cases=plan.edge_cases,
                estimated_complexity=plan.estimated_complexity,
                future_improvements=plan.future_improvements,
            ),
        )
    except ValueError as exc:
        logger.warning("POST /generate-plan: validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /generate-plan: error")
        raise HTTPException(status_code=500, detail="LLM planning failed") from exc


@router.post(
    "/generate-code",
    response_model=GenerateCodeResponse,
    tags=["ai"],
)

def generate_code(
    payload: GenerateCodePayload,
    use_case: GenerateCodeUseCase = Depends(get_generate_code_use_case),
) -> GenerateCodeResponse:

    """Generate Python code from a natural language prompt."""

    prompt = (payload.prompt or "").strip()
    if not prompt:
        logger.warning("POST /generate-code: empty prompt")
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    logger.info("POST /generate-code: request received")
    try:
        logger.info("POST /generate-code: use-case started")
        result = use_case.execute(GenerateCodeRequest(prompt=prompt))
        logger.info("POST /generate-code: use-case finished")
        return GenerateCodeResponse(success=True, generated_code=result.generated_code)
    except ValueError as exc:
        logger.warning("POST /generate-code: validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /generate-code: error")
        raise HTTPException(status_code=500, detail="LLM generation failed") from exc

