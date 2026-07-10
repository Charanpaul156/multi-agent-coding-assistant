"""API routes.

This module defines HTTP endpoints.
Business logic must remain in the application layer.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import get_execute_code_use_case, get_generate_code_use_case
from backend.application.use_cases import GenerateCodeRequest, GenerateCodeUseCase
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

