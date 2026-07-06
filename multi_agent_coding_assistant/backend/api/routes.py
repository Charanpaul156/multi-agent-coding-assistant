"""API routes.

This module defines HTTP endpoints.
Business logic must remain in the application layer.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import get_generate_code_use_case
from backend.application.use_cases import GenerateCodeRequest, GenerateCodeUseCase

logger = logging.getLogger(__name__)

router = APIRouter()


class GenerateCodePayload(BaseModel):
    prompt: str = Field(..., min_length=1)


class GenerateCodeResponse(BaseModel):
    success: bool
    generated_code: str


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

