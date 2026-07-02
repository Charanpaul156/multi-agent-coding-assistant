"""API routes.

Scaffolding only.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["system"])
def health_check() -> dict:
    """Health endpoint."""

    return {"status": "ok"}

