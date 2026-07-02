"""FastAPI dependency injection scaffolding.

This module centralizes dependencies for the API layer.

At this stage we only provide placeholders so future implementations can
cleanly plug into FastAPI's Depends() mechanism.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def placeholder_dependency() -> Any:
    """Return a placeholder dependency.

    Replace this when real orchestration/use-cases are implemented.
    """

    return None


def build_dependency_provider(factory: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap a factory to become a FastAPI dependency provider."""

    def _provider() -> Any:
        return factory()

    return _provider

