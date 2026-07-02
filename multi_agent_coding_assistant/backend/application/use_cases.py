"""Application use-cases (scaffolding).

Add future orchestration logic here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptRequest:
    """Incoming request DTO (placeholder)."""

    prompt: str


class BuildSoftwareUseCase:
    """Use-case placeholder."""

    def execute(self, request: PromptRequest) -> dict:
        """Execute the use-case.

        Business logic is intentionally omitted.
        """

        raise NotImplementedError("Use-case not implemented yet")

