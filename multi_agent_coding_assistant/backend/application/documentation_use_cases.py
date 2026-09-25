"""Application use-cases for code documentation.

Use-cases are framework-independent. Application layer must not depend on
FastAPI or Pydantic. Only dataclasses are used here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from agents.documentation_agent import (
    DocumentationAgent,
    DocumentationReport,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerateDocumentationRequest:
    """Request DTO for generating documentation for Python source code.

    ``code`` is always required. ``retrieved_context`` is optional context
    retrieved from the repository or knowledge base.
    """

    code: str
    retrieved_context: Optional[str] = None


@dataclass(frozen=True)
class GenerateDocumentationResult:
    """Result DTO containing the structured DocumentationReport."""

    report: DocumentationReport


class GenerateDocumentationUseCase:
    """Use-case: produce a structured DocumentationReport for Python code.

    The DocumentationAgent analyzes the code and optional retrieved context to
    generate structured documentation. It never executes code, writes files, or
    accesses the filesystem.
    """

    def __init__(self, documentation_agent: DocumentationAgent) -> None:
        self._documentation_agent = documentation_agent
        self.documentation_agent = documentation_agent

    def execute(
        self, request: GenerateDocumentationRequest
    ) -> GenerateDocumentationResult:
        """Execute the documentation-generation use-case.

        Args:
            request: Validated GenerateDocumentationRequest DTO.

        Returns:
            A GenerateDocumentationResult containing the DocumentationReport.

        Raises:
            TypeError: If request is not GenerateDocumentationRequest, or if code
                or retrieved_context has an invalid type.
            ValueError: If code is empty or whitespace-only.
        """
        if not isinstance(request, GenerateDocumentationRequest):
            raise TypeError("request must be a GenerateDocumentationRequest")

        if not isinstance(request.code, str):
            raise TypeError("code must be a string")
        if not request.code.strip():
            raise ValueError("code must be non-empty and non-whitespace")
        if request.retrieved_context is not None and not isinstance(
            request.retrieved_context, str
        ):
            raise TypeError("retrieved_context must be a string if provided")

        logger.info("GenerateDocumentationUseCase: start")
        report = self._documentation_agent.generate_documentation(
            request.code,
            retrieved_context=request.retrieved_context,
        )
        logger.info("GenerateDocumentationUseCase: finished")
        return GenerateDocumentationResult(report=report)
