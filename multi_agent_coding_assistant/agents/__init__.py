"""Specialized agent implementations for the Multi-Agent AI Software Engineering Platform.

Note: PlannerAgent is strictly preserved and must not be modified.
"""

from agents.coder_agent import CoderAgent, CoderAgentError
from agents.code_reviewer_agent import (
    ReviewFinding,
    ReviewReport,
    ReviewerAgent,
    ReviewerAgentError,
)
from agents.debugger_agent import (
    DebugReport,
    DebuggerAgent,
    DebuggerAgentError,
)
from agents.debug_agent import DebugAgent
from agents.documentation_agent import (
    DocumentationAgent,
    DocumentationAgentError,
    DocumentationReport,
)
from agents.planner_agent import ImplementationPlan, PlannerAgent, PlannerAgentError
from agents.rag_agent import (
    RagAgent,
    RagAgentError,
    ResearchReport,
    SourceReference,
)
from agents.validator_agent import (
    StructuredValidationReport,
    ValidationIssue,
    ValidationSeverity,
    ValidatorAgent,
)

__all__ = [
    "CoderAgent",
    "CoderAgentError",
    "DebugAgent",
    "DebuggerAgent",
    "DebuggerAgentError",
    "DebugReport",
    "DocumentationAgent",
    "DocumentationAgentError",
    "DocumentationReport",
    "ImplementationPlan",
    "PlannerAgent",
    "PlannerAgentError",
    "RagAgent",
    "RagAgentError",
    "ResearchReport",
    "ReviewFinding",
    "ReviewReport",
    "ReviewerAgent",
    "ReviewerAgentError",
    "SourceReference",
    "StructuredValidationReport",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidatorAgent",
]
