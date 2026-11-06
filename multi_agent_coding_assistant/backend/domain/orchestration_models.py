from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkflowMode(str, Enum):
    """Supported orchestration workflow modes."""

    CODEBASE_CHAT = "codebase_chat"
    SECURITY_AUDIT = "security_audit"
    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"
    BUG_FIX = "bug_fix"
    FEATURE_DEVELOPMENT = "feature_development"


class AgentExecutionStatus(str, Enum):
    """Lifecycle state for a specialized agent execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentExecutionRecord:
    """Structured record of one agent step in an orchestrated workflow."""

    agent_name: str
    status: AgentExecutionStatus
    started_at: str
    completed_at: str
    duration_ms: float
    input_summary: str
    output_summary: str
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestrationResult:
    """Result of a completed orchestration run."""

    mode: WorkflowMode
    success: bool
    summary: str
    agent_records: List[AgentExecutionRecord] = field(default_factory=list)
    total_duration_ms: float = 0.0
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
