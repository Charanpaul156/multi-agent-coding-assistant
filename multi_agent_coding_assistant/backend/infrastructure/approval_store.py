"""In-memory ticket store for proposed ChangeSets pending human approval.

Ties approval state to a specific proposed ChangeSet, ensuring that only
explicitly approved, non-stale changes can be applied.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.domain.change_models import (
    ApprovalStatus,
    ChangeSet,
    DiffEntry,
    ValidationReport,
)


@dataclass
class ProposalTicket:
    """A proposed change ticket waiting for human review and approval."""

    token: str
    repository_path: str
    change_set: ChangeSet
    validation: ValidationReport
    diffs: list[DiffEntry]
    created_at: float = field(default_factory=time.time)
    status: ApprovalStatus = ApprovalStatus.PREVIEW


class ApprovalStore:
    """In-memory proposal ticket store with TTL."""

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._tickets: dict[str, ProposalTicket] = {}
        self._ttl_seconds = ttl_seconds

    def create_ticket(
        self,
        *,
        repository_path: str,
        change_set: ChangeSet,
        validation: ValidationReport,
        diffs: list[DiffEntry],
    ) -> ProposalTicket:
        """Create and store a proposal ticket with a cryptographically secure token."""
        token = secrets.token_urlsafe(32)
        ticket = ProposalTicket(
            token=token,
            repository_path=repository_path,
            change_set=change_set,
            validation=validation,
            diffs=diffs,
            created_at=time.time(),
            status=ApprovalStatus.PREVIEW,
        )
        self._tickets[token] = ticket
        return ticket

    def get_ticket(self, token: str) -> Optional[ProposalTicket]:
        """Retrieve a ticket by token, returning None if expired or not found."""
        ticket = self._tickets.get(token)
        if ticket is None:
            return None
        if time.time() - ticket.created_at > self._ttl_seconds:
            self._tickets.pop(token, None)
            return None
        return ticket

    def update_status(
        self, token: str, status: ApprovalStatus
    ) -> Optional[ProposalTicket]:
        """Update the approval status of a ticket."""
        ticket = self.get_ticket(token)
        if ticket is None:
            return None
        ticket.status = status
        return ticket

    def clear(self) -> None:
        """Clear all stored tickets (useful for tests)."""
        self._tickets.clear()
