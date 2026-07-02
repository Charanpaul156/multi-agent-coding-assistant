"""Tool registry placeholder.

Use this later to register CrewAI tools.
"""

from typing import Any


class ToolRegistry:
    """Register tools for agents."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, name: str, tool: Any) -> None:
        self._tools[name] = tool

    def get(self, name: str) -> Any:
        return self._tools[name]

