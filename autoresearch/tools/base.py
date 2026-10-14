"""Base protocol and data types for composable trajectory analysis tools."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolResult:
    """Standardized output from any trajectory analysis tool."""

    output: Any
    """Primary output: dict (features), str (text), float (score), list (flags)."""

    output_type: str
    """One of: 'features', 'text', 'score', 'flags'."""

    cost_usd: float = 0.0
    """API cost incurred (0.0 for pure-compute tools)."""

    latency_s: float = 0.0
    """Wall-clock seconds."""

    metadata: dict = field(default_factory=dict)
    """Tool-specific metadata (model name, token counts, etc.)."""


@runtime_checkable
class TrajectoryTool(Protocol):
    """Interface implemented by registered executable tools for composition.

    The T01-T30 spreadsheet is a design catalog; several rows do not yet have a
    runtime implementation and therefore do not satisfy this protocol today.
    """

    tool_id: str
    """e.g. 'T05'."""

    tool_name: str
    """Human-readable name, e.g. 'toolcall_count_stats'."""

    input_type: str
    """What this tool consumes: 'trajectory' | 'features' | 'text' | 'turns'."""

    output_type: str
    """What this tool produces: 'features' | 'text' | 'score' | 'flags'."""

    def process(self, input_data: Any, context: dict) -> ToolResult:
        """
        Process input and return a ToolResult.

        Args:
            input_data: Depends on input_type.
                - 'trajectory': raw JSON dict (the loaded .json file)
                - 'turns': list of SimpleNamespace turn objects from load_trace_auto
                - 'features': dict of extracted features from an upstream tool
                - 'text': rendered string (e.g. compact table)
            context: Shared pipeline context carrying:
                - 'manifest': manifest entry metadata (construct, bucket, is_benign)
                - 'outputs': {tool_id: ToolResult} from upstream stages
        """
        ...

    def estimate_cost(self, input_data: Any) -> float:
        """Return estimated cost in USD. 0.0 for pure-compute tools."""
        ...


class BaseComputeTool:
    """
    Convenience base class for zero-cost, pure-compute tools.

    Subclasses only need to implement _process(input_data, context) -> (output, metadata).
    """

    tool_id: str = ""
    tool_name: str = ""
    input_type: str = "trajectory"
    output_type: str = "features"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def process(self, input_data: Any, context: dict) -> ToolResult:
        t0 = time.monotonic()
        output, metadata = self._process(input_data, context)
        elapsed = time.monotonic() - t0
        return ToolResult(
            output=output,
            output_type=self.output_type,
            cost_usd=0.0,
            latency_s=elapsed,
            metadata=metadata,
        )

    def _process(self, input_data: Any, context: dict) -> tuple[Any, dict]:
        raise NotImplementedError

    def estimate_cost(self, input_data: Any) -> float:
        return 0.0
