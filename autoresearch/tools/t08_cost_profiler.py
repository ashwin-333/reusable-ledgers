"""T08 — Cost / latency profiler wrapper.

Wraps any other tool invocation to measure cost and latency, enabling
the 'efficiency vs large teacher' story and cheap validation before full eval.

This is a meta-tool: it wraps another tool's process() call.

Input:  whatever the wrapped tool expects
Output: whatever the wrapped tool produces (with cost metadata enriched)
"""

from __future__ import annotations

import time
from typing import Any

from .base import ToolResult, BaseComputeTool


class CostProfiler(BaseComputeTool):
    """T08: Wraps another tool to profile cost and latency."""

    tool_id = "T08"
    tool_name = "cost_latency_profiler"
    input_type = "any"
    output_type = "any"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._wrapped_tool = None
        self._cumulative_cost = 0.0
        self._cumulative_latency = 0.0
        self._call_count = 0

    def wrap(self, tool: Any) -> "CostProfiler":
        """Attach a tool to profile."""
        self._wrapped_tool = tool
        self.input_type = tool.input_type
        self.output_type = tool.output_type
        return self

    def process(self, input_data: Any, context: dict) -> ToolResult:
        if self._wrapped_tool is None:
            raise RuntimeError("CostProfiler has no wrapped tool. Call .wrap(tool) first.")

        t0 = time.monotonic()
        result = self._wrapped_tool.process(input_data, context)
        elapsed = time.monotonic() - t0

        self._cumulative_cost += result.cost_usd
        self._cumulative_latency += elapsed
        self._call_count += 1

        result.metadata["profiled_latency_s"] = elapsed
        result.metadata["profiled_cost_usd"] = result.cost_usd
        result.metadata["cumulative_cost_usd"] = self._cumulative_cost
        result.metadata["cumulative_latency_s"] = self._cumulative_latency
        result.metadata["call_count"] = self._call_count

        return result

    def estimate_cost(self, input_data: Any) -> float:
        if self._wrapped_tool:
            return self._wrapped_tool.estimate_cost(input_data)
        return 0.0

    def get_summary(self) -> dict:
        """Return profiling summary."""
        return {
            "total_cost_usd": self._cumulative_cost,
            "total_latency_s": self._cumulative_latency,
            "n_calls": self._call_count,
            "avg_latency_s": (
                self._cumulative_latency / self._call_count
                if self._call_count > 0
                else 0.0
            ),
            "avg_cost_usd": (
                self._cumulative_cost / self._call_count
                if self._call_count > 0
                else 0.0
            ),
        }
