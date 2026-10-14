"""T04 — Trajectory-to-table renderer.

Renders a trajectory as a compact markdown/CSV table of actions.
Directly 'presents info better': dense tabular view fits small context windows.

Input:  'trajectory' (raw JSON dict)
Output: 'text'       (compact markdown table)
"""

from __future__ import annotations

from typing import Any

from .base import BaseComputeTool
from .t03_toolcall_extractor import extract_tool_records


def render_trajectory_table(
    trajectory: dict,
    *,
    include_output: bool = False,
    max_output_chars: int = 120,
    format: str = "markdown",
) -> str:
    """
    Render trajectory as a compact table.

    Columns: Step | Role | Tool | State-Changing? | Output (optional)
    """
    records = extract_tool_records(trajectory)
    conv = trajectory.get("conversation") or trajectory.get("messages") or []

    # Also include non-tool messages (user query, assistant reasoning)
    rows: list[dict] = []
    step = 0
    for msg in conv:
        role = msg.get("role", "")
        content = (msg.get("content", "") or "")[:200]

        if role == "system":
            continue
        elif role == "user":
            rows.append({"step": step, "role": "user", "tool": "-", "state_change": "-", "summary": content[:120]})
            step += 1
        elif role == "assistant":
            # Find matching tool records
            matching = [r for r in records if r["step"] == step - 1 or r["step"] == step]
            if matching:
                for r in matching:
                    row = {
                        "step": step,
                        "role": "assistant",
                        "tool": r["tool_name"],
                        "state_change": "Y" if r["is_state_changing"] else "N",
                        "summary": r["output_text"][:max_output_chars] if include_output else "",
                    }
                    rows.append(row)
            elif content:
                rows.append({"step": step, "role": "assistant", "tool": "-", "state_change": "-", "summary": content[:120]})
            step += 1

    if format == "markdown":
        return _to_markdown(rows, include_output)
    else:
        return _to_csv(rows, include_output)


def _to_markdown(rows: list[dict], include_output: bool) -> str:
    if include_output:
        header = "| Step | Role | Tool | SC? | Output |"
        sep = "|------|------|------|-----|--------|"
        lines = [header, sep]
        for r in rows:
            lines.append(f"| {r['step']} | {r['role']} | {r['tool']} | {r['state_change']} | {r['summary']} |")
    else:
        header = "| Step | Role | Tool | SC? |"
        sep = "|------|------|------|-----|"
        lines = [header, sep]
        for r in rows:
            lines.append(f"| {r['step']} | {r['role']} | {r['tool']} | {r['state_change']} |")
    return "\n".join(lines)


def _to_csv(rows: list[dict], include_output: bool) -> str:
    if include_output:
        header = "step,role,tool,state_change,output"
        lines = [header]
        for r in rows:
            lines.append(f"{r['step']},{r['role']},{r['tool']},{r['state_change']},{r['summary']}")
    else:
        header = "step,role,tool,state_change"
        lines = [header]
        for r in rows:
            lines.append(f"{r['step']},{r['role']},{r['tool']},{r['state_change']}")
    return "\n".join(lines)


class TrajectoryTableRenderer(BaseComputeTool):
    """T04: Render trajectory as a compact table."""

    tool_id = "T04"
    tool_name = "trajectory_table_renderer"
    input_type = "trajectory"
    output_type = "text"

    def _process(self, input_data: Any, context: dict) -> tuple[Any, dict]:
        table = render_trajectory_table(
            input_data,
            include_output=self.params.get("include_output", False),
            max_output_chars=self.params.get("max_output_chars", 120),
            format=self.params.get("format", "markdown"),
        )
        n_rows = table.count("\n")
        return table, {"n_rows": n_rows, "n_chars": len(table)}
