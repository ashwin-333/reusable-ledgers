"""T03 — Tool-call extractor / normalizer.

Parses a trajectory JSON into a normalized list of (step, tool_name, args, output)
records. Removes formatting noise and provides a consistent view across frameworks.

Input:  'trajectory' (raw JSON dict)
Output: 'features'   (list of normalized tool-call records)
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import BaseComputeTool


# Patterns for extracting tool names from various formats
_FUNC_JSON = re.compile(r'"function"\s*:\s*"([^"]+)"')
_TOOL_CALL_DASH = re.compile(r"^-\s+(\S+)", re.MULTILINE)
# repr-style entries used by the `smoothed` dataset, e.g.
#   function='search_contacts_by_name' args={'query': 'x'} id='toolu_01...'
_FUNC_REPR = re.compile(r"function\s*=\s*['\"]([^'\"]+)['\"]")


def _extract_tool_names(tool_calls: Any) -> list[str]:
    """Extract tool names from a tool_calls field, trying multiple formats.

    Accepts a string (`synthetic`) or a list of repr-strings/dicts (`smoothed`).
    """
    if not tool_calls:
        return []

    # List form: recurse per element.
    if isinstance(tool_calls, list):
        names: list[str] = []
        for item in tool_calls:
            if isinstance(item, dict):
                fn = item.get("function", {})
                name = fn.get("name") if isinstance(fn, dict) else fn
                if name:
                    names.append(str(name))
            else:
                names.extend(_extract_tool_names(item))
        return names

    if not isinstance(tool_calls, str):
        return []

    tool_calls_str = tool_calls
    if not tool_calls_str.strip() or tool_calls_str.strip() == "Tool calls:":
        return []

    # Try JSON format first
    try:
        parsed = json.loads(tool_calls_str)
        if isinstance(parsed, list):
            names = []
            for call in parsed:
                if isinstance(call, dict):
                    fn = call.get("function", {})
                    name = fn.get("name") if isinstance(fn, dict) else fn
                    if name:
                        names.append(str(name))
            if names:
                return names
    except (json.JSONDecodeError, TypeError):
        pass

    # Try "function": "name" pattern
    names = _FUNC_JSON.findall(tool_calls_str)
    if names:
        return names

    # Try function='name' repr pattern
    names = _FUNC_REPR.findall(tool_calls_str)
    if names:
        return names

    # Try "- tool_name" list format (common in this repo's trajectory format)
    text = tool_calls_str
    if text.startswith("Tool calls:"):
        text = text[len("Tool calls:") :]
    names = _TOOL_CALL_DASH.findall(text.strip())
    if names:
        return names

    return []


def extract_tool_records(trajectory: dict) -> list[dict]:
    """
    Extract normalized tool-call records from a trajectory JSON.

    Returns a list of dicts:
        {
            "step": int,          # 0-based step index
            "role": str,          # "assistant" typically
            "tool_name": str,     # normalized tool name
            "args_text": str,     # raw args text (if available)
            "output_text": str,   # raw tool output (if available)
            "is_state_changing": bool,  # heuristic: not a read-only tool
        }
    """
    conv = trajectory.get("conversation") or trajectory.get("messages") or []

    records: list[dict] = []
    step = 0
    pending_tools: list[str] = []

    for msg in conv:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls_str = msg.get("tool_calls", "") or ""

        if role == "assistant":
            # Extract tool names from this assistant turn
            names = _extract_tool_names(tool_calls_str)
            if names:
                pending_tools = names
                for name in names:
                    records.append(
                        {
                            "step": step,
                            "role": role,
                            "tool_name": name,
                            "args_text": "",
                            "output_text": "",
                            "is_state_changing": _is_state_changing(name),
                        }
                    )
                step += 1

        elif role == "tool" and records:
            # Stitch tool output into the most recent record(s)
            # that don't have output yet
            for rec in reversed(records):
                if not rec["output_text"]:
                    rec["output_text"] = content
                    break

    return records


# Read-only tool patterns (heuristic)
_READ_ONLY_PREFIXES = (
    "read", "get", "list", "search", "find", "view", "show", "check",
    "describe", "fetch", "query", "lookup", "inspect", "browse",
)


def _is_state_changing(tool_name: str) -> bool:
    """Heuristic: is this tool likely to change state (vs. read-only)?"""
    name_lower = tool_name.lower().replace("_", "").replace("-", "")
    for prefix in _READ_ONLY_PREFIXES:
        if name_lower.startswith(prefix):
            return False
    return True


class ToolCallExtractor(BaseComputeTool):
    """T03: Extract and normalize tool calls from a trajectory."""

    tool_id = "T03"
    tool_name = "toolcall_extractor"
    input_type = "trajectory"
    output_type = "features"

    def _process(self, input_data: Any, context: dict) -> tuple[Any, dict]:
        records = extract_tool_records(input_data)
        tool_names = [r["tool_name"] for r in records]
        return {
            "records": records,
            "tool_sequence": tool_names,
            "n_tool_calls": len(records),
            "n_state_changing": sum(1 for r in records if r["is_state_changing"]),
            "n_read_only": sum(1 for r in records if not r["is_state_changing"]),
        }, {"n_records": len(records)}
