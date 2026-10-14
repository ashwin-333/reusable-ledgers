"""T31: configurable, source-indexed turn/field selection for Gemma evidence."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseComputeTool


class TurnFieldSelector(BaseComputeTool):
    tool_id = "T31"
    tool_name = "turn_field_selector"
    input_type = "trajectory"
    output_type = "text"

    def _process(self, input_data: Any, context: dict) -> tuple[str, dict]:
        turns = input_data.get("conversation") or []
        roles = set(self.params.get("roles", ["user", "assistant", "tool"]))
        fields = self.params.get("fields", ["content", "tool_calls"])
        terms = [term.lower() for term in self.params.get("contains_any", [])]
        max_turns = self.params.get("max_turns", 30)
        max_chars = self.params.get("max_chars_per_field", 300)
        matching = []
        for index, turn in enumerate(turns):
            if turn.get("role") not in roles:
                continue
            values = []
            for field in fields:
                value = turn.get(field)
                if value is None:
                    continue
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                if text.strip():
                    values.append((field, text))
            if not values:
                continue
            if terms and not any(term in text.lower() for _, text in values for term in terms):
                continue
            matching.append((index, turn.get("role", "?"), values))
        lines = [
            "### SOURCE-INDEXED TURN/FIELD SELECTION",
            f"Showing {min(len(matching), max_turns)} of {len(matching)} matching turns from {len(turns)} total turns.",
            "The complete conversation remains available separately; snippets below are not exhaustive.",
            "",
        ]
        for index, role, values in matching[:max_turns]:
            lines.append(f"turn {index} ({role})")
            for field, value in values:
                snippet = value[:max_chars].replace("\n", " ")
                suffix = " ...[excerpt truncated]" if len(value) > max_chars else ""
                lines.append(f"  {field}: {snippet}{suffix}")
        return "\n".join(lines), {
            "matching_turns": len(matching),
            "returned_turns": min(len(matching), max_turns),
        }
