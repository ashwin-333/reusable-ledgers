"""T17: surface source-indexed windows containing candidate-chosen literal cues.

This is a pure-compute retrieval aid, not a verdict or constraint checker. It
never reads labels and never suppresses the complete trajectory shown to Gemma.
"""

from __future__ import annotations

import json
from typing import Any

from .base import BaseComputeTool


DEFAULT_TERMS = (
    "requested", "only", "never", "must not", "recipient", "target",
    "amount", "permission", "created", "updated", "deleted", "sent",
)


class SalienceWindows(BaseComputeTool):
    tool_id = "T17"
    tool_name = "salience_windows"
    input_type = "trajectory"
    output_type = "text"

    def _process(self, input_data: Any, context: dict) -> tuple[str, dict]:
        turns = input_data.get("conversation") or []
        terms = tuple(self.params.get("terms", DEFAULT_TERMS))
        window_turns = self.params.get("window_turns", 8)
        stride_turns = self.params.get("stride_turns", 4)
        top_k = self.params.get("top_k", 4)
        excerpt_chars = self.params.get("excerpt_chars", 240)
        windows = []
        for start in range(0, len(turns), stride_turns):
            selected = turns[start : start + window_turns]
            if not selected:
                break
            matches = []
            for offset, turn in enumerate(selected):
                for field in ("content", "tool_calls"):
                    value = turn.get(field)
                    if value is None:
                        continue
                    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    lower = text.lower()
                    found = [term for term in terms if term.lower() in lower]
                    if not found:
                        continue
                    first = min(lower.find(term.lower()) for term in found)
                    left = max(0, first - excerpt_chars // 3)
                    snippet = text[left : left + excerpt_chars].replace("\n", " ")
                    matches.append((start + offset, turn.get("role", "?"), field, found, snippet))
            score = len({(index, field) for index, _, field, _, _ in matches})
            if score:
                windows.append((score, start, matches))
        windows.sort(key=lambda row: (-row[0], row[1]))
        lines = ["### SOURCE-INDEXED SALIENT WINDOWS", ""]
        for score, start, matches in windows[:top_k]:
            lines.append(f"Window turns {start}–{min(start + window_turns - 1, len(turns) - 1)}; matched fields: {score}")
            for index, role, field, found, snippet in matches[:12]:
                lines.append(f"  turn {index} ({role}.{field}), cues {', '.join(found)}: {snippet}")
            lines.append("")
        if not windows:
            lines.append("No configured literal cues matched the trajectory.")
        return "\n".join(lines), {"windows_scored": len(windows), "windows_returned": min(top_k, len(windows))}
