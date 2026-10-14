"""T20 — State-change / env diff detector.

Parses tool outputs to detect before/after diffs (files, records, config).
Pinpoints irreversible/consequential actions for the model.

Input:  'trajectory' (raw JSON dict)
Output: 'features'   (list of detected state changes + summary)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .base import BaseComputeTool

# Patterns indicating state-changing operations in tool outputs
_STATE_CHANGE_INDICATORS = [
    # Write/modify operations
    (r"\b(?:created|wrote|written|saved|stored|updated|modified|changed|deleted|removed)\b", "write_op"),
    # Transfer/send operations
    (r"\b(?:transferred?|sent|delivered|forwarded|dispatched|posted|submitted)\b", "transfer_op"),
    # Status changes
    (r"\bstatus:\s*(?:success|completed|done|confirmed|approved)\b", "status_success"),
    (r"\bconfirmation[_-]?(?:id|number|code)\b", "confirmation"),
    # Database operations
    (r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)\s+", "db_mutation"),
    # File system mutations
    (r"\b(?:mv|cp|rm|mkdir|touch|truncate|append)\s+", "fs_mutation"),
]

# Patterns for extracting key entities from tool outputs
_ENTITY_PATTERNS = [
    (r"(?:recipient|to|target):\s*(\S+)", "recipient"),
    (r"(?:amount|value|sum):\s*\$?([\d,.]+)", "amount"),
    (r"(?:file|path):\s*(\S+)", "file_path"),
    (r"(?:email|address):\s*([\w.+-]+@[\w.-]+)", "email_target"),
]


def detect_state_changes(trajectory: dict) -> dict[str, Any]:
    """
    Detect state-changing operations in tool outputs.

    Returns:
        changes: list of detected state changes
        entities: extracted key entities (recipients, amounts, files)
        n_changes: total count
        n_consequential: count of transfer/write operations
    """
    conv = trajectory.get("conversation") or trajectory.get("messages") or []

    sc_compiled = [
        (re.compile(p, re.IGNORECASE), name)
        for p, name in _STATE_CHANGE_INDICATORS
    ]
    ent_compiled = [
        (re.compile(p, re.IGNORECASE), name) for p, name in _ENTITY_PATTERNS
    ]

    changes: list[dict] = []
    entities: list[dict] = []

    for step, msg in enumerate(conv):
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        # Focus on tool outputs (role=tool) and assistant actions
        if role not in ("tool", "assistant"):
            continue

        for regex, change_type in sc_compiled:
            for match in regex.finditer(content):
                changes.append(
                    {
                        "type": change_type,
                        "match": match.group(0)[:80],
                        "step": step,
                        "role": role,
                    }
                )

        for regex, entity_type in ent_compiled:
            for match in regex.finditer(content):
                entities.append(
                    {
                        "type": entity_type,
                        "value": match.group(1)[:80],
                        "step": step,
                    }
                )

    type_counts = Counter(c["type"] for c in changes)
    consequential_types = {"transfer_op", "write_op", "db_mutation", "fs_mutation"}
    n_consequential = sum(
        1 for c in changes if c["type"] in consequential_types
    )

    return {
        "changes": changes[:50],  # cap to avoid huge outputs
        "entities": entities[:30],
        "n_changes": len(changes),
        "n_consequential": n_consequential,
        "change_type_counts": dict(type_counts),
        "has_transfers": any(c["type"] == "transfer_op" for c in changes),
        "has_confirmations": any(c["type"] == "confirmation" for c in changes),
        "unique_recipients": list(
            {e["value"] for e in entities if e["type"] == "recipient"}
        ),
    }


class StateDiffDetector(BaseComputeTool):
    """T20: Detect state-changing operations in trajectory tool outputs."""

    tool_id = "T20"
    tool_name = "state_change_diff_detector"
    input_type = "trajectory"
    output_type = "features"

    def _process(self, input_data: Any, context: dict) -> tuple[Any, dict]:
        result = detect_state_changes(input_data)
        return result, {
            "n_changes": result["n_changes"],
            "n_consequential": result["n_consequential"],
        }
