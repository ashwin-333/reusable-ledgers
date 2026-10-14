"""T05 — Tool-call count & ratio statistics.

Cheap numeric features targeting C3 (structuring): per-tool counts, total steps,
unique tools, read:write ratio, and tool distribution entropy.

Input:  'features' (from T03: must contain 'tool_sequence' and 'records')
Output: 'features' (dict of float statistics)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .base import BaseComputeTool


def compute_toolcall_stats(
    tool_sequence: list[str],
    records: list[dict],
    *,
    normalize_by_length: bool = True,
) -> dict[str, float]:
    """
    Compute statistical features from a tool-call sequence.

    Returns a dict of float features:
        n_total, n_unique, n_state_changing, n_read_only,
        read_write_ratio, entropy, top_tool_fraction,
        per-tool counts (tool_count_<name>), ...
    """
    n_total = len(tool_sequence)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_unique": 0,
            "n_state_changing": 0,
            "n_read_only": 0,
            "read_write_ratio": 0.0,
            "entropy": 0.0,
            "top_tool_fraction": 0.0,
            "state_change_fraction": 0.0,
        }

    counts = Counter(tool_sequence)
    n_unique = len(counts)

    n_sc = sum(1 for r in records if r.get("is_state_changing", True))
    n_ro = n_total - n_sc

    # Shannon entropy of tool distribution
    entropy = 0.0
    for c in counts.values():
        p = c / n_total
        if p > 0:
            entropy -= p * math.log2(p)

    # Most-used tool fraction
    top_count = counts.most_common(1)[0][1] if counts else 0
    top_tool_fraction = top_count / n_total

    stats: dict[str, float] = {
        "n_total": float(n_total),
        "n_unique": float(n_unique),
        "n_state_changing": float(n_sc),
        "n_read_only": float(n_ro),
        "read_write_ratio": n_ro / n_sc if n_sc > 0 else float(n_ro),
        "entropy": entropy,
        "top_tool_fraction": top_tool_fraction,
        "state_change_fraction": n_sc / n_total if n_total > 0 else 0.0,
    }

    # Per-tool counts (prefixed to avoid collisions)
    for tool_name, count in counts.most_common():
        safe_name = tool_name.replace(" ", "_").replace("-", "_").lower()
        stats[f"tool_count_{safe_name}"] = float(count)
        if normalize_by_length and n_total > 0:
            stats[f"tool_frac_{safe_name}"] = count / n_total

    return stats


class ToolCallStats(BaseComputeTool):
    """T05: Compute tool-call count & ratio statistics."""

    tool_id = "T05"
    tool_name = "toolcall_count_stats"
    input_type = "features"
    output_type = "features"

    def _process(self, input_data: Any, context: dict) -> tuple[Any, dict]:
        tool_seq = input_data.get("tool_sequence", [])
        records = input_data.get("records", [])
        normalize = self.params.get("normalize_by_length", True)

        stats = compute_toolcall_stats(
            tool_seq, records, normalize_by_length=normalize
        )

        # Merge upstream features so downstream tools can access both
        merged = {**input_data, **stats}
        return merged, {"n_features": len(stats)}
