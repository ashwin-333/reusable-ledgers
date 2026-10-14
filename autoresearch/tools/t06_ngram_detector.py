"""T06 — Tool-sequence n-gram / repetition detector.

Surfaces repeated action clusters and burst patterns that indicate structuring
(splitting one task into many). Specifically targets C3.

Input:  'features' (from T03: must contain 'tool_sequence')
Output: 'features' (repetition metrics + flagged clusters)
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .base import BaseComputeTool


def detect_ngram_repetitions(
    tool_sequence: list[str],
    *,
    n_values: tuple[int, ...] = (2, 3, 4),
    min_repeat_count: int = 2,
) -> dict[str, Any]:
    """
    Detect repeated n-gram patterns in a tool-call sequence.

    Returns:
        repeated_ngrams: list of (ngram_tuple, count) pairs exceeding min_repeat_count
        max_burst_length: longest consecutive run of the same tool
        burst_tools: tools that appear in bursts
        repetition_score: 0-1 score (higher = more repetitive)
    """
    if len(tool_sequence) < 2:
        return {
            "repeated_ngrams": [],
            "n_repeated_patterns": 0,
            "max_burst_length": len(tool_sequence),
            "burst_tools": [],
            "repetition_score": 0.0,
        }

    # N-gram extraction
    repeated: list[tuple[tuple[str, ...], int]] = []
    for n in n_values:
        if len(tool_sequence) < n:
            continue
        ngrams = [
            tuple(tool_sequence[i : i + n])
            for i in range(len(tool_sequence) - n + 1)
        ]
        counts = Counter(ngrams)
        for ngram, count in counts.most_common():
            if count >= min_repeat_count:
                repeated.append((ngram, count))

    # Burst detection: consecutive runs of the same tool
    max_burst = 1
    current_burst = 1
    burst_tools: set[str] = set()
    for i in range(1, len(tool_sequence)):
        if tool_sequence[i] == tool_sequence[i - 1]:
            current_burst += 1
            if current_burst > max_burst:
                max_burst = current_burst
                burst_tools.add(tool_sequence[i])
        else:
            current_burst = 1

    # Repetition score: fraction of the sequence covered by repeated n-grams
    total_positions = len(tool_sequence)
    covered = set()
    for ngram, count in repeated:
        n = len(ngram)
        for i in range(len(tool_sequence) - n + 1):
            if tuple(tool_sequence[i : i + n]) == ngram:
                for j in range(i, i + n):
                    covered.add(j)

    repetition_score = len(covered) / total_positions if total_positions > 0 else 0.0

    return {
        "repeated_ngrams": [
            {"pattern": list(ng), "count": c} for ng, c in repeated[:20]
        ],
        "n_repeated_patterns": len(repeated),
        "max_burst_length": max_burst,
        "burst_tools": sorted(burst_tools),
        "repetition_score": repetition_score,
    }


class NgramDetector(BaseComputeTool):
    """T06: Detect n-gram repetitions and bursts in tool sequences."""

    tool_id = "T06"
    tool_name = "ngram_repetition_detector"
    input_type = "features"
    output_type = "features"

    def _process(self, input_data: Any, context: dict) -> tuple[Any, dict]:
        tool_seq = input_data.get("tool_sequence", [])
        n_values = tuple(self.params.get("n_values", [2, 3, 4]))
        min_repeat = self.params.get("min_repeat_count", 2)

        result = detect_ngram_repetitions(
            tool_seq, n_values=n_values, min_repeat_count=min_repeat
        )

        merged = {**input_data, **result}
        return merged, {
            "n_patterns": result["n_repeated_patterns"],
            "max_burst": result["max_burst_length"],
        }
