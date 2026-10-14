"""T07 — Action entropy / diversity.

One-number summaries of 'is behavior drifting': entropy of tool distribution,
novelty-over-time curve, and diversity metrics computed over a sliding window.

Input:  'features' (from T03: must contain 'tool_sequence')
Output: 'features' (entropy, diversity, novelty metrics)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .base import BaseComputeTool


def compute_action_entropy(
    tool_sequence: list[str],
    *,
    window_size: int = 10,
    smoothing: float = 0.0,
) -> dict[str, Any]:
    """
    Compute entropy and diversity metrics for a tool-call sequence.

    Returns:
        global_entropy: Shannon entropy of the full sequence
        windowed_entropies: entropy per sliding window
        novelty_curve: fraction of new tools seen at each step
        diversity_ratio: unique_tools / total_tools
        entropy_trend: slope of windowed entropy (positive = diversifying)
        entropy_variance: variance of windowed entropies
    """
    n = len(tool_sequence)
    if n == 0:
        return {
            "global_entropy": 0.0,
            "mean_windowed_entropy": 0.0,
            "entropy_trend": 0.0,
            "entropy_variance": 0.0,
            "diversity_ratio": 0.0,
            "novelty_curve_mean": 0.0,
            "novelty_curve_final": 0.0,
        }

    # Global entropy
    counts = Counter(tool_sequence)
    global_entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            global_entropy -= p * math.log2(p)

    # Diversity ratio
    diversity_ratio = len(counts) / n

    # Novelty curve: at each position, what fraction of tools seen so far are new?
    seen: set[str] = set()
    novelty_values: list[float] = []
    for tool in tool_sequence:
        is_new = tool not in seen
        seen.add(tool)
        novelty_values.append(1.0 if is_new else 0.0)

    novelty_mean = sum(novelty_values) / len(novelty_values) if novelty_values else 0.0
    novelty_final = novelty_values[-1] if novelty_values else 0.0

    # Windowed entropy
    windowed_entropies: list[float] = []
    if n >= window_size:
        for start in range(n - window_size + 1):
            window = tool_sequence[start : start + window_size]
            wc = Counter(window)
            ent = 0.0
            for c in wc.values():
                p = c / window_size
                if p > 0:
                    ent -= p * math.log2(p)
            windowed_entropies.append(ent)
    else:
        windowed_entropies = [global_entropy]

    mean_windowed = sum(windowed_entropies) / len(windowed_entropies)

    # Entropy trend (simple linear slope)
    if len(windowed_entropies) >= 2:
        n_w = len(windowed_entropies)
        x_mean = (n_w - 1) / 2
        y_mean = mean_windowed
        num = sum((i - x_mean) * (e - y_mean) for i, e in enumerate(windowed_entropies))
        den = sum((i - x_mean) ** 2 for i in range(n_w))
        entropy_trend = num / den if den > 0 else 0.0
    else:
        entropy_trend = 0.0

    # Entropy variance
    if len(windowed_entropies) >= 2:
        var = sum((e - mean_windowed) ** 2 for e in windowed_entropies) / (
            len(windowed_entropies) - 1
        )
    else:
        var = 0.0

    return {
        "global_entropy": global_entropy,
        "mean_windowed_entropy": mean_windowed,
        "entropy_trend": entropy_trend,
        "entropy_variance": var,
        "diversity_ratio": diversity_ratio,
        "novelty_curve_mean": novelty_mean,
        "novelty_curve_final": novelty_final,
    }


class ActionEntropy(BaseComputeTool):
    """T07: Compute action entropy and diversity metrics."""

    tool_id = "T07"
    tool_name = "action_entropy_diversity"
    input_type = "features"
    output_type = "features"

    def _process(self, input_data: Any, context: dict) -> tuple[Any, dict]:
        tool_seq = input_data.get("tool_sequence", [])
        window = self.params.get("window_size", 10)

        result = compute_action_entropy(tool_seq, window_size=window)
        merged = {**input_data, **result}
        return merged, {
            "global_entropy": result["global_entropy"],
            "diversity_ratio": result["diversity_ratio"],
        }
