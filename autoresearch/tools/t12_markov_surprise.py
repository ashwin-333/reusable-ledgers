"""T12 — Markov transition-surprise.

Detects low-probability tool→tool transitions. Builds a Markov model of tool
sequences and flags transitions that are surprising (low probability under the
model). A small model can't track global action sequences — this does it cheaply.

Input:  'features' (from T03: must contain 'tool_sequence')
Output: 'features' (per-step surprise scores + summary)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .base import BaseComputeTool


def compute_markov_surprise(
    tool_sequence: list[str],
    *,
    order: int = 1,
    smoothing: float = 0.1,
    reference_sequences: list[list[str]] | None = None,
) -> dict[str, Any]:
    """
    Compute transition surprise for each step in a tool sequence.

    If reference_sequences is provided, builds the Markov model from them
    (the 'benign population'). Otherwise builds from the trajectory itself
    (self-surprise: unusual patterns within the trajectory).

    Args:
        tool_sequence: list of tool names in order
        order: Markov order (1 = bigram, 2 = trigram, ...)
        smoothing: Laplace smoothing for unseen transitions
        reference_sequences: optional list of tool sequences to build the model from

    Returns:
        per_step_surprise: list of surprise values (nats) per transition
        mean_surprise: average surprise across all transitions
        max_surprise: maximum surprise (most unusual transition)
        max_surprise_transition: the most unusual transition tuple
        high_surprise_count: number of transitions above 2*mean
        surprise_trend: slope of surprise over time (rising = behavior changing)
    """
    n = len(tool_sequence)
    if n <= order:
        return {
            "per_step_surprise": [],
            "mean_surprise": 0.0,
            "max_surprise": 0.0,
            "max_surprise_transition": [],
            "high_surprise_count": 0,
            "surprise_trend": 0.0,
        }

    # Build transition counts
    transition_counts: dict[tuple, Counter] = defaultdict(Counter)
    context_counts: dict[tuple, int] = defaultdict(int)

    source_sequences = reference_sequences if reference_sequences else [tool_sequence]

    for seq in source_sequences:
        for i in range(len(seq) - order):
            context = tuple(seq[i : i + order])
            next_tool = seq[i + order]
            transition_counts[context][next_tool] += 1
            context_counts[context] += 1

    # Collect all unique tools for smoothing
    all_tools: set[str] = set()
    for seq in source_sequences:
        all_tools.update(seq)
    all_tools.update(tool_sequence)
    vocab_size = len(all_tools)

    # Compute per-step surprise
    surprises: list[float] = []
    transitions: list[tuple] = []

    for i in range(n - order):
        context = tuple(tool_sequence[i : i + order])
        next_tool = tool_sequence[i + order]

        count = transition_counts[context][next_tool]
        total = context_counts[context]

        # Laplace-smoothed probability
        prob = (count + smoothing) / (total + smoothing * vocab_size)
        surprise = -math.log(prob) if prob > 0 else 10.0  # cap at 10 nats

        surprises.append(surprise)
        transitions.append((*context, next_tool))

    if not surprises:
        return {
            "per_step_surprise": [],
            "mean_surprise": 0.0,
            "max_surprise": 0.0,
            "max_surprise_transition": [],
            "high_surprise_count": 0,
            "surprise_trend": 0.0,
        }

    mean_s = sum(surprises) / len(surprises)
    max_idx = max(range(len(surprises)), key=lambda i: surprises[i])
    max_s = surprises[max_idx]
    max_transition = list(transitions[max_idx])

    # High-surprise count (> 2x mean)
    threshold = mean_s * 2 if mean_s > 0 else 1.0
    high_count = sum(1 for s in surprises if s > threshold)

    # Surprise trend (linear slope)
    if len(surprises) >= 2:
        n_s = len(surprises)
        x_mean = (n_s - 1) / 2
        y_mean = mean_s
        num = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(surprises))
        den = sum((i - x_mean) ** 2 for i in range(n_s))
        trend = num / den if den > 0 else 0.0
    else:
        trend = 0.0

    return {
        "per_step_surprise": surprises,
        "mean_surprise": mean_s,
        "max_surprise": max_s,
        "max_surprise_transition": max_transition,
        "high_surprise_count": high_count,
        "surprise_trend": trend,
    }


class MarkovSurprise(BaseComputeTool):
    """T12: Markov transition-surprise detector."""

    tool_id = "T12"
    tool_name = "markov_transition_surprise"
    input_type = "features"
    output_type = "features"

    def _process(self, input_data: Any, context: dict) -> tuple[Any, dict]:
        tool_seq = input_data.get("tool_sequence", [])
        order = self.params.get("order", 1)
        smoothing = self.params.get("smoothing", 0.1)

        # Use reference sequences from population if available
        reference = None
        pop = context.get("population_tool_sequences")
        if pop and len(pop) >= 5:
            reference = pop

        result = compute_markov_surprise(
            tool_seq, order=order, smoothing=smoothing, reference_sequences=reference
        )

        # Don't include per_step_surprise in merged (can be large)
        summary = {k: v for k, v in result.items() if k != "per_step_surprise"}
        merged = {**input_data, **summary}

        return merged, {
            "mean_surprise": result["mean_surprise"],
            "max_surprise": result["max_surprise"],
            "high_surprise_count": result["high_surprise_count"],
        }
