"""Linear scorer — sign-corrected, z-normalized weighted sum of features.

The hand-tuned ThresholdScorer has three structural problems that the feature
probe made visible:

  * it weights features that carry no signal (anomaly_score, macro-AUC 0.47)
    most heavily, and inverted features (repetition_score, 0.45) positively;
  * it saturates counts (``min(1.0, n_flags / 3)``), collapsing a 519-value
    feature into ties;
  * it clips ``state_change_fraction`` below 0.5, discarding most of its range.

This scorer instead takes an explicit, fitted spec: each feature carries a
weight whose sign encodes its direction, and a (mean, std) pair used to
z-normalize it. Those numbers must be fit on a dev split and are recorded in
the pipeline YAML, so scoring stays a pure function of one trajectory — no
dependence on the evaluation population.

Params:
    features: [{name, weight, mean, std}, ...]
    clip:     z-score clamp, default 4.0 (limits single-feature dominance)
    scale:    logistic temperature, default 1.0
"""

from __future__ import annotations

import math
import time
from typing import Any

from .base import ToolResult

# Feature names prefixed with a scanner ID are read from that tool's output in
# the pipeline context rather than from the merged feature dict.
_SCANNER_PREFIXES = ("T20_", "T21_", "T22_")


def _resolve(name: str, features: dict, context: dict) -> float | None:
    """Look up a feature by name, following scanner prefixes into context."""
    for prefix in _SCANNER_PREFIXES:
        if name.startswith(prefix):
            tool_id = prefix.rstrip("_")
            result = context.get("outputs", {}).get(tool_id)
            if result is None or not isinstance(result.output, dict):
                return None
            val = result.output.get(name[len(prefix) :])
            return float(val) if isinstance(val, (int, float, bool)) else None

    val = features.get(name) if isinstance(features, dict) else None
    return float(val) if isinstance(val, (int, float, bool)) else None


class LinearScorer:
    """Weighted sum of z-normalized features, squashed to a 1-5 score."""

    tool_id = "linear_scorer"
    tool_name = "linear_feature_scorer"
    input_type = "features"
    output_type = "score"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def process(self, input_data: Any, context: dict) -> ToolResult:
        t0 = time.monotonic()
        specs = self.params.get("features", [])
        clip = float(self.params.get("clip", 4.0))
        scale = float(self.params.get("scale", 1.0))

        total = 0.0
        n_used = 0
        n_missing = 0
        for spec in specs:
            val = _resolve(spec["name"], input_data, context)
            if val is None or not math.isfinite(val):
                n_missing += 1
                continue
            std = float(spec.get("std", 1.0)) or 1.0
            z = (val - float(spec.get("mean", 0.0))) / std
            z = max(-clip, min(clip, z))
            total += float(spec.get("weight", 0.0)) * z
            n_used += 1

        # Logistic squash keeps the score in range without clipping ties at the
        # ends, which would reintroduce the tie problem the probe surfaced.
        score = 1.0 + 4.0 / (1.0 + math.exp(-total * scale / max(1, len(specs)) * 2.0))

        return ToolResult(
            output=score,
            output_type="score",
            cost_usd=0.0,
            latency_s=time.monotonic() - t0,
            metadata={
                "scoring_method": "linear_z",
                "n_features_used": n_used,
                "n_features_missing": n_missing,
                "raw_sum": total,
            },
        )

    def estimate_cost(self, input_data: Any) -> float:
        return 0.0
