"""Population anomaly scorer — distance from a fitted benign reference population.

AR-003 found that T09 (isolation forest) and T10 (LOF) are inert here, and the
cause was structural rather than a tuning problem: the evaluator scores one
trajectory at a time, so those tools fall back to fixed expected-range
heuristics with no population to isolate against. The filed follow-up was to
fit them on a reference population of benign trajectories instead.

This scorer does that while keeping scoring a pure function of one trajectory,
the same contract ``linear_scorer`` follows: the reference population's
per-feature location and scale are fitted offline (benign dev trajectories
only) and frozen into the pipeline YAML, so evaluation has no dependence on the
evaluation population.

Score is a robust squared Mahalanobis-style distance under a diagonal
covariance: the sum of squared robust z-scores across the configured features,
squashed to the usual 1-5 range. Unlike ``linear_scorer`` this is unsigned and
label-free — it asks "how far from typical benign behavior is this?", not "which
direction did the fitted weights say was suspicious?".

Params:
    features: [{name, center, scale}, ...]   fitted on benign trajectories only
    clip:     per-feature z clamp, default 4.0
    scale:    logistic temperature on the distance, default 1.0
"""

from __future__ import annotations

import math
import time
from typing import Any

from .base import ToolResult
from .linear_scorer import _resolve


class PopulationAnomalyScorer:
    """Distance from a frozen benign reference population, squashed to 1-5."""

    tool_id = "population_anomaly"
    tool_name = "population_anomaly_scorer"
    input_type = "features"
    output_type = "score"

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.features = self.params.get("features", [])
        if not self.features:
            raise ValueError(
                "population_anomaly requires a fitted 'features' list; run "
                "`python -m autoresearch fit-anomaly` to produce one"
            )
        self.clip = float(self.params.get("clip", 4.0))
        self.scale = float(self.params.get("scale", 1.0))

    def process(self, input_data: Any, context: dict) -> ToolResult:
        t0 = time.monotonic()

        total = 0.0
        n_used = 0
        for spec in self.features:
            value = _resolve(spec["name"], input_data, context)
            if value is None:
                continue
            scale = float(spec.get("scale", 0.0))
            if scale <= 0:
                continue
            z = (value - float(spec["center"])) / scale
            z = max(-self.clip, min(self.clip, z))
            total += z * z
            n_used += 1

        # Mean squared z keeps the distance comparable when a feature is missing.
        distance = total / n_used if n_used else 0.0
        squashed = 1.0 / (1.0 + math.exp(-self.scale * (distance - 1.0)))
        score = 1.0 + 4.0 * squashed

        return ToolResult(
            output=score,
            output_type="score",
            cost_usd=0.0,
            latency_s=time.monotonic() - t0,
            metadata={
                "scoring_method": "population_anomaly",
                "raw_sum": distance,
                "n_features_used": n_used,
            },
        )

    def estimate_cost(self, input_data: Any) -> float:
        return 0.0
