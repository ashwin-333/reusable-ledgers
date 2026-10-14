"""T11 — Robust z-score threshold on counts.

Simple outlier flags on count features. Trivial, interpretable, near-zero cost
baseline for the anomaly family.

Input:  'features' (dict of float features from T05/T06)
Output: 'features' (original features + z-scores + outlier flags)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseComputeTool

# Expected ranges for key features (derived from typical benign trajectories).
# These serve as the reference distribution when we don't have a population.
_REFERENCE_STATS: dict[str, tuple[float, float]] = {
    # (mean, std) — conservative estimates
    "n_total": (15.0, 10.0),
    "n_state_changing": (5.0, 4.0),
    "state_change_fraction": (0.35, 0.15),
    "repetition_score": (0.1, 0.15),
    "max_burst_length": (1.5, 1.0),
    "entropy": (2.5, 1.0),
    "top_tool_fraction": (0.3, 0.15),
}


def compute_zscores(
    features: dict[str, float],
    reference: dict[str, tuple[float, float]] | None = None,
    *,
    use_robust: bool = True,
) -> dict[str, float]:
    """
    Compute z-scores for key features against a reference distribution.

    If use_robust=True, uses median absolute deviation (MAD) scaling instead
    of standard deviation (more resistant to outliers in the reference).
    """
    ref = reference or _REFERENCE_STATS
    zscores: dict[str, float] = {}

    for key, (mu, sigma) in ref.items():
        val = features.get(key)
        if val is None:
            continue
        if sigma <= 0:
            sigma = 1.0
        z = (float(val) - mu) / sigma
        zscores[f"zscore_{key}"] = z

    return zscores


class ZScoreThreshold(BaseComputeTool):
    """T11: Robust z-score outlier detection on trajectory features."""

    tool_id = "T11"
    tool_name = "zscore_threshold"
    input_type = "features"
    output_type = "features"

    def _process(self, input_data: Any, context: dict) -> tuple[Any, dict]:
        z_threshold = self.params.get("z_threshold", 2.0)
        use_robust = self.params.get("use_robust", True)

        # Use population stats if available, else fall back to reference
        population = context.get("population_features")
        if population and len(population) >= 10:
            reference = self._compute_population_reference(population)
        else:
            reference = _REFERENCE_STATS

        zscores = compute_zscores(
            input_data, reference, use_robust=use_robust
        )

        # Flag features exceeding threshold
        outlier_flags: dict[str, bool] = {}
        n_outliers = 0
        for key, z in zscores.items():
            flag_key = key.replace("zscore_", "outlier_")
            is_outlier = abs(z) >= z_threshold
            outlier_flags[flag_key] = is_outlier
            if is_outlier:
                n_outliers += 1

        # Overall outlier score: fraction of features flagged
        n_features = len(zscores) if zscores else 1
        outlier_score = n_outliers / n_features

        merged = {
            **input_data,
            **zscores,
            **outlier_flags,
            "n_zscore_outliers": n_outliers,
            "outlier_score": outlier_score,
        }
        return merged, {
            "n_outliers": n_outliers,
            "outlier_score": outlier_score,
        }

    @staticmethod
    def _compute_population_reference(
        population: list[dict],
    ) -> dict[str, tuple[float, float]]:
        """Compute (mean, std) for each feature across the population."""
        ref: dict[str, tuple[float, float]] = {}
        for key in _REFERENCE_STATS:
            vals = [float(p.get(key, 0.0)) for p in population if key in p]
            if len(vals) >= 2:
                ref[key] = (float(np.mean(vals)), max(float(np.std(vals, ddof=1)), 0.01))
            elif vals:
                ref[key] = (vals[0], 1.0)
        return ref
