"""T10 — Local Outlier Factor (LOF).

Density-based outlier score per trajectory. Complements Isolation Forest (T09)
by catching local anomalies that IForest may miss.

Input:  'features' (dict of float features from T05/T06/T07)
Output: 'features' (original features + lof_score, lof_is_anomaly)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseComputeTool

_DEFAULT_FEATURE_KEYS = [
    "n_total",
    "n_unique",
    "n_state_changing",
    "n_read_only",
    "read_write_ratio",
    "entropy",
    "top_tool_fraction",
    "state_change_fraction",
    "repetition_score",
    "max_burst_length",
    "n_repeated_patterns",
    "global_entropy",
    "diversity_ratio",
]


def _extract_feature_vector(features: dict, keys: list[str]) -> np.ndarray:
    return np.array([float(features.get(k, 0.0)) for k in keys])


class LocalOutlierFactor(BaseComputeTool):
    """T10: Local Outlier Factor anomaly detection.

    Like T09, operates in two modes:
    1. Single-trajectory: heuristic score from feature deviations
    2. Batch: fits sklearn LOF on a population
    """

    tool_id = "T10"
    tool_name = "local_outlier_factor"
    input_type = "features"
    output_type = "features"

    def _process(self, input_data: Any, context: dict) -> tuple[Any, dict]:
        feature_keys = self.params.get("feature_keys", _DEFAULT_FEATURE_KEYS)
        n_neighbors = self.params.get("n_neighbors", 20)
        contamination = self.params.get("contamination", 0.1)

        vec = _extract_feature_vector(input_data, feature_keys)
        population = context.get("population_features")

        if population is not None and len(population) >= max(n_neighbors + 1, 10):
            lof_score, is_anomaly = self._batch_score(
                vec, population, feature_keys, n_neighbors, contamination
            )
            mode = "batch"
        else:
            lof_score, is_anomaly = self._heuristic_score(input_data)
            mode = "heuristic"

        merged = {
            **input_data,
            "lof_score": lof_score,
            "lof_is_anomaly": is_anomaly,
        }
        return merged, {"lof_score": lof_score, "mode": mode}

    def _batch_score(
        self,
        vec: np.ndarray,
        population: list[dict],
        feature_keys: list[str],
        n_neighbors: int,
        contamination: float,
    ) -> tuple[float, bool]:
        from sklearn.neighbors import LocalOutlierFactor as LOF

        X = np.array([_extract_feature_vector(p, feature_keys) for p in population])
        # LOF with novelty=True allows scoring new samples
        clf = LOF(
            n_neighbors=min(n_neighbors, len(X) - 1),
            contamination=contamination,
            novelty=True,
        )
        clf.fit(X)

        raw_score = clf.score_samples(vec.reshape(1, -1))[0]
        # More negative = more anomalous. Normalize to [0, 1].
        lof_score = max(0.0, min(1.0, -raw_score - 1.0))
        is_anomaly = bool(clf.predict(vec.reshape(1, -1))[0] == -1)
        return lof_score, is_anomaly

    def _heuristic_score(self, features: dict) -> tuple[float, bool]:
        """Fallback heuristic when no population is available."""
        signals: list[float] = []

        # Low diversity + high repetition = local anomaly
        div = features.get("diversity_ratio", 0.5)
        if div < 0.2:
            signals.append(1.0 - div * 5)

        rep = features.get("repetition_score", 0.0)
        signals.append(min(1.0, rep * 1.5))

        # Very high or very low entropy relative to n_total
        n_total = features.get("n_total", 1)
        entropy = features.get("global_entropy", features.get("entropy", 1.5))
        if n_total > 5:
            max_possible = math.log2(n_total) if n_total > 1 else 1.0
            normalized = entropy / max_possible if max_possible > 0 else 0.5
            # Both extremes (very low and very high) are suspicious
            deviation = abs(normalized - 0.5) * 2
            signals.append(deviation)

        score = sum(signals) / len(signals) if signals else 0.0
        threshold = self.params.get("heuristic_threshold", 0.4)
        return score, score >= threshold


import math  # needed for heuristic_score
