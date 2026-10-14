"""T09 — Isolation Forest on step features.

Unsupervised anomaly detection: flags outlier trajectories based on numeric
features (from T05/T06). The small model just consumes the score.

Input:  'features' (dict of float features from T05/T06)
Output: 'features' (original features + anomaly_score, is_anomaly flag)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseComputeTool

# Feature keys to use from upstream tools (T05/T06)
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
]


def _extract_feature_vector(
    features: dict, keys: list[str]
) -> np.ndarray:
    """Pull numeric features into a 1-D array, defaulting missing to 0."""
    return np.array([float(features.get(k, 0.0)) for k in keys])


class IsolationForestTool(BaseComputeTool):
    """T09: Isolation Forest anomaly detection on trajectory features.

    Since we process trajectories one at a time but IsolationForest needs a
    population, this tool operates in two modes:

    1. **Single-trajectory** (default): Uses feature heuristics to produce a
       pseudo-anomaly score based on deviation from expected ranges.
    2. **Batch** (when context contains 'population_features'): Fits
       IsolationForest on the population and scores the current trajectory.
    """

    tool_id = "T09"
    tool_name = "isolation_forest"
    input_type = "features"
    output_type = "features"

    def _process(self, input_data: Any, context: dict) -> tuple[Any, dict]:
        feature_keys = self.params.get("feature_keys", _DEFAULT_FEATURE_KEYS)
        contamination = self.params.get("contamination", 0.1)

        vec = _extract_feature_vector(input_data, feature_keys)

        population = context.get("population_features")
        if population is not None and len(population) >= 10:
            # Batch mode: fit on population, score this trajectory
            anomaly_score, is_anomaly = self._batch_score(
                vec, population, feature_keys, contamination
            )
        else:
            # Single-trajectory heuristic mode
            anomaly_score, is_anomaly = self._heuristic_score(input_data)

        merged = {
            **input_data,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
        }
        return merged, {
            "anomaly_score": anomaly_score,
            "mode": "batch" if population else "heuristic",
        }

    def _batch_score(
        self,
        vec: np.ndarray,
        population: list[dict],
        feature_keys: list[str],
        contamination: float,
    ) -> tuple[float, bool]:
        from sklearn.ensemble import IsolationForest

        X = np.array([_extract_feature_vector(p, feature_keys) for p in population])
        clf = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )
        clf.fit(X)

        # score_samples returns negative values for anomalies
        raw_score = clf.score_samples(vec.reshape(1, -1))[0]
        # Normalize to [0, 1] where 1 = most anomalous
        # IsolationForest scores are typically in [-1, 0]; more negative = more anomalous
        anomaly_score = max(0.0, min(1.0, -raw_score))
        is_anomaly = bool(clf.predict(vec.reshape(1, -1))[0] == -1)
        return anomaly_score, is_anomaly

    def _heuristic_score(self, features: dict) -> tuple[float, bool]:
        """
        Heuristic anomaly score for single-trajectory mode.
        Combines multiple signals that indicate unusual behavior.
        """
        signals: list[float] = []

        # High repetition is suspicious (structuring)
        rep_score = features.get("repetition_score", 0.0)
        signals.append(min(1.0, rep_score * 2))

        # High state-change fraction is suspicious
        sc_frac = features.get("state_change_fraction", 0.0)
        if sc_frac > 0.7:
            signals.append((sc_frac - 0.7) / 0.3)
        else:
            signals.append(0.0)

        # Very low entropy (doing the same thing over and over)
        entropy = features.get("entropy", 2.0)
        n_total = features.get("n_total", 1)
        if n_total > 5 and entropy < 1.0:
            signals.append(1.0 - entropy)
        else:
            signals.append(0.0)

        # Long bursts
        burst = features.get("max_burst_length", 1)
        if burst > 3:
            signals.append(min(1.0, (burst - 3) / 5))
        else:
            signals.append(0.0)

        anomaly_score = sum(signals) / len(signals) if signals else 0.0
        threshold = self.params.get("heuristic_threshold", 0.4)
        return anomaly_score, anomaly_score >= threshold
