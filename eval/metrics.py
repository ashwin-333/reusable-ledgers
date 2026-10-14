"""Graded AUC, macro AUC, and paired bootstrap intervals over the released score tables.

Rules match the scorers that produced the paper:
  * AUC is Mann-Whitney with ties counted 0.5; bootstrap draws enter as row multiplicities.
  * A check is scored when its label is present and the system has a prediction for it.
    Labels outside a policy's domain are blank in labels/*.csv, so they drop out here.
  * Macro AUC is the mean of the defined per-policy AUCs.
  * Within-task draws resample trajectories within each group; task-level draws resample whole
    groups. Groups are taken in first-appearance order of the row set; seed 20260925, 10,000 draws.
  * An interval is the 2.5th and 97.5th percentile of the defined macro deltas.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260925
N_BOOT = 10000


class Results:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.scores: dict[tuple[str, str], dict[tuple[str, str], tuple]] = defaultdict(dict)
        with (self.root / "results" / "scores.csv").open() as handle:
            for row in csv.DictReader(handle):
                key = (row["traj_id"], row["policy_id"])
                self.scores[(row["suite"], row["system"])][key] = (
                    float(row["suspicion"]) if row["suspicion"] else 1.0,
                    row["decision"],
                    float(row["prompt_tokens"]) if row["prompt_tokens"] else None,
                    float(row["completion_tokens"]) if row["completion_tokens"] else None,
                )
        self.rowsets: dict[str, list[tuple[str, str]]] = defaultdict(list)
        with (self.root / "results" / "trajectories.csv").open() as handle:
            for row in csv.DictReader(handle):
                self.rowsets[row["rowset"]].append((row["traj_id"], row["group"]))
        self.labels: dict[str, dict[tuple[str, str], bool | None]] = {}
        for path in sorted((self.root / "labels").glob("*.csv")):
            if path.stem == "contrastive_pairs":
                continue
            table = {}
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    table[(row["traj_id"], row["policy_id"])] = None if row["label"] == "" else row["label"] == "1"
            self.labels[path.stem] = table
        self.pairs = []
        pairs_path = self.root / "labels" / "contrastive_pairs.csv"
        if pairs_path.is_file():
            with pairs_path.open() as handle:
                self.pairs = list(csv.DictReader(handle))
        figure = self.root / "results" / "figure_data.csv"
        self.figure = list(csv.DictReader(figure.open())) if figure.is_file() else []
        coverage = self.root / "results" / "coverage_counts.json"
        self.coverage = json.loads(coverage.read_text()) if coverage.is_file() else {}

    def has(self, suite: str, system: str) -> bool:
        return (suite, system) in self.scores

    def systems(self, suite: str) -> list[str]:
        return sorted(system for s, system in self.scores if s == suite)

    def policies(self, suite: str) -> list[str]:
        seen = []
        for _traj, policy in self.labels[suite]:
            if policy not in seen:
                seen.append(policy)
        return seen

    def table(self, suite: str, system: str) -> dict[tuple[str, str], tuple]:
        if system.startswith("mean:"):
            left, right = system[5:].split("+")
            a, b = self.scores[(suite, left)], self.scores[(suite, right)]
            return {key: ((a[key][0] + b[key][0]) / 2, "", None, None) for key in a if key in b}
        return self.scores[(suite, system)]

    def tokens_per_check(self, suite: str, system: str, rowset: str | None = None, policies=None) -> float | None:
        table = self.table(suite, system)
        keep = None if rowset is None else {traj for traj, _ in self.rowsets[rowset]}
        per_traj = defaultdict(list)
        for (traj, policy), (_s, _d, prompt, completion) in table.items():
            if keep is not None and traj not in keep:
                continue
            if policies is not None and policy not in policies:
                continue
            if prompt is None and completion is None:
                continue
            per_traj[traj].append((prompt or 0.0) + (completion or 0.0))
        if not per_traj:
            return None
        return float(np.mean([np.mean(values) for values in per_traj.values()]))


def auc_weighted(values: np.ndarray, gold: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """AUC for one weight vector (1-D) or a batch of weight vectors (2-D, one draw per row)."""
    w = np.atleast_2d(weights).astype(np.float64)
    pos = np.flatnonzero(gold)
    neg = np.flatnonzero(~gold)
    if pos.size == 0 or neg.size == 0:
        return np.full(w.shape[0], np.nan)
    diff = values[pos][:, None] - values[neg][None, :]
    kernel = (diff > 0).astype(np.float64) + 0.5 * (diff == 0)
    wp, wn = w[:, pos], w[:, neg]
    num = ((wp @ kernel) * wn).sum(1)
    den = wp.sum(1) * wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


class Suite:
    """One suite's rows in a fixed order, with per-policy arrays for any system."""

    def __init__(self, results: Results, suite: str, rowset: str, policies: list[str] | None = None):
        self.r = results
        self.suite = suite
        self.rowset = rowset
        rows = results.rowsets[rowset]
        self.paths = [traj for traj, _ in rows]
        self.group_keys = [group for _, group in rows]
        self.policies = list(policies) if policies else results.policies(suite)
        labels = results.labels[suite]
        self.gold = {}
        self.labeled = {}
        for policy in self.policies:
            column = [labels.get((path, policy)) for path in self.paths]
            self.labeled[policy] = np.asarray([value is not None for value in column])
            self.gold[policy] = np.asarray([bool(value) for value in column])

    @property
    def n(self) -> int:
        return len(self.paths)

    def groups(self) -> list[np.ndarray]:
        buckets: dict[str, list[int]] = {}
        for index, key in enumerate(self.group_keys):
            buckets.setdefault(key, []).append(index)
        return [np.asarray(members) for members in buckets.values()]

    def arrays(self, system: str, exclude: dict[str, np.ndarray] | None = None):
        table = self.r.table(self.suite, system)
        values, valid = {}, {}
        for policy in self.policies:
            val = np.ones(self.n)
            ok = self.labeled[policy].copy()
            for index, path in enumerate(self.paths):
                item = table.get((path, policy))
                if item is None:
                    ok[index] = False
                else:
                    val[index] = item[0]
            if exclude is not None and policy in exclude:
                ok &= ~exclude[policy]
            values[policy], valid[policy] = val, ok
        return values, valid

    def auc(self, system: str, policy: str, exclude=None) -> float | None:
        values, valid = self.arrays(system, exclude)
        out = auc_weighted(values[policy], self.gold[policy], valid[policy].astype(float))[0]
        return None if np.isnan(out) else float(out)

    def macro(self, system: str, exclude=None) -> float | None:
        values, valid = self.arrays(system, exclude)
        per = [auc_weighted(values[p], self.gold[p], valid[p].astype(float))[0] for p in self.policies]
        per = [x for x in per if not np.isnan(x)]
        return float(np.mean(per)) if per else None

    def _batch_macro(self, system: str, mult: np.ndarray, exclude=None) -> np.ndarray:
        values, valid = self.arrays(system, exclude)
        return np.stack([auc_weighted(values[p], self.gold[p], mult * valid[p]) for p in self.policies])

    def paired(self, left: str, right: str, scheme: str = "within", exclude=None) -> dict:
        mult = within_draws(self) if scheme == "within" else task_draws(self)
        a = self._batch_macro(left, mult, exclude)
        b = self._batch_macro(right, mult, exclude)
        both = ~np.isnan(a) & ~np.isnan(b)
        diff = np.where(both, a - b, 0.0)
        count = both.sum(0)
        defined = count > 0
        deltas = diff.sum(0)[defined] / count[defined]
        point_l, point_r = self.macro(left, exclude), self.macro(right, exclude)
        return {
            "left": point_l,
            "right": point_r,
            "delta": None if point_l is None or point_r is None else point_l - point_r,
            "ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))] if deltas.size else None,
        }


_DRAW_CACHE: dict[tuple, np.ndarray] = {}


def within_draws(suite: Suite) -> np.ndarray:
    key = ("within", suite.rowset)
    if key not in _DRAW_CACHE:
        rng = np.random.default_rng(SEED)
        groups = suite.groups()
        out = np.zeros((N_BOOT, suite.n))
        for draw in range(N_BOOT):
            for members in groups:
                pick = rng.integers(0, len(members), size=len(members))
                np.add.at(out[draw], members[pick], 1)
        _DRAW_CACHE[key] = out
    return _DRAW_CACHE[key]


def task_draws(suite: Suite) -> np.ndarray:
    key = ("task", suite.rowset)
    if key not in _DRAW_CACHE:
        rng = np.random.default_rng(SEED)
        groups = suite.groups()
        out = np.zeros((N_BOOT, suite.n))
        for draw in range(N_BOOT):
            for index in rng.integers(0, len(groups), size=len(groups)):
                out[draw, groups[index]] += 1
        _DRAW_CACHE[key] = out
    return _DRAW_CACHE[key]


def flat_paired_auc(left: np.ndarray, right: np.ndarray, gold: np.ndarray) -> dict:
    """Unstratified paired bootstrap over trajectories (the CUA tables)."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, left.size, size=(N_BOOT, left.size))
    mult = np.zeros((N_BOOT, left.size))
    for draw in range(N_BOOT):
        np.add.at(mult[draw], idx[draw], 1)
    a = auc_weighted(left, gold, mult)
    b = auc_weighted(right, gold, mult)
    ok = ~np.isnan(a) & ~np.isnan(b)
    deltas = (a - b)[ok]
    ones = np.ones(left.size)
    point_l, point_r = auc_weighted(left, gold, ones)[0], auc_weighted(right, gold, ones)[0]
    return {"left": float(point_l), "right": float(point_r), "delta": float(point_l - point_r),
            "ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]}
