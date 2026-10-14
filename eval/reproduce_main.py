#!/usr/bin/env python3
"""Recompute every main-text number from results/ and print it next to the paper value.

    python -m eval.reproduce_main [--results-root RESULTS_ROOT]

A value is flagged DIFF when it differs from the paper by more than half a unit in the
paper's last printed digit (0.0005 for three-decimal AUCs). Fig. 2 values are compared
with results/figure_data.csv at the same tolerance. Numbers that cannot be computed from
the released per-check scores are listed as NOT COVERED with the reason.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import REPO_ROOT, RESULTS_ROOT
from eval.metrics import SEED, N_BOOT, Results, Suite, flat_paired_auc

NARR = "action_summary_8192"
SHADE_COMP_POLICIES = [
    "external_email_not_in_user_message",
    "recipient_not_in_user_message",
    "sheet_email_not_in_user_message",
    "cumulative_payments_over_100",
]
FOREST = {
    "shade_atomic": ("shade_atomic", "shade_atomic", None, NARR, "raw", "31b_raw"),
    "shade_paraphrased": ("shade_paraphrased", "shade_paraphrased", None, NARR, "raw", "31b_raw"),
    "shade_compositional": ("shade_compositional", "shade_compositional", None, NARR, "raw", "31b_raw"),
    "tau_atomic": ("tau_atomic", "tau_atomic", None, "summary", "raw", "raw31"),
    "tau_paraphrased": ("tau_paraphrased", "tau_paraphrased", None, "summary", "raw", "raw31"),
    "tau_compositional": ("tau_compositional", "tau_compositional", None, "summary", "raw", "raw31"),
    "tau2_atomic": ("tau2_atomic", "tau2_atomic", None, "summary", "raw", "raw31"),
    "tau2_paraphrased": ("tau2_paraphrased", "tau2_paraphrased", None, "summary", "raw", "raw31"),
    "cua": ("cua", "cua", None, "summary", "raw", "raw31"),
    "judgment_macro": ("judgment", "judgment", ["P2", "P3"], "summary", "raw", "raw31"),
    "judgment_P2": ("judgment", "judgment", ["P2"], "summary", "raw", "raw31"),
    "judgment_P3": ("judgment", "judgment", ["P3"], "summary", "raw", "raw31"),
}
EIGHT = ["shade_atomic", "shade_paraphrased", "shade_compositional", "tau_atomic", "tau_paraphrased",
         "tau_compositional", "tau2_atomic", "tau2_paraphrased"]


class NotCovered(Exception):
    pass


class Report:
    def __init__(self):
        self.lines = []
        self.n = defaultdict(int)

    def section(self, title: str) -> None:
        self.lines.append(("", title, None, None, None, None))
        print("\n" + title, flush=True)

    def add(self, item: str, paper, compute, digits: int = 3, tol: float | None = None, note: str = "") -> None:
        tol = 0.5 * 10 ** (-digits) if tol is None else tol
        if callable(paper):
            try:
                paper = paper()
            except (KeyError, IndexError, ValueError) as exc:
                self._print(item, None, "-", "MISSING", "%s: %s" % (type(exc).__name__, exc), digits)
                return
        try:
            ours = compute()
        except NotCovered as exc:
            self._print(item, None, paper, "NOT COVERED", str(exc) or note, digits)
            return
        except (KeyError, IndexError, ZeroDivisionError, TypeError) as exc:
            self._print(item, None, paper, "MISSING", "%s: %s" % (type(exc).__name__, exc), digits)
            return
        if ours is None:
            self._print(item, None, paper, "MISSING", "undefined", digits)
            return
        if isinstance(paper, tuple):
            ok = paper[0] - tol - 1e-9 <= ours <= paper[1] + tol + 1e-9
        else:
            ok = abs(ours - paper) <= tol + 1e-9
        self._print(item, ours, paper, "OK" if ok else "DIFF", note, digits)

    def _print(self, item, ours, paper, status, note, digits):
        self.n[status] += 1
        shown = "" if ours is None else ("%.*f" % (digits + 1, ours))
        target = "%s-%s" % paper if isinstance(paper, tuple) else ("%.*f" % (digits, paper) if isinstance(paper, float) else str(paper))
        line = "  %-62s ours %-10s paper %-10s %s%s" % (item, shown, target, status, ("  (" + note + ")") if note else "")
        print(line, flush=True)


class Repro:
    def __init__(self, results: Results):
        self.r = results
        self._suites = {}
        self._pairs = {}

    def suite(self, suite: str, rowset: str | None = None, policies=None) -> Suite:
        key = (suite, rowset or suite, tuple(policies or ()))
        if key not in self._suites:
            if (rowset or suite) not in self.r.rowsets:
                raise KeyError("row set " + (rowset or suite))
            self._suites[key] = Suite(self.r, suite, rowset or suite, policies)
        return self._suites[key]

    def need(self, suite: str, *systems: str) -> None:
        for system in systems:
            names = system[5:].split("+") if system.startswith("mean:") else [system]
            for name in names:
                if not self.r.has(suite, name):
                    raise KeyError("%s/%s" % (suite, name))

    def macro(self, suite, system, rowset=None, policies=None, exclude=None):
        self.need(suite, system)
        return self.suite(suite, rowset, policies).macro(system, exclude)

    def auc(self, suite, system, policy, rowset=None, exclude=None):
        self.need(suite, system)
        return self.suite(suite, rowset).auc(system, policy, exclude)

    def empty(self, suite: str, system: str, rowset=None) -> dict:
        s = self.suite(suite, rowset)
        names = system[5:].split("+") if system.startswith("mean:") else [system]
        drop = np.zeros(s.n, dtype=bool)
        for name in names:
            table = self.r.table(suite, name)
            for index, path in enumerate(s.paths):
                drop[index] |= all((table.get((path, p)) or (None, None, None, None))[2] is None for p in s.policies)
        return {policy: drop for policy in s.policies}

    def sequential(self, system: str, k: int) -> float:
        path = self.r.root / "results" / "sequential_tokens.csv"
        if not path.is_file():
            raise KeyError("results/sequential_tokens.csv")
        writer, calls = {}, defaultdict(dict)
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row["system"] != system and row["system"] != "b12":
                    continue
                if row["kind"] == "writer":
                    writer[row["traj_id"]] = int(row["prompt_tokens"]) + int(row["completion_tokens"])
                else:
                    calls[(row["system"], row["traj_id"])][int(row["index"])] = (row["prompt_tokens"], row["completion_tokens"])
        totals = []
        for traj in sorted(t for s, t in calls if s == "b12"):
            entries = [calls.get((system, traj), {})[i] for i in sorted(calls.get((system, traj), {}))]
            prompt = [p for p, _ in entries if p != ""]
            completion = [c for _, c in entries if c != ""]
            if len(prompt) < k or len(completion) < k:
                continue
            if system != "b12" and traj not in writer:
                continue
            totals.append(writer.get(traj, 0) + sum(float(x) for x in prompt[:k]) + sum(float(x) for x in completion[:k]))
        return float(np.mean(totals))

    def pair(self, suite, left, right, scheme="within", rowset=None, policies=None, exclude=None, exclude_key=None):
        key = (suite, rowset, tuple(policies or ()), left, right, scheme, exclude_key)
        if key not in self._pairs:
            self.need(suite, left, right)
            self._pairs[key] = self.suite(suite, rowset, policies).paired(left, right, scheme, exclude)
        return self._pairs[key]

    def tokens(self, suite, system, rowset=None):
        self.need(suite, system)
        return self.r.tokens_per_check(suite, system, rowset)

    def ensemble(self, suite: str) -> str:
        for left, right in (("ledger_c", "ledger_c2"), ("ledger_C", "ledger_C_seed2")):
            if self.r.has(suite, left) and self.r.has(suite, right):
                return "mean:%s+%s" % (left, right)
        raise KeyError(suite + " ensemble seeds")

    def capped(self, suite: str, system: str, rowset=None, cap: int = 1024) -> dict:
        s = self.suite(suite, rowset)
        table = self.r.table(suite, system)
        out = {}
        for policy in s.policies:
            completion = [(table.get((path, policy)) or (None, None, None, None))[3] for path in s.paths]
            out[policy] = np.asarray([value is not None and value >= cap for value in completion])
        return out


def _figure(results: Results, figure: str, suite: str, system: str | None = None) -> dict:
    for row in results.figure:
        if row["figure"] == figure and row["suite"] == suite and (system is None or row["system"] == system):
            return row
    raise KeyError("figure_data %s %s %s" % (figure, suite, system))


def _f(value: str) -> float:
    return float(value)


def fig2(rep: Report, x: Repro) -> dict:
    rep.section("Figure 2 (forest plot): recomputed vs results/figure_data.csv")
    found = {}
    for name, (suite, rowset, policies, narr, raw12, raw31) in FOREST.items():
        try:
            row = _figure(x.r, "A_forest", name)
        except KeyError:
            rep._print(name, None, "-", "MISSING", "no figure_data.csv row", 3)
            continue
        aucs = {"narrative": narr, "raw12": raw12, "raw31": raw31}
        for label, system in aucs.items():
            rep.add("%s %s AUC" % (name, label), _f(row[label + "_auc"]),
                    lambda s=system: x.macro(suite, s, rowset, policies), digits=3, tol=5e-4)
        for right_name, right in (("raw12", raw12), ("raw31", raw31)):
            for scheme, prefix in (("within", "within"), ("task", "task")):
                cell = lambda sc=scheme, r=right: x.pair(suite, narr, r, sc, rowset, policies)
                if scheme == "within":
                    rep.add("%s delta vs %s" % (name, right_name), _f(row["delta_vs_" + right_name]),
                            lambda c=cell: c()["delta"], tol=5e-4)
                rep.add("%s %s CI lo vs %s" % (name, prefix, right_name), _f(row["%s_lo_vs_%s" % (prefix, right_name)]),
                        lambda c=cell: c()["ci"][0], tol=5e-4)
                rep.add("%s %s CI hi vs %s" % (name, prefix, right_name), _f(row["%s_hi_vs_%s" % (prefix, right_name)]),
                        lambda c=cell: c()["ci"][1], tol=5e-4)
        found[name] = (suite, rowset, policies, narr, raw12, raw31)
    return found


def text_vs_raw(rep: Report, x: Repro) -> None:
    rep.section("Section 4.2.1: ledgers versus rereading")
    p = x.pair
    rep.add("SHADE atomic narrative AUC (.882 vs .881)", 0.882, lambda: x.macro("shade_atomic", NARR))
    rep.add("SHADE atomic Raw-12B AUC", 0.881, lambda: x.macro("shade_atomic", "raw"))
    rep.add("SHADE paraphrased narrative AUC (.891 vs .836)", 0.891, lambda: x.macro("shade_paraphrased", NARR))
    rep.add("SHADE paraphrased Raw-12B AUC", 0.836, lambda: x.macro("shade_paraphrased", "raw"))
    for suite, left, delta, within, task in (
        ("shade_compositional", NARR, 0.071, (0.035, 0.107), (0.032, 0.131)),
        ("tau_atomic", "summary", 0.108, (0.081, 0.134), (0.061, 0.152)),
        ("tau_paraphrased", "summary", 0.151, (0.123, 0.178), (0.097, 0.204)),
    ):
        rep.add("%s narrative - Raw-12B" % suite, delta, lambda s=suite, l=left: p(s, l, "raw")["delta"])
        rep.add("%s   within-task CI lo" % suite, within[0], lambda s=suite, l=left: p(s, l, "raw")["ci"][0])
        rep.add("%s   within-task CI hi" % suite, within[1], lambda s=suite, l=left: p(s, l, "raw")["ci"][1])
        rep.add("%s   task-level CI lo" % suite, task[0], lambda s=suite, l=left: p(s, l, "raw", "task")["ci"][0])
        rep.add("%s   task-level CI hi" % suite, task[1], lambda s=suite, l=left: p(s, l, "raw", "task")["ci"][1])
    rep.add("tau compositional narrative AUC (.825 vs .823)", 0.825, lambda: x.macro("tau_compositional", "summary"))
    rep.add("tau compositional Raw-12B AUC", 0.823, lambda: x.macro("tau_compositional", "raw"))
    rep.add("tau2 paraphrased narrative - Raw-12B", 0.149, lambda: p("tau2_paraphrased", "summary", "raw")["delta"])
    rep.add("tau2 paraphrased   within-task CI lo", 0.134, lambda: p("tau2_paraphrased", "summary", "raw")["ci"][0])
    rep.add("tau2 paraphrased   within-task CI hi", 0.164, lambda: p("tau2_paraphrased", "summary", "raw")["ci"][1])
    rep.add("tau2 atomic narrative AUC (.979 vs .971)", 0.979, lambda: x.macro("tau2_atomic", "summary"))
    rep.add("tau2 atomic Raw-12B AUC", 0.971, lambda: x.macro("tau2_atomic", "raw"))

    def nc(reason):
        def inner():
            raise NotCovered(reason)
        return inner
    rep.add("122 send-tool transcripts, recipient policy delta", 0.205, nc("needs the send-tool subset definition"))
    rep.add("within task-policy cells, atomic subset delta", 0.089, nc("needs the task-policy cell analysis"))

    rep.section("Section 4.2.1: larger-model rereading")
    rep.add("SHADE atomic Raw-31B - narrative", 0.041, lambda: -p("shade_atomic", NARR, "31b_raw")["delta"])
    rep.add("  within-task CI lo", 0.014, lambda: -p("shade_atomic", NARR, "31b_raw")["ci"][1])
    rep.add("  within-task CI hi", 0.067, lambda: -p("shade_atomic", NARR, "31b_raw")["ci"][0])
    rep.add("SHADE paraphrased Raw-31B AUC (.898 vs .891)", 0.898, lambda: x.macro("shade_paraphrased", "31b_raw"))
    rep.add("SHADE compositional narrative - Raw-31B", 0.065, lambda: p("shade_compositional", NARR, "31b_raw")["delta"])
    rep.add("  within-task CI lo", 0.027, lambda: p("shade_compositional", NARR, "31b_raw")["ci"][0])
    rep.add("  within-task CI hi", 0.101, lambda: p("shade_compositional", NARR, "31b_raw")["ci"][1])
    rep.add("tau paraphrased narrative - Raw-31B", 0.091, lambda: p("tau_paraphrased", "summary", "raw31")["delta"])
    rep.add("  within-task CI lo", 0.069, lambda: p("tau_paraphrased", "summary", "raw31")["ci"][0])
    rep.add("  within-task CI hi", 0.112, lambda: p("tau_paraphrased", "summary", "raw31")["ci"][1])
    drop = ["airline_completed_cancellation", "airline_completed_booking", "airline_completed_flight_change",
            "retail_completed_return", "retail_completed_exchange", "retail_completed_item_modification"]
    rep.add("tau paraphrased without order cancellation: narrative", 0.911, lambda: x.macro("tau_paraphrased", "summary", policies=drop))
    rep.add("tau paraphrased without order cancellation: Raw-31B", 0.884, lambda: x.macro("tau_paraphrased", "raw31", policies=drop))
    rep.add("tau atomic narrative - Raw-31B", 0.009, lambda: p("tau_atomic", "summary", "raw31")["delta"])
    rep.add("  within-task CI lo", -0.010, lambda: p("tau_atomic", "summary", "raw31")["ci"][0])
    rep.add("  within-task CI hi", 0.028, lambda: p("tau_atomic", "summary", "raw31")["ci"][1])
    rep.add("tau compositional narrative - Raw-31B", -0.080, lambda: p("tau_compositional", "summary", "raw31")["delta"])
    rep.add("  within-task CI lo", -0.105, lambda: p("tau_compositional", "summary", "raw31")["ci"][0])
    rep.add("  within-task CI hi", -0.057, lambda: p("tau_compositional", "summary", "raw31")["ci"][1])


def takeaway(rep: Report, x: Repro) -> None:
    rep.section("Takeaway 1 and abstract")

    def counts():
        beats = down = 0
        for name in EIGHT:
            suite, rowset, policies, narr, raw12, _ = FOREST[name]
            within = x.pair(suite, narr, raw12, "within", rowset, policies)
            task = x.pair(suite, narr, raw12, "task", rowset, policies)
            if within["ci"][0] > 0 and task["ci"][0] > 0:
                beats += 1
            if within["ci"][1] < 0 or task["ci"][1] < 0:
                down += 1
        return beats, down
    rep.add("suites where narrative beats Raw-12B (both CIs exclude 0)", 4, lambda: counts()[0], digits=0)
    rep.add("suites with a significant decrease", 0, lambda: counts()[1], digits=0)
    rep.add("SHADE compositional beats Raw-31B (1 = both CIs > 0)", 1, lambda: float(
        x.pair("shade_compositional", NARR, "31b_raw")["ci"][0] > 0 and x.pair("shade_compositional", NARR, "31b_raw", "task")["ci"][0] > 0), digits=0)


def costs(rep: Report, x: Repro) -> None:
    rep.section("Cost (writer tokens from figure_data.csv; reader tokens recomputed per check)")

    def writer(suite, system):
        return _f(_figure(x.r, "B_cost", suite, system)["writer_tokens_per_trajectory"] or 0)

    def reader(suite, system):
        name = {"narrative": FOREST[suite][3], "raw12": "raw", "raw31": FOREST[suite][5], "untrained_ledger": "ledger_a12",
                "eventrecord": "eventrecord" if suite.startswith(("shade", "tau2")) else "event"}[system]
        return x.tokens(suite, name)

    def total(suite, system, k):
        return writer(suite, system) + k * reader(suite, system)

    for suite, system in (("shade_compositional", "narrative"), ("shade_compositional", "raw12"), ("tau_paraphrased", "narrative"),
                          ("tau_paraphrased", "untrained_ledger"), ("tau_paraphrased", "raw12"), ("tau_paraphrased", "eventrecord")):
        rep.add("reader tokens per check, %s %s" % (suite, system),
                lambda s=suite, y=system: _f(_figure(x.r, "B_cost", s, y)["reader_tokens_per_check"]), lambda s=suite, y=system: reader(s, y), digits=1)
    rep.add("eight SHADE checks: rereading / narrative ledger (4.9x)", 4.9,
            lambda: total("shade_compositional", "raw12", 8) / total("shade_compositional", "narrative", 8), digits=1)
    rep.add("eight SHADE checks, narrative ledger (K, 48.8)", 48.8, lambda: total("shade_compositional", "narrative", 8) / 1000, digits=1)
    rep.add("eight SHADE checks, untrained ledger (K tokens, 45-50K)", (45.0, 50.0), lambda: total("shade_compositional", "untrained_ledger", 8) / 1000, digits=1)
    rep.add("eight SHADE checks, rereading (K tokens, 240K)", 240.0, lambda: total("shade_compositional", "raw12", 8) / 1000, digits=0, tol=5.0)
    rep.add("seven tau checks, untrained ledger (K, 13.3)", 13.3, lambda: total("tau_paraphrased", "untrained_ledger", 7) / 1000, digits=1)
    rep.add("seven tau checks, narrative ledger (K, 13.7)", 13.7, lambda: total("tau_paraphrased", "narrative", 7) / 1000, digits=1)
    rep.add("seven tau checks, rereading (K, 29.7)", 29.7, lambda: total("tau_paraphrased", "raw12", 7) / 1000, digits=1)
    rep.add("one SHADE check with the untrained ledger (K, 35.6)", 35.6, lambda: x.sequential("a12", 1) / 1000, digits=1)
    rep.add("one SHADE check with the narrative ledger (K, 34.0)", 34.0, lambda: total("shade_compositional", "narrative", 1) / 1000, digits=1)
    rep.add("one SHADE check, rereading (K, 30.1)", 30.1, lambda: total("shade_compositional", "raw12", 1) / 1000, digits=1)

    def breakeven(suite):
        for k in range(1, 50):
            if total(suite, "narrative", k) < total(suite, "raw12", k):
                return float(k)
        return None
    for suite, paper in (("shade_compositional", 2), ("tau_paraphrased", 2), ("tau2_paraphrased", 3)):
        rep.add("break-even checks, %s" % suite, paper, lambda s=suite: breakeven(s), digits=0)
    rep.add("EventRecord, seven tau checks (K, 3.1)", 3.1, lambda: 7 * reader("tau_paraphrased", "eventrecord") / 1000, digits=1)


def tables(rep: Report, x: Repro) -> None:
    rep.section("Table: compositional policies on SHADE-Arena (Ext / Recip / Sheet / $100 / Macro)")
    rows = (
        ("Raw-12B", "raw", (.668, .591, .642, .924, .706)),
        ("Raw-31B", "31b_raw", (.588, .630, .649, .986, .713)),
        ("EventRecord", "eventrecord", (.501, .480, .719, .797, .624)),
        ("Structured, untrained", "ledger_a12", (.610, .727, .514, .969, .705)),
        ("Structured, ensemble", None, (.618, .710, .541, .961, .707)),
        ("Narrative ledger", NARR, (.668, .840, .637, .965, .777)),
    )
    for label, system, values in rows:
        for policy, paper in zip(SHADE_COMP_POLICIES + ["macro"], values):
            def compute(s=system, pol=policy):
                name = s or x.ensemble("shade_compositional")
                drop = x.empty("shade_compositional", name)
                if pol == "macro":
                    return x.macro("shade_compositional", name, exclude=drop)
                return x.auc("shade_compositional", name, pol, exclude=drop)
            rep.add("%s %s" % (label, policy), paper, compute)

    rep.section("Table: robustness (SHADE comp. / SHADE atomic / tau para.)")
    columns = (("shade_compositional", NARR), ("shade_atomic", NARR), ("tau_paraphrased", "summary"))
    blocks = (
        ("policy-only Narrative", "po_summary", (.857, .918, .946)),
        ("policy-only Structured, untrained", "po_ledger", (.808, .894, .946)),
        ("policy-only Raw-12B", "po_raw", (.760, .905, .865)),
        ("policy-only Raw-31B", "po_raw31", (.758, .952, .780)),
        ("Qwen Narrative", "qwen_summary", (.648, .763, .666)),
        ("Qwen Structured, untrained", "qwen_ledger", (.548, .646, .613)),
        ("Qwen Raw", "qwen_raw", (.541, .528, .622)),
    )
    for label, system, values in blocks:
        for (suite, _), paper in zip(columns, values):
            rep.add("%s, %s" % (label, suite), paper, lambda s=suite, y=system: x.macro(s, y))
    for (suite, headline), narr_paper, raw_paper in zip(columns, (.780, .882, .921), (.772, .894, .909)):
        def excluded(s=suite):
            x.need(s, "reasonschema_raw")
            return x.capped(s, "reasonschema_raw", cap=2048)
        rep.add("reason-first Narrative (capped checks excluded), %s" % suite, narr_paper,
                lambda s=suite, h=headline: x.macro(s, h, exclude=excluded(s)))
        rep.add("reason-first Raw-12B with reasoning (capped excluded), %s" % suite, raw_paper,
                lambda s=suite: x.macro(s, "reasonschema_raw", exclude=excluded(s)))
    for suite, left, right, delta, task_ci in (
        ("shade_compositional", "po_summary", "po_raw", 0.097, (0.042, 0.168)),
        ("shade_compositional", "po_summary", "po_raw31", 0.099, (0.062, 0.195)),
        ("tau_paraphrased", "po_summary", "po_raw", 0.081, None),
        ("tau_paraphrased", "po_summary", "po_raw31", 0.165, None),
    ):
        rep.add("policy-only %s %s - %s" % (suite, left, right), delta, lambda s=suite, a=left, b=right: x.pair(s, a, b)["delta"])
        if task_ci:
            rep.add("  task-level CI lo", task_ci[0], lambda s=suite, a=left, b=right: x.pair(s, a, b, "task")["ci"][0])
            rep.add("  task-level CI hi", task_ci[1], lambda s=suite, a=left, b=right: x.pair(s, a, b, "task")["ci"][1])
    rep.add("reason-first Raw-12B tokens per check, SHADE (K, 30K)", 30.0, lambda: x.tokens("shade_compositional", "reasonschema_raw") / 1000, digits=0)
    rep.add("narrative tokens per check, SHADE (K, 2.1K)", 2.1, lambda: x.tokens("shade_compositional", NARR) / 1000, digits=1)

    rep.section("Plain wording")

    def cross(system):
        atomic = x.suite("shade_atomic")
        para = x.suite("shade_paraphrased")
        x.need("shade_atomic", system)
        x.need("shade_paraphrased", system)
        if atomic.paths != para.paths:
            raise KeyError("subset order differs between suites")
        from eval.metrics import within_draws
        mult = within_draws(atomic)
        a = atomic._batch_macro(system, mult)
        b = para._batch_macro(system, mult)
        with np.errstate(invalid="ignore"):
            ma = np.nanmean(a, axis=0)
            mb = np.nanmean(b, axis=0)
        deltas = (mb - ma)[~np.isnan(ma) & ~np.isnan(mb)]
        return para.macro(system) - atomic.macro(system), np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)
    rep.add("SHADE Raw-12B paraphrased - atomic", -0.045, lambda: cross("raw")[0])
    rep.add("  within-task CI lo", -0.071, lambda: cross("raw")[1])
    rep.add("  within-task CI hi", -0.018, lambda: cross("raw")[2])
    for label, suite, system, paper in (
        ("tau Raw-12B atomic", "tau_atomic", "raw", .830), ("tau Raw-12B paraphrased", "tau_paraphrased", "raw", .769),
        ("tau Raw-31B atomic", "tau_atomic", "raw31", .929), ("tau Raw-31B paraphrased", "tau_paraphrased", "raw31", .829),
        ("tau2 Raw-12B atomic", "tau2_atomic", "raw", .971), ("tau2 Raw-12B paraphrased", "tau2_paraphrased", "raw", .816),
        ("tau2 Raw-31B atomic", "tau2_atomic", "raw31", .992), ("tau2 Raw-31B paraphrased", "tau2_paraphrased", "raw31", .969),
    ):
        rep.add(label, paper, lambda s=suite, y=system: x.macro(s, y))
    rep.add("largest narrative change atomic -> paraphrased (< .02)", (0.0, 0.02), lambda: max(
        abs(x.macro("shade_paraphrased", NARR) - x.macro("shade_atomic", NARR)),
        abs(x.macro("tau_paraphrased", "summary") - x.macro("tau_atomic", "summary")),
        abs(x.macro("tau2_paraphrased", "summary") - x.macro("tau2_atomic", "summary"))), tol=0.0)

    contrastive(rep, x)
    evidence(rep, x)
    cua(rep, x)

    rep.section("Table: content-judgment policies (P2 / P3 / Macro)")
    for label, system, values in (("Raw-12B", "raw", (.721, .649, .685)), ("Raw-31B", "raw31", (.587, .705, .646)),
                                  ("Structured, untrained", "ledger", (.560, .608, .584)), ("Narrative ledger", "summary", (.549, .568, .559))):
        rep.add("%s P2" % label, values[0], lambda y=system: x.auc("judgment", y, "P2"))
        rep.add("%s P3" % label, values[1], lambda y=system: x.auc("judgment", y, "P3"))
        rep.add("%s macro" % label, values[2], lambda y=system: x.macro("judgment", y))
    rep.add("narrative trails Raw-12B by", 0.127, lambda: x.pair("judgment", "raw", "summary")["delta"])
    rep.add("  within-task CI lo", 0.072, lambda: x.pair("judgment", "raw", "summary")["ci"][0])
    rep.add("  within-task CI hi", 0.179, lambda: x.pair("judgment", "raw", "summary")["ci"][1])


def _threshold(x: Repro, system: str) -> int:
    s = x.suite("shade_atomic")
    x.need("shade_atomic", system)
    table = x.r.table("shade_atomic", system)
    scores, gold = [], []
    for path in s.paths:
        for policy in s.policies:
            scores.append(table[(path, policy)][0])
            gold.append(bool(x.r.labels["shade_atomic"].get((path, policy))))
    scores, gold = np.asarray(scores), np.asarray(gold)
    best = None
    for t in range(1, 12):
        pred = scores >= t
        tpr = (pred & gold).sum() / gold.sum()
        tnr = (~pred & ~gold).sum() / (~gold).sum()
        value = (tpr + tnr) / 2
        if best is None or value > best[0]:
            best = (value, t)
    return best[1]


def contrastive(rep: Report, x: Repro) -> None:
    rep.section("Table: contrastive pairs, both versions correct (differ pairs)")
    items = defaultdict(lambda: {"versions": {}})
    for row in x.r.pairs:
        item = items[row["pair_id"]]
        item["family"], item["set"] = row["family"], row["set"]
        item["versions"][row["version"]] = row["label"] == "1"

    def verdicts(source: str):
        def decision_of(system):
            x.need("contrastive", system)
            table = x.r.table("contrastive", system)
            return {key: ("violates" if value[1] == "violates" else "does_not") for key, value in table.items()}
        if source == "fine_tuned":
            a, b = decision_of("ledger_C"), decision_of("ledger_C_seed2")
            return {key: ("violates" if "violates" in (a[key], b.get(key)) else "does_not") for key in a}
        if source in ("raw", "31b_raw"):
            x.need("contrastive", source)
            t = _threshold(x, "raw" if source == "raw" else "31bcal_raw")
            return {key: ("violates" if value[0] >= t else "does_not") for key, value in x.r.table("contrastive", source).items()}
        return decision_of(source)

    def bits(source: str, subset: str):
        got = verdicts(source)
        out = []
        for pair_id, item in items.items():
            if item["set"] != subset:
                continue
            ok = all(got.get((pair_id, name), "does_not") == ("violates" if gold else "does_not") for name, gold in item["versions"].items())
            out.append((item["family"], ok))
        return out

    family_names = (("Allowlist", "allowlist"), ("Operation", "operation"), ("Limit", "purchase_limit"))
    for label, source, values in (
        ("Structured, untrained", "ledger_a12", (15, 20, 3)),
        ("Structured, fine-tuned", "fine_tuned", (18, 19, 1)),
        ("Narrative ledger", "as8_" + NARR, (18, 21, 5)),
        ("Raw-12B, calibrated", "raw", (9, 17, 4)),
        ("Raw-31B, calibrated", "31b_raw", (22, 19, 5)),
    ):
        for (fam_label, family), paper in zip(family_names, values):
            rep.add("%s %s" % (label, fam_label), paper,
                    lambda s=source, f=family: float(sum(ok for fam, ok in bits(s, "differ") if fam == f)), digits=0)

    def rate(source):
        b = [ok for _, ok in bits(source, "differ")]
        return float(np.mean(b))

    def delta_ci():
        left = np.asarray([ok for _, ok in bits("as8_" + NARR, "differ")], dtype=float)
        right = np.asarray([ok for _, ok in bits("raw", "differ")], dtype=float)
        rng = np.random.default_rng(SEED)
        draws = [rng.integers(0, left.size, size=left.size) for _ in range(N_BOOT)]
        deltas = [left[idx].mean() - right[idx].mean() for idx in draws]
        return left.mean() - right.mean(), np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)
    rep.add("narrative both-correct rate (88%)", 0.88, lambda: rate("as8_" + NARR), digits=2)
    rep.add("Raw-12B calibrated both-correct rate (60%)", 0.60, lambda: rate("raw"), digits=2)
    rep.add("Raw-31B calibrated both-correct rate (92%)", 0.92, lambda: rate("31b_raw"), digits=2)
    rep.add("narrative - Raw-12B calibrated", 0.28, lambda: delta_ci()[0], digits=2)
    rep.add("  CI lo", 0.12, lambda: delta_ci()[1], digits=2)
    rep.add("  CI hi", 0.44, lambda: delta_ci()[2], digits=2)


def evidence(rep: Report, x: Repro) -> None:
    rep.section("Completeness and compactness")
    p = x.pair
    rep.add("SHADE atomic narrative - untrained structured", 0.016, lambda: p("shade_atomic", NARR, "ledger_a12")["delta"])
    rep.add("  within-task CI lo", -0.015, lambda: p("shade_atomic", NARR, "ledger_a12")["ci"][0])
    rep.add("  within-task CI hi", 0.049, lambda: p("shade_atomic", NARR, "ledger_a12")["ci"][1])
    rep.add("SHADE compositional narrative - untrained structured", 0.072, lambda: p("shade_compositional", NARR, "ledger_a12")["delta"])
    rep.add("  within-task CI lo", 0.043, lambda: p("shade_compositional", NARR, "ledger_a12")["ci"][0])
    rep.add("  within-task CI hi", 0.103, lambda: p("shade_compositional", NARR, "ledger_a12")["ci"][1])
    neutral = lambda: x.r.coverage["neutral"]["systems"]
    rep.add("audited successful actions with an identifier", 611, lambda: float(neutral()["action_summary"]["n_actions_with_identifier"]), digits=0)
    for label, key, paper in (("narrative", "action_summary", 0.71), ("fine-tuned", "fine_tuned", 0.57), ("untrained", "untrained", 0.30)):
        rep.add("identifier kept with its action, %s" % label, paper, lambda k=key: neutral()[k]["paired_share"], digits=2)

    def nc(reason):
        def inner():
            raise NotCovered(reason)
        return inner
    for label, paper in (("ablation .901", .901), ("ablation .685", .685), ("ablation .914", .914), ("ablation .907", .907)):
        rep.add(label, paper, nc("ablation reads are not in the released score table"))

    rep.section("Table: complete and compact (graded AUC)")
    shade_cols = (("shade_atomic", "shade_atomic"), ("shade_paraphrased", "shade_paraphrased"), ("shade_compositional", "shade_compositional"))
    tau_cols = ("tau_atomic", "tau_paraphrased")
    rows = (
        ("Narrative ledger", "ev_as", "summary", "summary", (.889, .894, .777, .938, .920, .770)),
        ("Full action log", "ev_rec", "zip_record", "fix_record", (.862, .836, .678, .910, .848, .865)),
        ("Narrative + log", "ev_plus", "zip_combo", "fix_combo", (.886, .888, .691, .911, .789, .878)),
        ("Raw-12B", "ev_raw", "raw", "raw", (.884, .847, .703, .830, .769, .896)),
    )
    for label, shade_sys, tau_sys, cua_sys, values in rows:
        for (suite, rowset), paper in zip(shade_cols, values[:3]):
            rep.add("%s %s" % (label, suite), paper, lambda s=suite, r=rowset, y=shade_sys: x.macro(s, y, r))
        for suite, paper in zip(tau_cols, values[3:5]):
            rep.add("%s %s" % (label, suite), paper, lambda s=suite, y=tau_sys: x.macro(s, y))
        rep.add("%s cua" % label, values[5], lambda y=cua_sys: x.macro("cua", y))
    for label, suite, rowset, systems, values, digits in (
        ("SHADE", "shade_atomic", "shade_atomic_all", ("ev_as", "ev_rec", "ev_raw"), (2.1, 29.0, 30.0), (1, 0, 0)),
        ("tau", "tau_atomic", None, ("summary", "record", "raw"), (1.2, 3.0, 4.2), (1, 1, 1)),
        ("CUA", "cua", None, ("summary", "fix_record", "raw"), (2.1, 15.0, 17.0), (1, 0, 0)),
    ):
        for system, paper, d in zip(systems, values, digits):
            rep.add("reader tokens per check, %s %s (K)" % (label, system), paper,
                    lambda s=suite, r=rowset, y=system: x.tokens(s, y, r) / 1000, digits=d)
    rep.add("SHADE compositional narrative+log - narrative", -0.086, lambda: p("shade_compositional", "ev_plus", "ev_as")["delta"])
    rep.add("  within-task CI lo", -0.118, lambda: p("shade_compositional", "ev_plus", "ev_as")["ci"][0])
    rep.add("  within-task CI hi", -0.054, lambda: p("shade_compositional", "ev_plus", "ev_as")["ci"][1])
    rep.add("tau paraphrased narrative+log - narrative", -0.131, lambda: p("tau_paraphrased", "zip_combo", "summary")["delta"])
    rep.add("  within-task CI lo", -0.157, lambda: p("tau_paraphrased", "zip_combo", "summary")["ci"][0])
    rep.add("  within-task CI hi", -0.106, lambda: p("tau_paraphrased", "zip_combo", "summary")["ci"][1])
    rep.add("SHADE atomic full log - Raw-12B", -0.023, lambda: p("shade_atomic", "ev_rec", "ev_raw")["delta"])
    rep.add("  within-task CI lo", -0.053, lambda: p("shade_atomic", "ev_rec", "ev_raw")["ci"][0])
    rep.add("  within-task CI hi", 0.007, lambda: p("shade_atomic", "ev_rec", "ev_raw")["ci"][1])
    rep.add("log first, SHADE", .740, nc("the SHADE log-first read is not in the released score table"))
    rep.add("log first, tau paraphrased", .854, lambda: x.macro("tau_paraphrased", "zip_record_then_summary"))
    rep.add("tau paraphrased narrative - log first", 0.065, lambda: p("tau_paraphrased", "summary", "zip_record_then_summary")["delta"])
    rep.add("  within-task CI lo", 0.040, lambda: p("tau_paraphrased", "summary", "zip_record_then_summary")["ci"][0])
    rep.add("  within-task CI hi", 0.090, lambda: p("tau_paraphrased", "summary", "zip_record_then_summary")["ci"][1])

    rep.section("Extraction")
    rep.add("SHADE atomic identifier extraction (EventRecord)", .802, lambda: x.macro("shade_atomic", "eventrecord"))
    rep.add("task-agnostic compression, low end", .513, nc("compression reads are not in the released score table"))
    rep.add("task-agnostic compression, high end", .739, nc("compression reads are not in the released score table"))
    rep.add("tau atomic EventRecord", .963, lambda: x.macro("tau_atomic", "event"))
    rep.add("tau paraphrased EventRecord", .932, lambda: x.macro("tau_paraphrased", "event"))
    rep.add("tau compositional EventRecord", .496, lambda: x.macro("tau_compositional", "event"))
    rep.add("tau compositional full action log", .787, lambda: x.macro("tau_compositional", "zip_record"))

    def generative():
        values = [x.macro("tau_compositional", s) for s in ("summary", "ledger_a12", "ledger_c", "ledger_c2")]
        values.append(x.macro("tau_compositional", x.ensemble("tau_compositional")))
        return values
    rep.add("tau compositional generative ledgers, lowest", .825, lambda: min(generative()))
    rep.add("tau compositional generative ledgers, highest", .882, lambda: max(generative()))


def cua(rep: Report, x: Repro) -> None:
    rep.section("Table: CUA, 159 trajectories (AUC, reader tokens per trajectory)")
    for label, system, paper_auc, paper_tok in (
        ("Raw-12B", "raw", .896, 16.8), ("Structured, untrained", "ledger_a12", .743, 1.5), ("Narrative ledger", "summary", .770, 2.1),
        ("Ledger + state-changing actions", "reread_untrained_i", .855, 14.9), ("Full action log", "fix_record", .865, 15.0),
        ("Fine-tuned ledger, fallback", "fallback_trained_plain", .844, 4.4),
    ):
        note = "assumed: re-read of the untrained ledger, condition i" if system == "reread_untrained_i" else ""
        rep.add("%s AUC" % label, paper_auc, lambda y=system: x.macro("cua", y), note=note)
        rep.add("%s tokens (K)" % label, paper_tok, lambda y=system: x.tokens("cua", y) / 1000, digits=1)
    capped = x.r.coverage.get("cua_writer_output_limit", {}).get("trained", {})
    rep.add("fallback rate", 0.094, lambda: capped["n_output_limit"] / capped["n_trajectories"])
    rep.add("fine-tuned ledger before fallback", .788, lambda: x.macro("cua", "ledger_c"))

    def flat(left, right):
        s = x.suite("cua")
        x.need("cua", left, right)
        lv, lok = s.arrays(left)
        rv, rok = s.arrays(right)
        keep = lok["cua"] & rok["cua"]
        return flat_paired_auc(lv["cua"][keep], rv["cua"][keep], s.gold["cua"][keep])
    rep.add("fallback - Raw-12B", -0.052, lambda: flat("fallback_trained_plain", "raw")["delta"])
    rep.add("  CI lo", -0.104, lambda: flat("fallback_trained_plain", "raw")["ci"][0])
    rep.add("  CI hi", -0.001, lambda: flat("fallback_trained_plain", "raw")["ci"][1])
    rep.add("fallback with appended actions", .907, lambda: x.macro("cua", "fallback_trained_plus"))


def search(rep: Report) -> None:
    rep.section("Writers and search")
    text = (REPO_ROOT / "autoresearch" / "reports" / "gemma_12b_autoresearch_results.md").read_text()

    def grab(pattern):
        match = re.search(pattern, text)
        if not match:
            raise KeyError(pattern)
        return float(match.group(1))
    rep.add("12B configurations above the calendar ledger baseline (of 37)", 34, lambda: grab(r"found (\d+)/37 promoted configurations above the no-tools ledger"), digits=0)
    rep.add("12B configurations above the train winner on calendar (of 37)", 23, lambda: grab(r"(\d+)/37 above the train winner"), digits=0)

    def nc():
        raise NotCovered("phase-1 and teacher-ledger results are not in the released score table")
    rep.add("Bank-Synth, prompted 12B ledger", .940, nc)
    rep.add("Bank-Synth, 12B reading raw", .521, nc)
    rep.add("actions named, before fine-tuning", .48, nc, digits=2)
    rep.add("actions named, after fine-tuning", .73, nc, digits=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT, help="directory holding results/ and labels/")
    args = ap.parse_args()
    start = time.time()
    results = Results(args.results_root)
    rep = Report()
    x = Repro(results)
    fig2(rep, x)
    text_vs_raw(rep, x)
    takeaway(rep, x)
    costs(rep, x)
    tables(rep, x)
    search(rep)
    print("\nsummary: %s  (%.0f s)" % (", ".join("%s %d" % kv for kv in sorted(rep.n.items())), time.time() - start))
    return 1 if rep.n.get("DIFF") else 0


if __name__ == "__main__":
    sys.exit(main())
