#!/usr/bin/env python3
"""Export per-check scores and labels from stored prediction directories.

Reads prediction JSONs and label files, and writes only identifiers, labels,
decisions, graded suspicions, and token counts. No trajectory, ledger, policy,
or model-output text is written. Every input path is a command-line argument.

Outputs under --out:
  results/scores.csv           suite, system, traj_id, policy_id, label, decision, suspicion, prompt_tokens, completion_tokens
  results/trajectories.csv     rowset, position, traj_id, group (row orders and bootstrap strata)
  results/coverage_counts.json counts from the action-coverage audit, plus CUA writer output-limit counts
  results/sequential_tokens.csv system, traj_id, kind, index, prompt_tokens, completion_tokens (writer usage and
                               each width-1 reader call in frozen policy order, SHADE-Arena atomic, all 300)
  results/figure_data.csv      Fig. 2 and Fig. 4 values, without the provenance columns
  labels/<suite>.csv           traj_id, policy_id, label
  labels/contrastive_pairs.csv pair_id, family, set, version, label
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

SCORE_COLUMNS = ["suite", "system", "traj_id", "policy_id", "label", "decision", "suspicion", "prompt_tokens", "completion_tokens"]
TAU_ACTION = (
    "airline_completed_cancellation",
    "airline_completed_booking",
    "airline_completed_flight_change",
    "retail_completed_order_cancellation",
    "retail_completed_return",
    "retail_completed_exchange",
    "retail_completed_item_modification",
)
TAU_COMP = ("retail_two_orders", "total_charged_or_refunded_over_100")
JUDGMENT = ("P2", "P3")
CAP = "output token limit"
UNSAFE = re.compile(r"(^/|^~|/home/|/data/|/Users/|\\\\|[A-Za-z]:\\)")


def _task(row: dict) -> str:
    for key in ("task_id", "task", "task_name", "shade_task"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return str(row.get("domain") or "na")


def _grade(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1.0
    number = float(value)
    return number if 1.0 <= number <= 10.0 else 1.0


def _float_grade(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    return number if 1.0 <= number <= 10.0 else 1.0


def _int_grade(value) -> float:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return 1.0


def _label(value) -> str:
    if value is None:
        return ""
    return "1" if bool(value) else "0"


def _num(value) -> str:
    if value is None or value == "":
        return ""
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(round(value, 6))


def _load_dir(directory: Path) -> list[dict]:
    if not directory.is_dir():
        return []
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def _bench(directory: Path) -> tuple[list[dict], list[str]]:
    payload = json.loads((directory / "multi_policy_benchmark.json").read_text())
    policies_file = directory / "policies.json"
    source = json.loads(policies_file.read_text()) if policies_file.is_file() else payload
    return payload["trajectories"], [item["id"] for item in source["policies"]]


def _domain_mask(policy: str, row: dict) -> bool:
    if policy.startswith("airline_"):
        return str(row.get("domain") or "") == "airline"
    if policy.startswith("retail_"):
        return str(row.get("domain") or "") == "retail"
    return True


class Exporter:
    def __init__(self, out: Path):
        self.out = out
        (out / "results").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        self.scores = (out / "results" / "scores.csv").open("w", newline="")
        self.writer = csv.writer(self.scores)
        self.writer.writerow(SCORE_COLUMNS)
        self.order = csv.writer((out / "results" / "trajectories.csv").open("w", newline=""))
        self.order.writerow(["rowset", "position", "traj_id", "group"])
        self.counts: dict[str, dict[str, int]] = {}

    def check_id(self, traj_id: str) -> str:
        if UNSAFE.search(traj_id):
            raise SystemExit("refusing to export an absolute or machine-specific id: " + traj_id[:40])
        return traj_id

    def rowset(self, name: str, rows: list[dict], with_domain: bool) -> None:
        for position, row in enumerate(rows):
            group = _task(row)
            if with_domain:
                group = str(row.get("domain") or "") + "|" + group
            self.order.writerow([name, position, self.check_id(row["path"]), group])

    def labels(self, suite: str, rows: list[dict], policies: list[str]) -> dict[tuple[str, str], str]:
        found = {}
        with (self.out / "labels" / f"{suite}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["traj_id", "policy_id", "label"])
            for row in rows:
                for policy in policies:
                    value = (row.get("labels") or {}).get(policy)
                    if not _domain_mask(policy, row):
                        value = None
                    label = _label(value)
                    found[(row["path"], policy)] = label
                    writer.writerow([self.check_id(row["path"]), policy, label])
        return found

    def emit(self, suite, system, traj_id, policy, label, decision, suspicion, prompt, completion) -> None:
        self.writer.writerow([suite, system, self.check_id(traj_id), policy, label, decision or "", _num(suspicion), _num(prompt), _num(completion)])
        self.counts.setdefault(suite, {}).setdefault(system, 0)
        self.counts[suite][system] += 1

    def policy_run(self, suite: str, system: str, directory: Path, rows: list[dict], policies: list[str], labels: dict) -> None:
        known = {row["path"] for row in rows}
        for saved in _load_dir(directory):
            path = saved.get("path")
            if path not in known:
                continue
            if "widths" in saved:
                block = (saved.get("widths") or {}).get("1") or {}
                prompt = [int(x or 0) for x in block.get("prompt_tokens") or []]
                completion = [int(x or 0) for x in block.get("completion_tokens") or []]
                n = len(prompt)
                per_prompt = sum(prompt) / n if n else None
                per_completion = sum(completion) / n if n else None
                decisions = block.get("decisions") or {}
                suspicion = block.get("suspicion") or {}
                for policy in policies:
                    self.emit(suite, system, path, policy, labels[(path, policy)], decisions.get(policy),
                              _grade(suspicion.get(policy)), per_prompt, per_completion)
            elif "versions" in saved:
                versions = {item.get("id"): item for item in saved.get("versions") or []}
                for policy in policies:
                    item = versions.get(policy) or {}
                    score = _grade(item.get("suspicion"))
                    decision = item.get("decision")
                    if not isinstance(item.get("suspicion"), (int, float)) or isinstance(item.get("suspicion"), bool) \
                            or decision not in ("violates", "does_not") or item.get("error"):
                        score = 1.0
                    self.emit(suite, system, path, policy, labels[(path, policy)], decision, score,
                              int(item.get("prompt_tokens") or 0), int(item.get("completion_tokens") or 0))

    def runs(self, suite: str, prefix: str, run_dir: Path, rows, policies, labels) -> None:
        if not run_dir.is_dir():
            print("skip (missing)", run_dir, flush=True)
            return
        for body in sorted(path for path in run_dir.iterdir() if (path / "predictions").is_dir()):
            self.policy_run(suite, prefix + body.name, body / "predictions", rows, policies, labels)
            print(suite, prefix + body.name, self.counts.get(suite, {}).get(prefix + body.name, 0), flush=True)

    def close(self) -> None:
        self.scores.close()


def _cua_rows(manifest: Path) -> list[dict]:
    rows = []
    for entry in json.loads(manifest.read_text()):
        benign = entry.get("is_benign")
        if isinstance(benign, str):
            benign = benign.strip().lower() == "true"
        rows.append({"path": entry["path"], "task_name": entry.get("task_name"), "labels": {"cua": not bool(benign)}})
    return rows


def _cua_emit(ex: Exporter, system: str, saved_rows: list[dict], known: set, labels: dict, rule) -> None:
    for saved in saved_rows:
        path = saved.get("path")
        if path not in known:
            continue
        ex.emit("cua", system, path, "cua", labels[(path, "cua")], saved.get("verdict"), rule(saved.get("suspicion")),
                int(saved.get("prompt_tokens") or 0), int(saved.get("completion_tokens") or 0))


def _capped(directory: Path) -> set[str]:
    found = set()
    for saved in _load_dir(directory):
        record = saved.get("record") or saved
        if CAP in str(record.get("ledger_error") or ""):
            key = saved.get("path") or record.get("path")
            if key:
                found.add(key)
    return found


def export_cua(ex: Exporter, args) -> dict:
    rows = _cua_rows(args.cua_manifest)
    ex.rowset("cua", rows, False)
    labels = ex.labels("cua", rows, ["cua"])
    known = {row["path"] for row in rows}
    oct3 = args.runs_root / "mp_controls" / "oct3"
    policy_dir = oct3 / "cua_policy"
    by_system = {}
    for body in sorted(path for path in policy_dir.iterdir() if (path / "predictions").is_dir()):
        loaded = _load_dir(body / "predictions")
        by_system[body.name] = {row["path"]: row for row in loaded if row.get("path")}
        _cua_emit(ex, body.name, loaded, known, labels, _float_grade)
    fix_dir = oct3 / "cua_fix" / "read"
    if fix_dir.is_dir():
        for body in sorted(path for path in fix_dir.iterdir() if (path / "predictions").is_dir()):
            _cua_emit(ex, "fix_" + body.name, _load_dir(body / "predictions"), known, labels, _float_grade)
    raw31 = args.runs_root / "graded" / "cua_cua" / "b" / "predictions"
    _cua_emit(ex, "raw31", _load_dir(raw31), known, labels, _grade)
    reread_root = oct3 / "cua_reread"
    reread = {}
    if reread_root.is_dir():
        for body in sorted(path for path in reread_root.iterdir() if (path / "predictions").is_dir()):
            loaded = _load_dir(body / "predictions")
            for condition in sorted({str(row.get("condition")) for row in loaded if row.get("condition")}):
                chosen = [row for row in loaded if str(row.get("condition")) == condition]
                reread[(body.name, condition)] = {row["path"]: row for row in chosen if row.get("path")}
                _cua_emit(ex, "reread_%s_%s" % (body.name, condition), chosen, known, labels, _float_grade)
    capped_counts = {}
    raw = by_system.get("raw") or {}
    for spec in args.cua_writer_records:
        name, _, directory = spec.partition("=")
        capped = _capped(Path(directory))
        capped_counts[name] = {"n_output_limit": len(capped & known), "n_trajectories": len(known)}
        plain_body = {"trained": "ledger_c", "trained_seed2": "ledger_c2"}.get(name)
        sources = {"plain": by_system.get(plain_body or "", {}), "plus": reread.get((name, "i"), {})}
        for kind, source in sources.items():
            if not source:
                continue
            chosen = []
            for row in rows:
                path = row["path"]
                pick = raw.get(path) if path in capped else source.get(path)
                if pick is not None:
                    chosen.append(pick)
            _cua_emit(ex, "fallback_%s_%s" % (name, kind), chosen, known, labels, _float_grade)
    return capped_counts


def export_sequential(ex: Exporter, runs_root: Path) -> None:
    with (ex.out / "results" / "sequential_tokens.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "traj_id", "kind", "index", "prompt_tokens", "completion_tokens"])
        for system in ("a12", "ledger_C", "ledger_C_seed2", "b12"):
            if system != "b12":
                for saved in _load_dir(runs_root / "shade" / system / "constructor_records"):
                    usage = (saved.get("record") or {}).get("ledger_usage") or {}
                    writer.writerow([system, ex.check_id(saved["path"]), "writer", 0,
                                     int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)])
            for saved in _load_dir(runs_root / "multi_policy_real" / ("e5_" + system) / "predictions"):
                block = (saved.get("widths") or {}).get("1") or {}
                prompt = block.get("prompt_tokens") or []
                completion = block.get("completion_tokens") or []
                for index in range(max(len(prompt), len(completion))):
                    writer.writerow([system, ex.check_id(saved["path"]), "reader", index,
                                     _num(prompt[index] if index < len(prompt) else None),
                                     _num(completion[index] if index < len(completion) else None)])
            print("sequential", system, flush=True)


def _judgment_rows(bench_dir: Path, labels_path: Path, manifest_path: Path) -> list[dict]:
    payload = json.loads(manifest_path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("trajectories") or payload.get("rows") or payload["entries"]
    digest_by_path = {row["path"]: row["sha256"] for row in payload if row.get("path") and row.get("sha256")}
    by_sha = {}
    for item in json.loads(labels_path.read_text()):
        if item.get("policy_id") in JUDGMENT:
            if item.get("label_source") not in ("agreed", "human-adjudicated"):
                raise SystemExit("unknown label source")
            by_sha[(item["sha256"], item["policy_id"])] = item
    rows, _ = _bench(bench_dir)
    out = []
    for row in rows:
        digest = row.get("sha256") or digest_by_path.get(row["path"])
        labels = {}
        for policy in JUDGMENT:
            item = by_sha.get((digest, policy))
            labels[policy] = None if item is None else item["violated"] == "yes"
        out.append({**row, "labels": labels})
    return out


def export_contrastive(ex: Exporter, args) -> None:
    items = json.loads(args.paired_bench.read_text())["items"]
    with (ex.out / "labels" / "contrastive_pairs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "family", "set", "version", "label"])
        for item in items:
            for spec in item["versions"]:
                writer.writerow([ex.check_id(str(item["id"])), item.get("family"), item.get("set"), spec["name"], _label(spec.get("violation"))])
    labels = {(str(item["id"]), spec["name"]): _label(spec.get("violation")) for item in items for spec in item["versions"]}
    controls = args.runs_root / "mp_controls"
    for run, prefix in (("paired_as8", "as8_"), ("paired_policy", ""), ("paired_policy_31b", "31b_")):
        run_dir = controls / run
        if not run_dir.is_dir():
            continue
        for body in sorted(path for path in run_dir.iterdir() if (path / "predictions").is_dir()):
            for saved in _load_dir(body / "predictions"):
                pair_id = str(saved["id"])
                for spec in saved.get("versions") or []:
                    key = (pair_id, spec["name"])
                    if key not in labels:
                        continue
                    ex.emit("contrastive", prefix + body.name, pair_id, spec["name"], labels[key], spec.get("decision"),
                            _int_grade(spec.get("suspicion")), spec.get("prompt_tokens"), spec.get("completion_tokens"))
            print("contrastive", prefix + body.name, flush=True)


def _numbers_only(value):
    if isinstance(value, dict):
        return {str(k): _numbers_only(v) for k, v in value.items() if not isinstance(v, str) or re.fullmatch(r"[A-Za-z0-9_.\-]+", v)}
    if isinstance(value, list):
        return [_numbers_only(v) for v in value if not isinstance(v, str) or re.fullmatch(r"[A-Za-z0-9_.\-]+", v)]
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, required=True, help="directory holding mp_controls/ and graded/")
    ap.add_argument("--shade-atomic-bench", type=Path, required=True, help="SHADE-Arena atomic benchmark directory")
    ap.add_argument("--shade-paraphrased-bench", type=Path, required=True)
    ap.add_argument("--shade-compositional-bench", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True, help="JSON with the 100-trajectory subset 'paths'")
    ap.add_argument("--tau-benches", type=Path, required=True, help="directory holding tau_action/, tau_nl/, tau_relational/, tau2_atomic/, tau2_nl/")
    ap.add_argument("--tau-corrected-labels", type=Path, required=True)
    ap.add_argument("--judgment-labels", type=Path, required=True)
    ap.add_argument("--shade-manifest", type=Path, required=True, help="SHADE-Arena manifest with path and sha256")
    ap.add_argument("--cua-manifest", type=Path, required=True)
    ap.add_argument("--cua-writer-records", action="append", default=[], help="NAME=DIR of writer constructor_records, e.g. trained=.../ledger_C/constructor_records")
    ap.add_argument("--paired-bench", type=Path, required=True)
    ap.add_argument("--coverage-json", type=Path, required=True)
    ap.add_argument("--coverage-neutral-json", type=Path, help="coverage_neutral.json from the format-neutral identifier audit")
    ap.add_argument("--figure-data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to write into a non-empty directory: " + str(args.out))

    ex = Exporter(args.out)
    controls = args.runs_root / "mp_controls"
    oct3 = controls / "oct3"
    robust = controls / "robust_20261005"
    subset = json.loads(args.subset.read_text())["paths"]

    shade = {
        "shade_atomic": (args.shade_atomic_bench, [("subset100", ""), ("shade100_31b", "31b_"), ("subset100_31b", "31bcal_"), ("oct3/shade_en", "ev_")], "shade_act"),
        "shade_paraphrased": (args.shade_paraphrased_bench, [("nl100", ""), ("nl100_31b", "31b_"), ("oct3/shade_nl", "ev_")], None),
        "shade_compositional": (args.shade_compositional_bench, [("rel300", ""), ("rel300_31b", "31b_"), ("oct3/shade_rel", "ev_")], "shade_rel"),
    }
    for suite, (bench, runs, robust_name) in shade.items():
        rows, policies = _bench(bench)
        by_path = {row["path"]: row for row in rows}
        ex.rowset(suite + "_all", rows, False)
        if suite != "shade_compositional":
            ex.rowset(suite, [by_path[path] for path in subset], False)
        else:
            ex.rowset(suite, rows, False)
        labels = ex.labels(suite, rows, policies)
        for run, prefix in runs:
            ex.runs(suite, prefix, controls / run, rows, policies, labels)
        if robust_name:
            for experiment, prefix in (("policy_only", "po_"), ("reason", "reason_"), ("reason_schema", "reasonschema_"), ("qwen", "qwen_")):
                ex.runs(suite, prefix, robust / experiment / robust_name, rows, policies, labels)

    corrected = {row["path"]: row["labels"] for row in json.loads(args.tau_corrected_labels.read_text())["rows"]}
    tau = {
        "tau_atomic": ("tau_action", "tau_act", "tau_action", TAU_ACTION),
        "tau_paraphrased": ("tau_nl", "tau_nl", "tau_nl", TAU_ACTION),
        "tau_compositional": ("tau_relational", "tau_rel", "tau_relational", TAU_COMP),
    }
    for suite, (bench_name, stored, zipped, policies) in tau.items():
        rows, _ = _bench(args.tau_benches / bench_name)
        rows = [{**row, "labels": {**(row.get("labels") or {}), **{p: corrected[row["path"]].get(p) for p in policies}}} for row in rows]
        ex.rowset(suite, rows, True)
        labels = ex.labels(suite, rows, list(policies))
        ex.runs(suite, "", oct3 / stored, rows, list(policies), labels)
        ex.runs(suite, "zip_", oct3 / "zip_reread" / zipped, rows, list(policies), labels)
        if suite == "tau_paraphrased":
            for experiment, prefix in (("policy_only", "po_"), ("reason", "reason_"), ("reason_schema", "reasonschema_"), ("qwen", "qwen_")):
                ex.runs(suite, prefix, robust / experiment / "tau_nl", rows, list(policies), labels)

    for suite, folder in (("tau2_atomic", "tau2_atomic"), ("tau2_paraphrased", "tau2_nl")):
        rows, policies = _bench(args.tau_benches / folder)
        ex.rowset(suite, rows, True)
        labels = ex.labels(suite, rows, policies)
        ex.runs(suite, "", oct3 / folder, rows, policies, labels)

    rows = _judgment_rows(args.shade_atomic_bench, args.judgment_labels, args.shade_manifest)
    ex.rowset("judgment", rows, False)
    labels = ex.labels("judgment", rows, list(JUDGMENT))
    ex.runs("judgment", "", controls / "judgment_20261005", rows, list(JUDGMENT), labels)

    capped = export_cua(ex, args)
    export_contrastive(ex, args)
    export_sequential(ex, args.runs_root)
    ex.close()

    coverage = _numbers_only(json.loads(args.coverage_json.read_text()))
    coverage["cua_writer_output_limit"] = capped
    if args.coverage_neutral_json:
        neutral = json.loads(args.coverage_neutral_json.read_text())
        coverage["neutral"] = {
            "n_successful_state_changing": int(neutral["n_successful_state_changing"]),
            "systems": {system: {key: block[key] for key in ("n_actions_with_identifier", "anywhere", "anywhere_share", "paired", "paired_share")}
                        for system, block in neutral["systems"].items()},
        }
    (args.out / "results" / "coverage_counts.json").write_text(json.dumps(coverage, indent=1) + "\n")

    with args.figure_data.open() as source, (args.out / "results" / "figure_data.csv").open("w", newline="") as dest:
        reader = csv.DictReader(source)
        keep = [name for name in reader.fieldnames if name not in ("source", "stored_report")]
        writer = csv.DictWriter(dest, fieldnames=keep, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            writer.writerow(row)

    leaks = []
    for path in args.out.rglob("*"):
        if path.is_file():
            text = path.read_text(errors="replace")
            if re.search(r"/home/|/data/|/Users/|user_data", text):
                leaks.append(str(path.relative_to(args.out)))
    if leaks:
        raise SystemExit("machine-specific paths found in outputs; nothing should be copied: " + ", ".join(leaks))
    (args.out / "results" / "export_counts.json").write_text(json.dumps(ex.counts, indent=1, sort_keys=True) + "\n")
    print(json.dumps({suite: len(systems) for suite, systems in ex.counts.items()}), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
