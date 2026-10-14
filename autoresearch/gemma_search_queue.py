"""Durable, train-only queue for parallel Gemma candidate evaluations.

This module never proposes candidates or accesses the calendar manifest. Each
worker consumes immutable candidate snapshots and uses one local Gemma server.
Multiple workers may run against the same queue directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .candidate_config import CandidateConfig
from .config import RESULTS_DIR
from .gemma_baseline import DEFAULT_PROMPT, _load_manifest, _write_json
from .gemma_candidate import candidate_source_sha256, evaluate
from .reconstruction_metric import GEMMA_DECODER_SYSTEM_PROMPT, RECONSTRUCTION_METRIC_NAME, SANITIZER_VERSION
from config import OPENAI_BASE_URL


STAGES = ("screen", "train")
BASELINE_CONFIG = RESULTS_DIR / "gemma_baseline" / "train" / "run_config.json"
BASELINE_RESULT = BASELINE_CONFIG.with_name("reconstruction_metric.json")


def _exclusive_json(path: Path, value: dict[str, Any]) -> bool:
    """Publish a complete JSON file once, without exposing partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            return False
    finally:
        temporary.unlink(missing_ok=True)


def _replace_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_config(root: Path) -> dict[str, Any]:
    config = json.loads((root / "search_config.json").read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported Gemma search workspace version")
    return config


def _runtime_tree_sha256() -> str:
    """Detect code/catalog edits that require a fresh worker process import."""
    root = Path(__file__).parent
    paths = [*root.glob("*.py"), *(root / "tools").glob("*.py"), root / "candidate_tools.json"]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _screen_entries(entries: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Choose one complete benign/malicious pair per task and construct."""
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for entry in entries:
        key = (entry["domain"], entry.get("ground_truth_construct") or entry["construct"])
        groups.setdefault(key, {}).setdefault(entry["pair_key"], []).append(entry)
    if len(groups) != 12:
        raise ValueError(f"expected three tasks × four constructs, found {len(groups)} strata")
    rng = random.Random(seed)
    selected: set[str] = set()
    for key, pairs in sorted(groups.items()):
        valid = [
            pair_key for pair_key, rows in pairs.items()
            if len(rows) == 2 and {bool(row["is_benign"]) for row in rows} == {True, False}
        ]
        if not valid:
            raise ValueError(f"no complete benign/malicious pair in stratum {key}")
        selected.add(rng.choice(sorted(valid)))
    return [entry for entry in entries if entry["pair_key"] in selected]


def initialize(
    *, root: Path, data_root: Path, train_manifest: Path, model_revision: str,
    wall_hours: float = 48.0, screen_seed: int = 20260923,
) -> dict[str, Any]:
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"search workspace is not empty: {root}")
    if not 0 < wall_hours <= 48:
        raise ValueError("wall_hours must be between 0 and 48")
    entries = _load_manifest(train_manifest)
    baseline = json.loads(BASELINE_CONFIG.read_text())["config"]
    manifest_sha = hashlib.sha256(train_manifest.read_bytes()).hexdigest()
    if manifest_sha != baseline["manifest_sha256"]:
        raise ValueError("train manifest differs from the completed Gemma baseline")
    if model_revision != baseline["model_revision"]:
        raise ValueError("model revision differs from the completed Gemma baseline")
    reader_hash = hashlib.sha256(GEMMA_DECODER_SYSTEM_PROMPT.encode()).hexdigest()
    scorer_path = Path(__file__).parent / "lds_v2.py"
    scorer_hash = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
    if reader_hash != baseline["reader_prompt_sha256"]:
        raise ValueError("fixed Gemma reader prompt differs from the completed baseline")
    if scorer_hash != baseline["scorer_code_sha256"]:
        raise ValueError("LDS-v2 scorer code differs from the completed baseline")
    if SANITIZER_VERSION != baseline["sanitizer_version"] or RECONSTRUCTION_METRIC_NAME != baseline["metric"]:
        raise ValueError("sanitizer or metric identity differs from the completed baseline")
    if baseline["unknown_policy"] != "pass":
        raise ValueError("baseline used a different UNKNOWN policy")
    screen = _screen_entries(entries, screen_seed)
    if len(screen) != 24:
        raise ValueError(f"screen manifest should contain 24 trajectories, found {len(screen)}")
    root.mkdir(parents=True, exist_ok=True)
    screen_manifest = root / "screen_manifest.json"
    _write_json(screen_manifest, screen)
    config = {
        "schema_version": 1,
        "data_root": str(data_root.resolve()),
        "train_manifest": str(train_manifest.resolve()),
        "train_manifest_sha256": manifest_sha,
        "screen_manifest": str(screen_manifest.resolve()),
        "model": baseline["model"],
        "model_revision": model_revision,
        "max_model_len": baseline["max_model_len"],
        "constructor_max_tokens": baseline["constructor_max_tokens"],
        "reader_max_tokens": baseline["reader_max_tokens"],
        "temperature": baseline["temperature"],
        "seed": baseline["seed"],
        "screen_seed": screen_seed,
        "wall_hours": wall_hours,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(root / "search_config.json", config)
    seed_candidate = CandidateConfig.materialize(
        prompt_name="frozen_gemma_seed",
        prompt_content=DEFAULT_PROMPT.read_text(),
        hypothesis="Completed no-tools Gemma baseline; fixed search comparator.",
    )
    store_candidate(root, seed_candidate)
    enqueue(root, seed_candidate.candidate_id, "screen")
    baseline_score = json.loads(BASELINE_RESULT.read_text())["score"]
    _write_json(root / "seed.json", {
        "candidate_id": seed_candidate.candidate_id,
        "full_train_lds_v2_from_completed_baseline": baseline_score,
        "source": str(BASELINE_RESULT),
    })
    return config


def store_candidate(root: Path, candidate: CandidateConfig) -> Path:
    path = root / "candidates" / f"{candidate.candidate_id}.json"
    snapshot = candidate.snapshot_dict()
    if not _exclusive_json(path, snapshot):
        old = CandidateConfig.from_dict(json.loads(path.read_text()))
        if old.execution_hash() != candidate.execution_hash():
            raise ValueError(f"candidate ID collision or modified snapshot: {path}")
    source_path = root / "candidates" / f"{candidate.candidate_id}.source.json"
    source_hash = candidate_source_sha256(candidate)
    if not _exclusive_json(source_path, {"candidate_source_sha256": source_hash}):
        old_hash = json.loads(source_path.read_text())["candidate_source_sha256"]
        if old_hash != source_hash:
            raise ValueError(
                f"candidate tool/evaluator code changed after snapshot: {candidate.candidate_id}"
            )
    return path


def enqueue(root: Path, candidate_id: str, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError("search workers evaluate only screen/train manifests")
    candidate_path = root / "candidates" / f"{candidate_id}.json"
    candidate = CandidateConfig.from_dict(json.loads(candidate_path.read_text()))
    if candidate.candidate_id != candidate_id:
        raise ValueError("candidate ID does not match its snapshot")
    source_hash = json.loads((root / "candidates" / f"{candidate_id}.source.json").read_text())[
        "candidate_source_sha256"
    ]
    job_id = f"{candidate_id}_{stage}"
    failed = root / "jobs" / "failed" / f"{job_id}.json"
    if failed.exists():
        raise ValueError(f"job previously failed; use retry-failed for {job_id}")
    for state in ("pending", "running", "done"):
        path = root / "jobs" / state / f"{job_id}.json"
        if path.exists():
            return path
    path = root / "jobs" / "pending" / f"{job_id}.json"
    if not _exclusive_json(path, {
        "job_id": job_id, "candidate_id": candidate_id, "stage": stage,
        "candidate_execution_hash": candidate.execution_hash(),
        "candidate_source_sha256": source_hash,
        "enqueued_at": datetime.now(timezone.utc).isoformat(),
    }):
        return path
    return path


def _deadline(root: Path, config: dict[str, Any]) -> float:
    path = root / "started_at.json"
    started = time.time()
    first_worker = _exclusive_json(path, {"started_at_unix": started})
    if first_worker:
        write_report(root)
    started = float(json.loads(path.read_text())["started_at_unix"])
    return started + float(config["wall_hours"]) * 3600


def _record_path(root: Path) -> Path:
    started = json.loads((root / "started_at.json").read_text())["started_at_unix"]
    day = datetime.fromtimestamp(started, timezone.utc).date().isoformat()
    suffix = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:8]
    return Path(__file__).parent.parent / "experiments" / f"{day}-gemma-autoresearch-{suffix}.md"


def write_report(root: Path) -> Path:
    """Create/update the dated run record only after a worker has started."""
    config = _load_config(root)
    seed = json.loads((root / "seed.json").read_text())
    train = job_results(root, "train")
    winner = train[0] if train else None
    frozen_path = root / "winner.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    done_path = root / "research_done.json"
    done = json.loads(done_path.read_text()) if done_path.exists() else None
    run_status = (
        f"search ended ({done['reason']})" if done else
        "search running"
    )
    lines = [
        f"# Frozen-Gemma ledger autoresearch — {_record_path(root).stem[:10]}",
        "",
        "Full trajectories → candidate evidence tools → Gemma ledger → fixed sanitizer → fixed Gemma reader → LDS-v2. No Haiku or weight updates.",
        "",
        f"- Status: {run_status}",
        f"- Search workspace: `{root.resolve()}`",
        f"- Train manifest: `{config['train_manifest']}`",
        f"- Screen manifest: `{config['screen_manifest']}` (24 paired cases)",
        f"- Model: `{config['model']}` at `{config['model_revision']}`",
        f"- Context/output: {config['max_model_len']}/{config['constructor_max_tokens']} tokens",
        f"- Search time budget: {config['wall_hours']} hours from first worker start",
        f"- No-tools seed: `{seed['candidate_id']}`; completed full-train LDS-v2 {seed['full_train_lds_v2_from_completed_baseline']:.4f}",
        f"- Frozen winner: `{frozen['candidate_id']}` ({'seed' if frozen['is_seed'] else 'new candidate'})" if frozen else "- Frozen winner: pending",
        "",
    ]
    if winner:
        candidate_id = winner["candidate_id"]
        train_result = json.loads((Path(winner["result_dir"]) / "reconstruction_metric.json").read_text())
        transfer_path = root / "results" / candidate_id / "transfer" / "reconstruction_metric.json"
        transfer = json.loads(transfer_path.read_text()) if transfer_path.exists() else None
        baseline_transfer_path = BASELINE_RESULT.parent.parent / "heldout/reconstruction_metric.json"
        baseline_transfer = json.loads(baseline_transfer_path.read_text())["score"]
        lines.extend([
            f"- Best experimental full-train candidate: `{candidate_id}`",
            f"- Candidate snapshot: `{root / 'candidates' / (candidate_id + '.json')}`",
            "",
            "| Split | N | LDS-v2 | Δ vs no-tools seed | Balanced accuracy | C1–C4 macro-F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for name, result in (("Train", train_result), ("Calendar transfer", transfer)):
            if result is None:
                continue
            scored = result["scored"]
            baseline_score = seed["full_train_lds_v2_from_completed_baseline"] if name == "Train" else baseline_transfer
            lines.append(
                f"| {name} | {scored['ul_metrics']['total']} | {result['score']:.4f} | "
                f"{result['score'] - baseline_score:+.4f} | "
                f"{scored['discrimination']['balanced_accuracy']:.4f} | "
                f"{scored['attribution']['macro_f1']:.4f} |"
            )
        lines.extend([
            "",
            f"- Train artifacts: `{winner['result_dir']}`",
            f"- Calendar transfer: {'completed' if transfer else 'not run'}",
            "- The calendar set was previously inspected; treat it as descriptive transfer evidence.",
            "",
            "| Split | Constructor errors | Unusable ledgers | Reader errors | Constructor wall time | Constructor input/output tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for name, result in (("Train", train_result), ("Calendar transfer", transfer)):
            if result is None:
                continue
            split_dir = root / "results" / candidate_id / ("train" if name == "Train" else "transfer")
            records = json.loads((split_dir / "records.json").read_text())
            meta = result["meta"]
            constructor_seconds = sum(float(row["ledger_duration_seconds"]) for row in records)
            pass_usages = [usage for row in records for usage in row.get("constructor_pass_usage", [])]
            input_tokens = sum(int(usage.get("prompt_tokens") or 0) for usage in pass_usages)
            output_tokens = sum(int(usage.get("completion_tokens") or 0) for usage in pass_usages)
            lines.append(
                f"| {name} | {meta['constructor_errors']} | {meta['sanitizer_unusable']} | "
                f"{meta['reader_errors']} | {constructor_seconds / 3600:.2f} hours | "
                f"{input_tokens:,}/{output_tokens:,} |"
            )
        lines.extend(["", "### Task breakdown", "", "| Split | Task | N | LDS-v2 |", "| --- | --- | ---: | ---: |"])
        for name, result in (("Train", train_result), ("Calendar transfer", transfer)):
            if result is None:
                continue
            for domain, row in result["scored"]["by_domain"].items():
                score = row["lds_v2"]
                lines.append(f"| {name} | {domain} | {row['n']} | {score:.4f} |" if score is not None else f"| {name} | {domain} | {row['n']} | — |")
        lines.append("")
    record = _record_path(root)
    _replace_text(record, "\n".join(lines))
    return record


def _claim(root: Path, worker_id: str) -> Path | None:
    pending = root / "jobs" / "pending"
    running = root / "jobs" / "running"
    running.mkdir(parents=True, exist_ok=True)
    for path in sorted(pending.glob("*.json")) if pending.exists() else ():
        target = running / path.name
        try:
            path.rename(target)
        except FileNotFoundError:
            continue
        job = json.loads(target.read_text())
        job["worker_id"] = worker_id
        job["claimed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(target, job)
        return target
    return None


def _heartbeat(path: Path, stop: threading.Event) -> None:
    while not stop.wait(30):
        try:
            os.utime(path, None)
        except FileNotFoundError:
            return


def run_worker(
    *, root: Path, base_url: str, worker_id: str, idle_seconds: float = 60.0,
    runtime_id: str = "",
) -> None:
    config = _load_config(root)
    deadline = _deadline(root, config)
    loaded_runtime_hash = _runtime_tree_sha256()
    last_job = time.monotonic()
    while time.time() < deadline:
        if (root / "research_done.json").exists() and not list((root / "jobs" / "pending").glob("*.json")):
            return
        if _runtime_tree_sha256() != loaded_runtime_hash:
            print("Gemma search code/catalog changed; restart this worker before another job", flush=True)
            return
        claimed = _claim(root, worker_id)
        if claimed is None:
            if requeue_stale(root, 180):
                continue
            if idle_seconds > 0 and time.monotonic() - last_job >= idle_seconds:
                return
            time.sleep(5)
            continue
        last_job = time.monotonic()
        job = json.loads(claimed.read_text())
        try:
            stage = job["stage"]
            if stage not in STAGES:
                raise ValueError(f"invalid queued stage: {stage}")
            candidate_id = job["candidate_id"]
            candidate_path = root / "candidates" / f"{candidate_id}.json"
            candidate = CandidateConfig.from_dict(json.loads(candidate_path.read_text()))
            if candidate.execution_hash() != job["candidate_execution_hash"]:
                raise ValueError(f"candidate snapshot changed after enqueue: {candidate_id}")
            if candidate_source_sha256(candidate) != job["candidate_source_sha256"]:
                raise ValueError(f"candidate tool/evaluator code changed after enqueue: {candidate_id}")
        except Exception as exc:
            job.update({
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            _write_json(claimed, job)
            failed = root / "jobs" / "failed" / claimed.name
            failed.parent.mkdir(parents=True, exist_ok=True)
            claimed.rename(failed)
            print(f"{job.get('job_id', claimed.stem)}: failed", flush=True)
            continue
        output_dir = root / "results" / candidate_id / stage
        args = argparse.Namespace(
            candidate=candidate_path,
            manifest=Path(config[f"{stage}_manifest"]),
            data_root=Path(config["data_root"]),
            output_dir=output_dir,
            split=stage,
            allow_transfer=False,
            base_url=base_url,
            model=config["model"],
            model_revision=config["model_revision"],
            runtime_id=runtime_id or worker_id,
            max_model_len=config["max_model_len"],
            constructor_max_tokens=config["constructor_max_tokens"],
            reader_max_tokens=config["reader_max_tokens"],
            temperature=config["temperature"],
            seed=config["seed"],
            timeout_seconds=900.0,
            resume=True,
            preflight_only=False,
        )
        stop = threading.Event()
        heartbeat = threading.Thread(target=_heartbeat, args=(claimed, stop), daemon=True)
        heartbeat.start()
        try:
            result = evaluate(args)
            job.update({
                "status": "done", "score": result["score"],
                "result_dir": str(output_dir),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            destination = root / "jobs" / "done" / claimed.name
        except Exception as exc:
            job.update({
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            destination = root / "jobs" / "failed" / claimed.name
        finally:
            stop.set()
            heartbeat.join(timeout=2)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json(claimed, job)
        claimed.rename(destination)
        print(f"{job['job_id']}: {job['status']}", flush=True)


def job_results(root: Path, stage: str) -> list[dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"invalid stage: {stage}")
    results = []
    for path in (root / "jobs" / "done").glob(f"*_{stage}.json"):
        job = json.loads(path.read_text())
        results.append(job)
    return sorted(results, key=lambda row: (-float(row["score"]), row["candidate_id"]))


def promote(root: Path, top_k: int) -> list[str]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    selected = []
    for row in job_results(root, "screen")[:top_k]:
        candidate_id = row["candidate_id"]
        enqueue(root, candidate_id, "train")
        selected.append(candidate_id)
    return selected


def status(root: Path) -> dict[str, Any]:
    counts = {
        state: len(list((root / "jobs" / state).glob("*.json")))
        for state in ("pending", "running", "done", "failed")
    }
    return {
        "jobs": counts,
        "best_screen": job_results(root, "screen")[:3],
        "best_train": job_results(root, "train")[:3],
    }


def requeue_stale(root: Path, stale_seconds: float) -> list[str]:
    if stale_seconds < 60:
        raise ValueError("stale_seconds must be at least 60")
    moved = []
    now = time.time()
    for path in (root / "jobs" / "running").glob("*.json"):
        if now - path.stat().st_mtime < stale_seconds:
            continue
        destination = root / "jobs" / "pending" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError(f"cannot requeue: pending job already exists: {destination}")
        path.rename(destination)
        moved.append(path.stem)
    return moved


def retry_failed(root: Path, job_id: str) -> Path:
    """Explicitly retry one failed job from its successful case checkpoints."""
    if not job_id or "/" in job_id or ".." in job_id:
        raise ValueError("job_id must be a single queue job name")
    failed = root / "jobs" / "failed" / f"{job_id}.json"
    if not failed.exists():
        raise ValueError(f"failed job does not exist: {failed}")
    destination = root / "jobs" / "pending" / failed.name
    if destination.exists() or (root / "jobs" / "running" / failed.name).exists() or (root / "jobs" / "done" / failed.name).exists():
        raise ValueError(f"job already has an active or completed copy: {job_id}")
    job = json.loads(failed.read_text())
    for key in ("status", "error", "finished_at", "claimed_at", "worker_id"):
        job.pop(key, None)
    job["retried_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(failed, job)
    failed.rename(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--data-root", type=Path, required=True)
    init.add_argument("--train-manifest", type=Path, required=True)
    init.add_argument("--model-revision", required=True)
    init.add_argument("--wall-hours", type=float, default=48.0)
    init.add_argument("--screen-seed", type=int, default=20260923)
    add = sub.add_parser("add")
    add.add_argument("--root", type=Path, required=True)
    add.add_argument("--candidate", type=Path, required=True)
    add.add_argument("--stage", choices=STAGES, default="screen")
    work = sub.add_parser("worker")
    work.add_argument("--root", type=Path, required=True)
    work.add_argument("--base-url", default=OPENAI_BASE_URL)
    work.add_argument("--worker-id", required=True)
    work.add_argument("--runtime-id", default="")
    work.add_argument("--idle-seconds", type=float, default=0.0)
    rank = sub.add_parser("promote")
    rank.add_argument("--root", type=Path, required=True)
    rank.add_argument("--top-k", type=int, default=2)
    show = sub.add_parser("status")
    show.add_argument("--root", type=Path, required=True)
    stale = sub.add_parser("requeue-stale")
    stale.add_argument("--root", type=Path, required=True)
    stale.add_argument("--stale-seconds", type=float, default=180.0)
    retry = sub.add_parser("retry-failed")
    retry.add_argument("--root", type=Path, required=True)
    retry.add_argument("--job-id", required=True)
    report = sub.add_parser("report")
    report.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        result = initialize(
            root=args.root, data_root=args.data_root, train_manifest=args.train_manifest,
            model_revision=args.model_revision, wall_hours=args.wall_hours,
            screen_seed=args.screen_seed,
        )
    elif args.command == "add":
        candidate = CandidateConfig.from_dict(json.loads(args.candidate.read_text()))
        store_candidate(args.root, candidate)
        result = {"job": str(enqueue(args.root, candidate.candidate_id, args.stage))}
    elif args.command == "worker":
        run_worker(
            root=args.root, base_url=args.base_url, worker_id=args.worker_id,
            runtime_id=args.runtime_id, idle_seconds=args.idle_seconds,
        )
        return
    elif args.command == "promote":
        result = {"promoted": promote(args.root, args.top_k)}
    elif args.command == "status":
        result = status(args.root)
    elif args.command == "report":
        result = {"record": str(write_report(args.root))}
    elif args.command == "retry-failed":
        result = {"pending_job": str(retry_failed(args.root, args.job_id))}
    else:
        result = {"requeued": requeue_stale(args.root, args.stale_seconds)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
