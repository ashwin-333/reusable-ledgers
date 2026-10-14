"""E5 on the real multi-policy set. Widths 1 and 10 only.

Same frozen reader wrapper as E5. Policy text is read from the benchmark
DATA built and is not edited. Ledger readers use the sanitized SHADE
ledgers that already exist. b12 reads the SHADE monitor-facing trajectory. b is the same read with
Gemma-4-31B-it. The average ensemble is scored later, not called as its own model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from config import DATA_ROOT, OPENAI_BASE_URL, READER_MODEL, RUNS_DIR, STATUS_DIR

HERE = Path(__file__).resolve().parent

from autoresearch import gemma_baseline as gb
from autoresearch.reconstruction_metric import ReconstructionMetricError, sanitize_ledger
from ledgers.read_policy import _one_width, _sha
from ledgers.shade_run import _trajectory_text

WIDTHS = (1, 10)
SFT = DATA_ROOT
RUNS = RUNS_DIR
SHADE_DATA = SFT / "shade_real"
LEDGER_SYSTEMS = ("a12", "ledger_C", "ledger_C_seed2")
SYSTEMS = LEDGER_SYSTEMS + ("b12", "b")


def locate_bench(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit if (explicit / "policies.json").is_file() else None
    candidates = []
    if (SFT / "READY_multi_policy_real").is_file():
        candidates.append(SFT / "multi_policy_real")
    inside = SFT / "multi_policy_real"
    if (inside / "READY_multi_policy_real").is_file():
        candidates.append(inside)
    for path in candidates:
        if (path / "policies.json").is_file():
            return path
    return None


def _load(bench: Path, hash_file: Path, allow_unfrozen: bool) -> tuple[str, str, list[dict], list[dict], str]:
    system_path = HERE / "prompts" / "e5_reader_system.txt"
    format_path = HERE / "prompts" / "e5_user_format.txt"
    system_sha = (HERE / "prompts" / "e5_reader_system.sha256").read_text().strip()
    format_sha = (HERE / "prompts" / "e5_user_format.sha256").read_text().strip()
    if _sha(system_path) != system_sha or _sha(format_path) != format_sha:
        raise SystemExit("E5 reader wrapper was edited; refusing to run")
    policy_path = bench / "policies.json"
    policy_sha = _sha(policy_path)
    if hash_file.is_file():
        frozen = hash_file.read_text().strip()
        if policy_sha != frozen:
            raise SystemExit("policies.json changed after it was frozen; refusing to run")
    elif not allow_unfrozen:
        raise SystemExit(f"policy hash is not frozen yet: {hash_file}")
    policies = json.loads(policy_path.read_text())["policies"]
    bench_file = bench / "multi_policy_benchmark.json"
    if not bench_file.is_file():
        raise SystemExit(f"missing {bench_file}")
    payload = json.loads(bench_file.read_text())
    rows = payload["trajectories"]
    if payload.get("policy_sha256") not in (None, policy_sha):
        raise SystemExit("benchmark policy_sha256 does not match policies.json")
    return system_path.read_text(), format_path.read_text(), policies, rows, policy_sha


def _shade_ledgers(system: str) -> dict[str, str]:
    records = RUNS / "shade" / system / "constructor_records"
    if not records.is_dir():
        raise SystemExit(f"missing SHADE ledgers: {records}")
    found = {}
    for path in records.glob("*.json"):
        saved = json.loads(path.read_text())
        record = saved.get("record") or {}
        raw = record.get("ledger_text") or ""
        text = ""
        if not record.get("ledger_error") and raw.strip():
            try:
                text = sanitize_ledger(raw)
            except ReconstructionMetricError:
                text = ""
        found[saved["path"]] = text
    return found


def _data_root(bench: Path, rows: list[dict]) -> Path:
    payload = json.loads((bench / "multi_policy_benchmark.json").read_text())
    stated = payload.get("data_root")
    if stated:
        return Path(stated)
    return SHADE_DATA


def _body(system: str, row: dict, ledgers: dict[str, str], data_root: Path) -> tuple[str, str | None]:
    if system in ("b12", "b"):
        try:
            return _trajectory_text(row, data_root), None
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            return "", f"{type(exc).__name__}: {exc}"
    if row["path"] not in ledgers:
        return "", "missing ledger"
    text = ledgers[row["path"]]
    if not text.strip():
        return "", "empty ledger"
    return text, None


def _check(bench: Path, hash_file: Path, allow_unfrozen: bool) -> None:
    _, _, policies, rows, policy_sha = _load(bench, hash_file, allow_unfrozen)
    data_root = _data_root(bench, rows)
    missing = []
    for system in LEDGER_SYSTEMS:
        ledgers = _shade_ledgers(system)
        for row in rows:
            if row["path"] not in ledgers:
                missing.append(f"{system} {row['path']}")
    absent = [row["path"] for row in rows if not (data_root / row["path"]).is_file()]
    if missing or absent:
        raise SystemExit(
            f"not ready: {len(missing)} ledger gaps, {len(absent)} missing trajectories, "
            f"first ledger {missing[:1]} first file {absent[:1]}"
        )
    print(json.dumps({
        "bench": str(bench),
        "policies": len(policies),
        "trajectories": len(rows),
        "policy_sha256": policy_sha,
        "data_root": str(data_root),
    }), flush=True)


def _finalize(output_dir: Path, rows: list[dict], system_sha: str) -> None:
    saved_rows = []
    for index, row in enumerate(rows, 1):
        path = output_dir / "predictions" / f"{index:03d}.json"
        if not path.is_file():
            return
        saved = json.loads(path.read_text())
        if saved.get("system_sha256") != system_sha or saved.get("path") != row["path"]:
            raise SystemExit(f"checkpoint mismatch: {path}")
        saved_rows.append(saved)
    tokens = {}
    for width in WIDTHS:
        if any(str(width) not in saved["widths"] for saved in saved_rows):
            continue
        per_traj = []
        for saved in saved_rows:
            calls = saved["widths"][str(width)]["prompt_tokens"]
            per_traj.append(sum(calls))
        tokens[str(width)] = {
            "mean_prompt_tokens_per_trajectory": (sum(per_traj) / len(per_traj)) if per_traj else 0,
            "n_trajectories": len(per_traj),
        }
    gb._write_json(output_dir / "e5_tokens.json", tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--hash-file", type=Path, default=STATUS_DIR / "e5_real_policy.sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL)
    parser.add_argument("--served-model", default=READER_MODEL)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--widths", default="1,10")
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--allow-unfrozen", action="store_true")
    args = parser.parse_args()
    bench = locate_bench(args.bench)
    if bench is None:
        raise SystemExit("READY_multi_policy_real is not available")
    if args.check:
        _check(bench, args.hash_file, args.allow_unfrozen)
        return
    requested = tuple(int(part) for part in args.widths.split(",") if part)
    if not requested or any(width not in WIDTHS for width in requested):
        raise SystemExit(f"widths must be a subset of {WIDTHS}")
    need = {str(width) for width in requested}
    system_prompt, template, policies, rows, policy_sha = _load(bench, args.hash_file, False)
    system_sha = hashlib.sha256(system_prompt.encode()).hexdigest()
    data_root = _data_root(bench, rows)
    ledgers = {} if args.system in ("b12", "b") else _shade_ledgers(args.system)

    def one(index: int, row: dict) -> str:
        checkpoint = args.output_dir / "predictions" / f"{index:03d}.json"
        saved = None
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("system_sha256") != system_sha or saved.get("policy_sha256") != policy_sha or saved.get("path") != row["path"]:
                raise SystemExit(f"checkpoint mismatch: {checkpoint}")
            missing = need - set((saved.get("widths") or {}))
            if not missing:
                return "resume"
        else:
            missing = set(need)
        body, skip = _body(args.system, row, ledgers, data_root)
        widths = dict((saved or {}).get("widths") or {})
        if skip or not body:
            for width in missing:
                widths[width] = {
                    "decisions": {item["id"]: "unknown" for item in policies},
                    "prompt_tokens": [],
                    "completion_tokens": [],
                    "error": skip or "empty record",
                }
        else:
            for width in sorted(missing, key=int):
                widths[width] = _one_width(
                    args.base_url, args.served_model, system_prompt, template,
                    policies, body, int(width), args.timeout_seconds,
                )
        gb._write_json(checkpoint, {
            "system_sha256": system_sha,
            "policy_sha256": policy_sha,
            "path": row["path"],
            "set": row.get("set"),
            "widths": widths,
        })
        return "ok"

    keep = None
    if args.subset:
        payload = json.loads(args.subset.read_text())
        keep = set(payload["paths"] if isinstance(payload, dict) else payload)
    selected = [(i, row) for i, row in enumerate(rows, 1) if keep is None or row["path"] in keep]
    if keep is not None and len(selected) != len(keep):
        raise SystemExit(f"subset has {len(keep)} paths but {len(selected)} match the benchmark")
    mine = [item for n, item in enumerate(selected) if n % args.num_chunks == args.chunk]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, i, row): i for i, row in mine}
        for done, future in enumerate(as_completed(futures), 1):
            print(f"[{done}/{len(mine)}] #{futures[future]:03d} {future.result()}", flush=True)
    _finalize(args.output_dir, rows, system_sha)


if __name__ == "__main__":
    main()
