"""Run one evaluation system on one manifest chunk against a running vLLM server.

Systems:
  a  base Gemma writes a ledger (gemma_factual_ledger_baseline.txt, JSON input)
     -> neutral-ledger-v3 sanitizer -> base Gemma reader
  c  Gemma + LoRA adapter writes a ledger (same prompt, SFT input format)
     -> same sanitizer -> base Gemma reader (never the adapter)

Checkpoints are one file per trajectory, keyed by the 1-based index in the full
manifest, so chunks can run as separate jobs and any job can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from config import OPENAI_BASE_URL, READER_MODEL_LARGE, REPO_ROOT, model_revision

HERE = Path(__file__).resolve().parent

from autoresearch import gemma_baseline as gb
from autoresearch.ledger_score import score_ledgers
from autoresearch.reconstruction_metric import (
    GEMMA_DECODER_SYSTEM_PROMPT,
    SANITIZER_VERSION,
)

BASE_MODEL = READER_MODEL_LARGE
MODEL_REVISION = model_revision(BASE_MODEL)
LEDGER_PROMPT_FILE = HERE / "prompts" / "gemma_factual_ledger_baseline.txt"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compact_user_prompt(entry: dict[str, Any], data_root: Path) -> str:
    """The SFT training input: '### INPUT TRAJECTORY' + render_turns_compact."""
    from ledgers.render import load_trace_auto, render_turns_compact
    from ledgers.prompt_multipart import build_cached_system_and_user

    trace = load_trace_auto(data_root / entry["path"])
    turns = [t for t in trace.turns if getattr(t, "role", "").lower() != "system"]
    _, user = build_cached_system_and_user(
        "{trajectory}", "{trajectory}", render_turns_compact(turns), dynamic_header="INPUT TRAJECTORY"
    )
    return user


def _json_user_prompt(entry: dict[str, Any], data_root: Path) -> str:
    trajectory_json, _ = gb._trajectory(entry, data_root)
    return gb._user_prompt(trajectory_json)


def _bundle_sha256(entries: list[dict[str, Any]], data_root: Path) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode())
        digest.update(b"\0")
        digest.update((data_root / entry["path"]).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _adapter_sha256(adapter_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(adapter_dir.iterdir()):
        if path.is_file() and path.name.startswith("adapter_"):
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _constructor_record(
    args: argparse.Namespace, cargs: SimpleNamespace, entry: dict, prompt: str, user: str,
    config_sha256: str, index: int,
) -> dict[str, Any]:
    """Same record schema and retry policy as gemma_baseline._constructor_record."""
    checkpoint = args.output_dir / "constructor_records" / f"{index:03d}.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("config_sha256") != config_sha256 or saved.get("path") != entry["path"]:
            raise ValueError(f"constructor checkpoint has a different configuration: {checkpoint}")
        return saved["record"]
    started = time.monotonic()
    response: dict[str, Any] = {"text": "", "usage": {}, "finish_reason": None}
    error: str | None = None
    try:
        import asyncio

        response = asyncio.run(gb._retry_complete(gb.PlainGemmaClient(cargs), prompt, user))
        if response["finish_reason"] == "length":
            error = "Gemma ledger reached the output token limit"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    record = {
        **entry,
        "ledger_text": response["text"] if error is None else "",
        "partial_ledger_text": response["text"] if error is not None else None,
        "ledger_usage": response["usage"],
        "ledger_model": cargs.model,
        "ledger_error": error,
        "ledger_duration_seconds": round(time.monotonic() - started, 3),
    }
    gb._write_json(checkpoint, {"config_sha256": config_sha256, "path": entry["path"], "record": record})
    ledger_path = args.output_dir / "ledgers" / f"{index:03d}.txt"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(response["text"])
    return record


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", choices=("a", "c"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-url", default=OPENAI_BASE_URL)
    p.add_argument("--server-flags", required=True, help="exact vLLM flags, recorded in the config hash")
    p.add_argument("--runtime-id", default="")
    p.add_argument("--adapter-name", default=None, help="served LoRA name (system c only)")
    p.add_argument("--adapter-dir", type=Path, default=None)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--num-chunks", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="first N chunk entries only (plumbing tests)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=170000)
    p.add_argument("--constructor-max-tokens", type=int, default=8192)
    p.add_argument("--reader-max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = p.parse_args()
    if args.system == "c" and not (args.adapter_name and args.adapter_dir):
        p.error("system c needs --adapter-name and --adapter-dir")
    if args.system != "c" and (args.adapter_name or args.adapter_dir):
        p.error("only system c may use an adapter")
    if args.adapter_name == BASE_MODEL:
        p.error("adapter name must differ from the base model name")
    return args


def _config(args: argparse.Namespace, entries: list[dict]) -> dict[str, Any]:
    prompt_file = LEDGER_PROMPT_FILE
    config = {
        "system": args.system,
        "manifest_sha256": _sha256(args.manifest.read_bytes()),
        "trajectory_bundle_sha256": _bundle_sha256(entries, args.data_root),
        "n_manifest": len(entries),
        "input_format": {
            "a": "complete-conversation-json-without-dataset-metadata-v1",
            "c": "sft-input-trajectory-compact-v1",
        }[args.system],
        "model": BASE_MODEL,
        "model_revision": MODEL_REVISION,
        "constructor_model": args.adapter_name or BASE_MODEL,
        "reader_model": BASE_MODEL,
        "adapter_sha256": _adapter_sha256(args.adapter_dir) if args.adapter_dir else None,
        "server_flags": args.server_flags,
        "max_model_len": args.max_model_len,
        "constructor_max_tokens": args.constructor_max_tokens,
        "reader_max_tokens": args.reader_max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "prompt_file": str(prompt_file.relative_to(REPO_ROOT)),
        "prompt_sha256": _sha256(prompt_file.read_bytes()),
        "reader_prompt_sha256": _sha256(GEMMA_DECODER_SYSTEM_PROMPT.encode()),
        "sanitizer_version": SANITIZER_VERSION,
        "runner_code_sha256": _sha256(Path(__file__).read_bytes()),
        "gemma_baseline_code_sha256": _sha256(Path(gb.__file__).read_bytes()),
        "metric_code_sha256": _sha256((REPO_ROOT / "autoresearch" / "reconstruction_metric.py").read_bytes()),
        "scorer_code_sha256": _sha256((REPO_ROOT / "autoresearch" / "lds_v2.py").read_bytes()),
        "unknown_policy": "pass",
    }
    return config


def _finalize(args: argparse.Namespace, entries: list[dict], config_sha256: str) -> bool:
    """Write merged artifacts once every trajectory in the manifest has a checkpoint."""
    graders, audits, records = [], [], []
    for index, entry in enumerate(entries, 1):
        path = args.output_dir / "reader_records" / f"{index:03d}.json"
        if not path.exists():
            return False
        saved = json.loads(path.read_text())
        if saved.get("config_sha256") != config_sha256 or saved.get("path") != entry["path"]:
            raise ValueError(f"checkpoint has a different configuration: {path}")
        graders.append(saved["grader"])
        audits.append(saved["audit"])
        constructor = json.loads((args.output_dir / "constructor_records" / f"{index:03d}.json").read_text())
        records.append(constructor["record"])
    scored = score_ledgers(graders, unknown_policy="pass")
    gb._write_json(args.output_dir / "grader_rows_private.json", graders)
    gb._write_json(args.output_dir / "records.json", records)
    gb._write_json(args.output_dir / "gemma_predictions_private.json", audits)
    meta = {"n": len(entries), "config_sha256": config_sha256, "runtime_id": args.runtime_id}
    meta["constructor_errors"] = sum(bool(r["ledger_error"]) for r in records)
    meta["sanitizer_unusable"] = sum(a.get("sanitizer_status") == "unusable" for a in audits)
    meta["reader_errors"] = sum("reader_error" in a for a in audits)
    gb._write_json(args.output_dir / "lds_v2.json", {"scored": scored, "meta": meta})
    print(f"system {args.system} complete: LDS-v2 {scored['lds_v2']:.4f} (n={len(entries)})", flush=True)
    return True


def main() -> None:
    args = _parse_args()
    entries = gb._load_manifest(args.manifest)
    config = _config(args, entries)
    config_sha256 = _sha256(json.dumps(config, sort_keys=True).encode())
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old.get("config_sha256") != config_sha256:
            raise ValueError(f"output directory has a different configuration: {args.output_dir}")
    else:
        gb._write_json(config_path, {"config_sha256": config_sha256, "config": config})

    mine = [(i, e) for i, e in enumerate(entries, 1) if (i - 1) % args.num_chunks == args.chunk]
    if args.limit is not None:
        mine = mine[: args.limit]
    prompt = LEDGER_PROMPT_FILE.read_text()
    common = dict(base_url=args.base_url, temperature=args.temperature, seed=args.seed,
                  timeout_seconds=args.timeout_seconds, output_dir=args.output_dir, resume=True,
                  data_root=args.data_root)
    cargs = SimpleNamespace(model=args.adapter_name or BASE_MODEL,
                            constructor_max_tokens=args.constructor_max_tokens, **common)
    rargs = SimpleNamespace(model=BASE_MODEL, reader_max_tokens=args.reader_max_tokens, **common)
    assert rargs.model == BASE_MODEL and rargs.model != args.adapter_name

    def one(index: int, entry: dict) -> str:
        user = (_compact_user_prompt if args.system == "c" else _json_user_prompt)(entry, args.data_root)
        record = _constructor_record(args, cargs, entry, prompt, user, config_sha256, index)
        _, audit = gb._reader_record(rargs, record, config_sha256, index)
        status = "ERROR" if record["ledger_error"] else "OK"
        return f"constructor {status} reader {audit['prediction']['verdict']}"

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, i, e): (i, e) for i, e in mine}
        for done, future in enumerate(as_completed(futures), 1):
            index, entry = futures[future]
            print(f"[{done}/{len(mine)}] #{index:03d} {future.result()}: {entry['path']} "
                  f"({time.monotonic() - started:.0f}s)", flush=True)
    _finalize(args, entries, config_sha256)


if __name__ == "__main__":
    main()
