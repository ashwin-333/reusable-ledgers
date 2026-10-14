"""Primary monitor: Gemma-4-12B-it writes and reads.

  a12  untrained 12B writes a ledger, then the same 12B base reads it
  b12  untrained 12B reads the raw trajectory and predicts directly
  c12  12B plus the ledger adapter writes; the reader request is the 12B
       base id, with no adapter applied
  d    12B plus the direct-SFT adapter predicts from the raw trajectory

Calendar inputs match systems (a), (b), and (c). CUA uses the monitor-facing
trajectory_text, never the raw conversation. CUA is not scored with LDS-v2.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from config import OPENAI_BASE_URL, WRITER_MODEL

HERE = Path(__file__).resolve().parent

from autoresearch import gemma_baseline as gb
from autoresearch.ledger_score import ledger_record
from autoresearch.reconstruction_metric import (
    GEMMA_DECODER_SYSTEM_PROMPT,
    GEMMA_RESPONSE_FORMAT,
    SANITIZER_VERSION,
    ReconstructionMetricError,
    parse_gemma_prediction,
)
from ledgers.run_eval import (
    DIRECT_PROMPT_FILE,
    LEDGER_PROMPT_FILE,
    _adapter_sha256,
    _bundle_sha256,
    _compact_user_prompt,
    _constructor_record,
    _finalize,
    _json_user_prompt,
    _sha256,
)

MODEL = WRITER_MODEL
# Later ledger adapters use the c12 split: adapter writes, base 12B reads.
LEDGER_SYSTEMS = ("c12", "ledger_C", "ledger_seed2", "ledger_C_seed2", "ledger_D")
# Later rationale adapters use the rationale path: the adapter is the reader.
RATIONALE_SYSTEMS = ("rationale", "rationale_seed2")
DIRECT_SYSTEMS = ("b12", "d", *RATIONALE_SYSTEMS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("a12", "b12", "c12", "d", "rationale", "ledger_C", "ledger_seed2", "ledger_C_seed2", "ledger_D", "rationale_seed2"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL)
    parser.add_argument("--server-flags", required=True)
    parser.add_argument("--runtime-id", default="")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--adapter-name", default=None)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--input", choices=("json", "trajectory-text"), default="json")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=170000)
    parser.add_argument("--constructor-max-tokens", type=int, default=8192)
    parser.add_argument("--reader-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    needs_adapter = args.system in (*LEDGER_SYSTEMS, "d", *RATIONALE_SYSTEMS)
    if needs_adapter and not (args.adapter_name and args.adapter_dir):
        parser.error(f"system {args.system} needs --adapter-name and --adapter-dir")
    if not needs_adapter and (args.adapter_name or args.adapter_dir):
        parser.error("a12 and b12 do not load an adapter")
    if args.adapter_name == MODEL:
        parser.error("adapter name must differ from the 12B base id")
    return args


def _served_ids(base_url: str) -> list[str]:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as response:
        body = json.loads(response.read())
    return [row["id"] for row in body["data"]]


def _probe(args: argparse.Namespace, model: str) -> None:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with ok"}],
        "temperature": 0,
        "max_tokens": 4,
        "seed": 0,
    }).encode()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read())
    if body.get("model") != model:
        raise SystemExit(f"probe for {model} was answered as {body.get('model')!r}")


def _assert_server(args: argparse.Namespace) -> None:
    ids = _served_ids(args.base_url)
    if args.system in ("a12", "b12"):
        if ids != [MODEL]:
            raise SystemExit(f"{args.system} must serve only {MODEL}, got {ids}")
        _probe(args, MODEL)
        return
    if args.system in LEDGER_SYSTEMS:
        if MODEL not in ids or args.adapter_name not in ids:
            raise SystemExit(f"{args.system} server ids {ids} must include {MODEL} and {args.adapter_name}")
        _probe(args, MODEL)
        return
    if args.adapter_name not in ids:
        raise SystemExit(f"direct adapter {args.adapter_name} is not served; ids={ids}")
    _probe(args, args.adapter_name)


def _trajectory_text(entry: dict, data_root: Path) -> str:
    document = json.loads((data_root / entry["path"]).read_text())
    text = document.get("trajectory_text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"missing trajectory_text: {entry['path']}")
    if "<thinking>" in text or "</thinking>" in text:
        raise RuntimeError(f"hidden reasoning remains in trajectory_text: {entry['path']}")
    return text


def _user(args: argparse.Namespace, entry: dict) -> str:
    if args.input == "trajectory-text":
        text = _trajectory_text(entry, args.data_root)
        if args.system in DIRECT_SYSTEMS:
            return text
        return gb._user_prompt(json.dumps({"trajectory_text": text}, ensure_ascii=False))
    if args.system in LEDGER_SYSTEMS:
        return _compact_user_prompt(entry, args.data_root)
    return _json_user_prompt(entry, args.data_root)


def _direct_request(args: argparse.Namespace, model: str, system: str, user: str) -> dict[str, Any]:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": args.temperature,
        "max_tokens": args.reader_max_tokens,
        "seed": args.seed,
        "response_format": GEMMA_RESPONSE_FORMAT,
    }).encode()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Gemma HTTP {error.code}: {error.read().decode(errors='replace')[:500]}") from error
    if body.get("model") != model:
        raise RuntimeError(f"direct reader answered as {body.get('model')!r}, not {model}")
    choice = body["choices"][0]
    return {"text": choice["message"]["content"], "usage": body.get("usage") or {},
            "finish_reason": choice.get("finish_reason")}


def _direct_record(args, entry, prompt, user, config_sha256, index, model) -> tuple[dict, dict]:
    checkpoint = args.output_dir / "predictions" / f"{index:03d}.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("config_sha256") != config_sha256 or saved.get("path") != entry["path"]:
            raise ValueError(f"direct checkpoint has a different configuration: {checkpoint}")
        return saved["grader"], saved["record"]
    started = time.monotonic()
    last_error: Exception | None = None
    response: dict[str, Any] = {"text": None, "usage": {}, "finish_reason": None}
    prediction = None
    attempts = 0
    for attempts in range(1, 4):
        try:
            response = _direct_request(args, model, prompt, user)
            prediction = parse_gemma_prediction(response["text"])
            break
        except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError, ReconstructionMetricError) as exc:
            last_error = exc
            if attempts < 3:
                time.sleep(2 ** (attempts - 1))
    if prediction is None:
        verdict, violations = "UNKNOWN", []
        error = f"{type(last_error).__name__}: {last_error}"
    else:
        verdict, violations, error = prediction.verdict, list(prediction.violations_found), None
    record = {
        "path": entry["path"], "prediction": {"verdict": verdict, "violations_found": violations},
        "raw_response": response["text"], "usage": response["usage"],
        "finish_reason": response["finish_reason"], "error": error, "attempts": attempts,
        "duration_seconds": round(time.monotonic() - started, 3), "model": model,
    }
    grader = ledger_record(entry, {"final_verdict": verdict, "violations_found": violations,
                                   "ledger_text": "", "ledger_usage": {}})
    gb._write_json(checkpoint, {"config_sha256": config_sha256, "path": entry["path"],
                                "record": record, "grader": grader})
    return grader, record


def _config(args: argparse.Namespace, entries: list[dict]) -> dict[str, Any]:
    direct = args.system in DIRECT_SYSTEMS
    prompt_file = DIRECT_PROMPT_FILE if direct else LEDGER_PROMPT_FILE
    if args.input == "json":
        if args.system in LEDGER_SYSTEMS:
            input_format = "sft-input-trajectory-compact-v1"
        else:
            input_format = "complete-conversation-json-without-dataset-metadata-v1"
    elif direct:
        input_format = "mrt-monitor-facing-trajectory-text"
    else:
        input_format = "mrt-monitor-facing-trajectory-text-as-json"
    return {
        "system": args.system,
        "manifest_sha256": _sha256(args.manifest.read_bytes()),
        "trajectory_bundle_sha256": _bundle_sha256(entries, args.data_root),
        "n_manifest": len(entries),
        "input_format": input_format,
        "model": MODEL,
        "model_revision": args.model_revision,
        "constructor_model": None if direct else (args.adapter_name or MODEL),
        "reader_model": args.adapter_name if args.system in ("d", *RATIONALE_SYSTEMS) else MODEL,
        "adapter_sha256": _adapter_sha256(args.adapter_dir) if args.adapter_dir else None,
        "server_flags": args.server_flags,
        "max_model_len": args.max_model_len,
        "constructor_max_tokens": args.constructor_max_tokens,
        "reader_max_tokens": args.reader_max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "prompt_sha256": _sha256(prompt_file.read_bytes()),
        "reader_prompt_sha256": None if direct else _sha256(GEMMA_DECODER_SYSTEM_PROMPT.encode()),
        "sanitizer_version": None if direct else SANITIZER_VERSION,
        "runner_code_sha256": _sha256(Path(__file__).read_bytes()),
        "unknown_policy": "pass",
    }


def main() -> None:
    args = _parse_args()
    entries = gb._load_manifest(args.manifest)
    if args.input == "trajectory-text" and len(entries) != 159:
        raise SystemExit(f"CUA manifest has {len(entries)} rows, expected 159")
    _assert_server(args)
    config = _config(args, entries)
    config_sha256 = _sha256(json.dumps(config, sort_keys=True).encode())
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old.get("config_sha256") != config_sha256:
            raise SystemExit(f"output directory has a different configuration: {args.output_dir}")
    else:
        gb._write_json(config_path, {"config_sha256": config_sha256, "config": config})

    mine = [(i, e) for i, e in enumerate(entries, 1) if (i - 1) % args.num_chunks == args.chunk]
    if args.limit is not None:
        mine = mine[: args.limit]
    direct = args.system in DIRECT_SYSTEMS
    prompt = (DIRECT_PROMPT_FILE if direct else LEDGER_PROMPT_FILE).read_text()
    if args.system in (*LEDGER_SYSTEMS, "d", *RATIONALE_SYSTEMS):
        request_model = args.adapter_name
    else:
        request_model = MODEL
    common = dict(base_url=args.base_url, temperature=args.temperature, seed=args.seed,
                  timeout_seconds=args.timeout_seconds, output_dir=args.output_dir, resume=True,
                  data_root=args.data_root)
    cargs = SimpleNamespace(model=request_model, constructor_max_tokens=args.constructor_max_tokens, **common)
    rargs = SimpleNamespace(model=MODEL, reader_max_tokens=args.reader_max_tokens, **common)
    if not direct and rargs.model != MODEL:
        raise SystemExit("12B reader must request the base id")
    if args.system in LEDGER_SYSTEMS and rargs.model == args.adapter_name:
        raise SystemExit(f"{args.system} reader must not request the adapter")

    def one(index: int, entry: dict) -> str:
        user = _user(args, entry)
        if direct:
            _, record = _direct_record(args, entry, prompt, user, config_sha256, index, request_model)
            return f"direct {record['prediction']['verdict']}"
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
    if args.input == "json":
        fin = SimpleNamespace(
            system="b" if direct else "a",
            output_dir=args.output_dir,
            runtime_id=args.runtime_id,
        )
        _finalize(fin, entries, config_sha256)


if __name__ == "__main__":
    main()
