"""Frozen Gemma baseline: full trajectory -> factual ledger -> Gemma reader.

This runner never calls a teacher model. The only model endpoint is the pinned
Gemma checkpoint, used for two separate inference calls per trajectory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


from .ledger_score import ledger_record, score_ledgers
from .reconstruction_metric import (
    GEMMA_DECODER_SYSTEM_PROMPT,
    RECONSTRUCTION_METRIC_NAME,
    SANITIZER_VERSION,
    VLLMGemmaClient,
    decode_ledgers,
)
from config import OPENAI_BASE_URL, READER_MODEL_LARGE, REPO_ROOT


DEFAULT_MODEL = READER_MODEL_LARGE
DEFAULT_PROMPT = REPO_ROOT / "ledgers" / "prompts" / "gemma_factual_ledger_baseline.txt"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"manifest must contain a nonempty list: {path}")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"manifest contains an entry without a path: {path}")
        if entry["path"] in seen:
            raise ValueError(f"manifest repeats a trajectory: {entry['path']}")
        seen.add(entry["path"])
    return entries


def _trajectory(entry: dict[str, Any], data_root: Path) -> tuple[str, bytes]:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe trajectory path: {relative}")
    source = data_root / relative
    raw = source.read_bytes()
    document = json.loads(raw)
    conversation = document.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError(f"trajectory has no complete conversation array: {source}")
    if not all(isinstance(turn, dict) for turn in conversation):
        raise ValueError(f"trajectory has a malformed conversation turn: {source}")
    # Preserve every conversation turn and tool-call field without truncation.
    # metadata.entity is a dataset annotation, not an agent-visible message.
    # Neither it nor the manifest labels should help the constructor.
    visible_conversation = [
        {key: value for key, value in turn.items() if key != "metadata"}
        for turn in conversation
    ]
    text = json.dumps({"conversation": visible_conversation}, ensure_ascii=False, separators=(",", ":"))
    return text, raw


def _user_prompt(trajectory_json: str) -> str:
    return "[AGENT_TRAJECTORY_JSON]\n" + trajectory_json + "\n[/AGENT_TRAJECTORY_JSON]"


def _preflight(args: argparse.Namespace, entries: list[dict[str, Any]], prompt: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=True,
        trust_remote_code=True,
    )
    max_input_tokens = args.max_model_len - args.constructor_max_tokens
    if max_input_tokens <= 0:
        raise ValueError("constructor output reserve must be smaller than model context")
    bundle_hash = hashlib.sha256()
    longest: list[dict[str, Any]] = []
    over_limit: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, 1):
        trajectory_json, raw = _trajectory(entry, args.data_root)
        bundle_hash.update(entry["path"].encode())
        bundle_hash.update(b"\0")
        bundle_hash.update(raw)
        bundle_hash.update(b"\0")
        encoded = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": _user_prompt(trajectory_json)},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        tokens = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if not isinstance(tokens, (list, tuple)) or len(tokens) < 16 or not isinstance(tokens[0], int):
            raise ValueError("tokenizer did not return a token ID sequence")
        row = {"path": entry["path"], "input_tokens": len(tokens)}
        longest.append(row)
        if len(tokens) > max_input_tokens:
            over_limit.append(row)
        if index % 24 == 0 or index == len(entries):
            print(f"Preflighted {index}/{len(entries)} full trajectories", flush=True)
    longest.sort(key=lambda row: row["input_tokens"], reverse=True)
    result = {
        "trajectories": len(entries),
        "input_limit": max_input_tokens,
        "longest": longest[:5],
        "over_limit_count": len(over_limit),
        "over_limit_examples": over_limit[:5],
        "trajectory_bundle_sha256": bundle_hash.hexdigest(),
        "token_counts": longest,
    }
    print(json.dumps({"preflight": {k: v for k, v in result.items() if k != "token_counts"}}, indent=2), flush=True)
    if over_limit:
        raise ValueError(
            f"{len(over_limit)} full trajectories exceed the {max_input_tokens}-token "
            "input limit; no trajectory was truncated or scored"
        )
    return result


class PlainGemmaClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def _complete(self, system: str, user: str) -> dict[str, Any]:
        payload = json.dumps({
            "model": self.args.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.args.temperature,
            "seed": self.args.seed,
            "max_tokens": self.args.constructor_max_tokens,
        }).encode()
        request = urllib.request.Request(
            self.args.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout_seconds) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise RuntimeError(f"Gemma HTTP {error.code}: {detail}") from error
        choice = body["choices"][0]
        text = choice["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemma returned an empty ledger")
        return {
            "text": text,
            "usage": body.get("usage") or {},
            "finish_reason": choice.get("finish_reason"),
        }

    async def complete(self, system: str, user: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._complete, system, user)


async def _retry_complete(client: PlainGemmaClient, system: str, user: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return await client.complete(system, user)
        except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError) as error:
            last_error = error
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def _constructor_record(
    args: argparse.Namespace,
    entry: dict[str, Any],
    prompt: str,
    config_sha256: str,
    index: int,
) -> dict[str, Any]:
    checkpoint = args.output_dir / "constructor_records" / f"{index:03d}.json"
    if args.resume and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("config_sha256") != config_sha256 or saved.get("path") != entry["path"]:
            raise ValueError(f"constructor checkpoint has a different configuration: {checkpoint}")
        return saved["record"]

    started = time.monotonic()
    response: dict[str, Any] = {"text": "", "usage": {}, "finish_reason": None}
    error: str | None = None
    try:
        trajectory_json, _ = _trajectory(entry, args.data_root)
        response = asyncio.run(_retry_complete(PlainGemmaClient(args), prompt, _user_prompt(trajectory_json)))
        if response["finish_reason"] == "length":
            error = "Gemma ledger reached the output token limit"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    record = {
        **entry,
        "ledger_text": response["text"] if error is None else "",
        "partial_ledger_text": response["text"] if error is not None else None,
        "ledger_usage": response["usage"],
        "ledger_model": args.model,
        "ledger_error": error,
        "ledger_duration_seconds": round(time.monotonic() - started, 3),
    }
    _write_json(checkpoint, {"config_sha256": config_sha256, "path": entry["path"], "record": record})
    ledger_path = args.output_dir / "ledgers" / f"{index:03d}.txt"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(response["text"])
    return record


def _unknown_reader_row(record: dict[str, Any], error: Exception) -> tuple[dict, dict]:
    grader = ledger_record(record, {
        "final_verdict": "UNKNOWN",
        "violations_found": [],
        "ledger_text": "",
        "ledger_usage": {},
    })
    audit = {
        "path": record["path"],
        "prediction": {"verdict": "UNKNOWN", "violations_found": []},
        "reader_error": f"{type(error).__name__}: {error}",
        "sanitizer_status": "reader_error",
    }
    return grader, audit


def _reader_record(
    args: argparse.Namespace,
    record: dict[str, Any],
    config_sha256: str,
    index: int,
) -> tuple[dict, dict]:
    checkpoint = args.output_dir / "reader_records" / f"{index:03d}.json"
    if args.resume and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("config_sha256") != config_sha256 or saved.get("path") != record["path"]:
            raise ValueError(f"reader checkpoint has a different configuration: {checkpoint}")
        return saved["grader"], saved["audit"]

    client = VLLMGemmaClient(
        base_url=args.base_url,
        model_name=args.model,
        max_tokens=args.reader_max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            graders, audits = asyncio.run(decode_ledgers([record], client=client, concurrency=1))
            grader, audit = graders[0], audits[0]
            break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    else:
        assert last_error is not None
        grader, audit = _unknown_reader_row(record, last_error)
    _write_json(checkpoint, {
        "config_sha256": config_sha256,
        "path": record["path"],
        "grader": grader,
        "audit": audit,
    })
    return grader, audit


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_run_record(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if args.experiment_record is None:
        return
    train_path = args.train_result_dir / "reconstruction_metric.json"
    train = json.loads(train_path.read_text())
    train_scored = json.loads((args.train_result_dir / "lds_v2.json").read_text())
    heldout_scored = result["scored"]
    def row(name: str, scored: dict[str, Any]) -> str:
        return (
            f"| {name} | {scored['ul_metrics']['total']} | {scored['lds_v2']:.4f} | "
            f"{scored['discrimination']['balanced_accuracy']:.4f} | "
            f"{scored['attribution']['macro_f1']:.4f} | "
            f"{scored['ul_metrics']['unknown_rate']:.1%} |"
        )
    lines = [
        f"# Frozen Gemma ledger baseline — {date.today().isoformat()}",
        "",
        "Complete agent conversation JSON (dataset annotations omitted) → frozen Gemma factual ledger → fixed sanitizer → "
        "frozen Gemma reader → LDS-v2. No teacher model, tools, or weight updates.",
        "",
        f"- Git commit: `{_git_commit()}`",
        f"- Model: `{args.model}` at revision `{args.model_revision}`",
        f"- Runtime: `{args.runtime_id}`",
        f"- Data root: `{args.data_root}`",
        f"- Constructor prompt: `{args.prompt_file}`",
        f"- Constructor context/output: {args.max_model_len}/{args.constructor_max_tokens} tokens",
        f"- Reader output: {args.reader_max_tokens} tokens; temperature {args.temperature}; seed {args.seed}",
        f"- Sanitizer: `{SANITIZER_VERSION}`; unknown policy: `pass`",
        f"- Train manifest: `{train['meta']['manifest']}`",
        f"- Held-out manifest: `{args.manifest}`",
        f"- Train artifacts: `{args.train_result_dir}`",
        f"- Held-out artifacts: `{args.output_dir}`",
        "",
        "| Split | N | LDS-v2 | Balanced accuracy | C1–C4 macro-F1 | Unknown |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        row("Train", train_scored),
        row("Held-out", heldout_scored),
        "",
        "## Task breakdown",
        "",
        "| Split | Task | N | LDS-v2 |",
        "| --- | --- | ---: | ---: |",
    ]
    for split_name, scored in (("Train", train_scored), ("Held-out", heldout_scored)):
        for task, task_result in scored["by_domain"].items():
            score = task_result["lds_v2"]
            lines.append(f"| {split_name} | {task} | {task_result['n']} | {score:.4f} |" if score is not None else f"| {split_name} | {task} | {task_result['n']} | — |")
    lines.extend([
        "",
        f"- Constructor failures: train {train['meta']['constructor_errors']}; held-out {result['meta']['constructor_errors']}",
        f"- Unusable ledgers after sanitizer: train {train['meta']['sanitizer_unusable']}; held-out {result['meta']['sanitizer_unusable']}",
        f"- Reader failures: train {train['meta']['reader_errors']}; held-out {result['meta']['reader_errors']}",
        "- Full configurations and token counts: `run_config.json` in each artifact directory.",
        "",
        "This score measures whether the ledger lets the reader recover reference "
        "labels. It is not a direct factual similarity score against another ledger.",
        "",
    ])
    args.experiment_record.parent.mkdir(parents=True, exist_ok=True)
    args.experiment_record.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "heldout"), required=True)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-id", default="")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--constructor-max-tokens", type=int, default=8192)
    parser.add_argument("--reader-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--experiment-record", type=Path)
    parser.add_argument("--train-result-dir", type=Path)
    args = parser.parse_args()
    if args.experiment_record and (args.split != "heldout" or not args.train_result_dir):
        parser.error("--experiment-record requires --split heldout and --train-result-dir")
    return args


def main() -> None:
    args = _parse_args()
    prompt = args.prompt_file.read_text()
    entries = _load_manifest(args.manifest)
    preflight = _preflight(args, entries, prompt)
    if args.preflight_only:
        return
    config = {
        "split": args.split,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest.read_bytes()),
        "trajectory_bundle_sha256": preflight["trajectory_bundle_sha256"],
        "input_format": "complete-conversation-json-without-dataset-metadata-v1",
        "model": args.model,
        "model_revision": args.model_revision,
        "max_model_len": args.max_model_len,
        "constructor_max_tokens": args.constructor_max_tokens,
        "reader_max_tokens": args.reader_max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "runner_code_sha256": _sha256(Path(__file__).read_bytes()),
        "metric_code_sha256": _sha256((Path(__file__).parent / "reconstruction_metric.py").read_bytes()),
        "ledger_score_code_sha256": _sha256((Path(__file__).parent / "ledger_score.py").read_bytes()),
        "scorer_code_sha256": _sha256((Path(__file__).parent / "lds_v2.py").read_bytes()),
        "ul_metrics_code_sha256": _sha256((Path(__file__).parent / "recompute_metrics.py").read_bytes()),
        "constructor_prompt_sha256": _sha256(prompt.encode()),
        "reader_prompt_sha256": _sha256(GEMMA_DECODER_SYSTEM_PROMPT.encode()),
        "sanitizer_version": SANITIZER_VERSION,
        "metric": RECONSTRUCTION_METRIC_NAME,
        "unknown_policy": "pass",
        "tools": [],
    }
    config_sha256 = _sha256(json.dumps(config, sort_keys=True).encode())
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old.get("config_sha256") != config_sha256:
            raise ValueError(f"output directory has a different configuration: {args.output_dir}")
    _write_json(config_path, {"config_sha256": config_sha256, "config": config, "preflight": preflight})

    records = []
    for index, entry in enumerate(entries, 1):
        record = _constructor_record(args, entry, prompt, config_sha256, index)
        records.append(record)
        status = "ERROR" if record["ledger_error"] else "OK"
        print(f"[{index}/{len(entries)}] constructor {status}: {entry['path']}", flush=True)

    graders: list[dict] = []
    audits: list[dict] = []
    for index, record in enumerate(records, 1):
        grader, audit = _reader_record(args, record, config_sha256, index)
        graders.append(grader)
        audits.append(audit)
        print(f"[{index}/{len(entries)}] reader {audit['prediction']['verdict']}: {record['path']}", flush=True)

    scored = score_ledgers(graders, unknown_policy="pass")
    metadata = {
        "experiment": "frozen-gemma-full-trajectory-ledger-baseline-v1",
        "split": args.split,
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "config_sha256": config_sha256,
        "model_revision": args.model_revision,
        "runtime_id": args.runtime_id,
        "constructor_errors": sum(bool(row["ledger_error"]) for row in records),
        "sanitizer_unusable": sum(row.get("sanitizer_status") == "unusable" for row in audits),
        "reader_errors": sum("reader_error" in row for row in audits),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    result = {"metric": RECONSTRUCTION_METRIC_NAME, "score": scored["lds_v2"], "scored": scored, "meta": metadata}
    _write_json(args.output_dir / "records.json", records)
    _write_json(args.output_dir / "gemma_predictions_private.json", audits)
    _write_json(args.output_dir / "reconstruction_grader_rows_private.json", graders)
    _write_json(args.output_dir / "lds_v2.json", scored)
    _write_json(args.output_dir / "reconstruction_metric.json", result)
    _write_run_record(args, result)
    print(f"{args.split} LDS-v2: {scored['lds_v2']:.4f}", flush=True)
    print(f"Artifacts: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
