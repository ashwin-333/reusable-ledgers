"""Graded suspicion reads. Readers only; stored ledgers are reused.

The suspicion rubric is frozen. The original policy files are not edited.
Each reader gets that same rubric appended, and the same suspicion field
added to its existing response schema. An empty ledger is not sent to the
model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from config import OPENAI_BASE_URL, READER_MODEL

HERE = Path(__file__).resolve().parent

from autoresearch import gemma_baseline as gb
from autoresearch.reconstruction_metric import GEMMA_DECODER_SYSTEM_PROMPT, GEMMA_RESPONSE_FORMAT
from ledgers.e3_policy import ATTACKS, POLICY_RESPONSE_FORMAT, _frozen_prompt
from ledgers.ledger_conditioned import (
    _format_template,
    _ledgers_for,
    _neutral_ledgers,
    _shade_ledgers,
    _trajectory,
    _user_message,
)
from ledgers.run_eval import _json_user_prompt
from ledgers.shade_run import RESPONSE_FORMAT as SHADE_RESPONSE_FORMAT
from ledgers.shade_run import VIOLATIONS, _trajectory_text as shade_trajectory, frozen_policy

RUBRIC_PATH = HERE / "prompts/graded_suspicion.txt"
RUBRIC_SHA = "584aa2f3d5e57d2279b4013601d5403674b4e198a403e81eb184aa20500f2714"
MODEL_12 = READER_MODEL
CALENDAR_LABELS = ("C1", "C2", "C3", "C4")


def rubric_text() -> str:
    text = RUBRIC_PATH.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != RUBRIC_SHA:
        raise SystemExit(f"suspicion rubric was edited ({digest} != {RUBRIC_SHA}); refusing to run")
    return text


def graded_system(base: str) -> str:
    return base.rstrip() + "\n\n" + rubric_text()


def with_suspicion(response_format: dict) -> dict:
    updated = copy.deepcopy(response_format)
    schema = updated["json_schema"]["schema"]
    schema["properties"]["suspicion"] = {"type": "integer"}
    required = list(schema.get("required") or [])
    if "suspicion" not in required:
        required.insert(0, "suspicion")
    schema["required"] = required
    return updated


def _reader(surface: str) -> tuple[str, dict, int, str, tuple[str, ...]]:
    if surface == "calendar":
        return graded_system(GEMMA_DECODER_SYSTEM_PROMPT), with_suspicion(GEMMA_RESPONSE_FORMAT), 256, "violations_found", CALENDAR_LABELS
    if surface == "cua":
        return graded_system(_frozen_prompt()), with_suspicion(POLICY_RESPONSE_FORMAT), 160, "attacks_found", ATTACKS
    return graded_system(frozen_policy()), with_suspicion(SHADE_RESPONSE_FORMAT), 160, "violations_found", VIOLATIONS


def _parse(text: str, found_key: str, allowed: tuple[str, ...]) -> tuple[int, str, list[str]]:
    payload = json.loads(text)
    suspicion = payload.get("suspicion")
    if isinstance(suspicion, float) and suspicion.is_integer():
        suspicion = int(suspicion)
    if not isinstance(suspicion, int) or not 1 <= suspicion <= 10:
        raise ValueError(f"suspicion was {suspicion!r}")
    verdict = payload.get("verdict")
    found = [item for item in payload.get(found_key) or [] if item in allowed]
    if verdict not in ("PASS", "FAIL"):
        raise ValueError("verdict was not PASS or FAIL")
    verdict = "FAIL" if found else "PASS"
    return suspicion, verdict, found


def _post(base_url: str, model: str, system: str, user: str, response_format: dict, max_tokens: int, timeout: float) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 0,
        "response_format": response_format,
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    if body.get("model") != model:
        raise RuntimeError(f"reader answered as {body.get('model')!r}, not {model}")
    text = body["choices"][0]["message"]["content"] or ""
    if not text.strip():
        raise RuntimeError("empty completion")
    return text


def _checkpoint(directory: Path, path: str) -> Path:
    name = hashlib.sha256(path.encode()).hexdigest()[:16]
    return directory / f"{name}.json"


def _raw_user(surface: str, entry: dict, data_root: Path) -> str:
    if surface == "calendar":
        return _json_user_prompt(entry, data_root)
    if surface == "cua":
        _source, text = _trajectory("cua", entry, data_root)
        return text
    return shade_trajectory(entry, data_root)


def _one_call(args, system, response_format, max_tokens, found_key, allowed, user: str) -> tuple[int | None, str, list[str], str | None, str]:
    if not user.strip():
        return None, "UNKNOWN", [], "empty input", ""
    last = "empty input"
    for attempt in range(3):
        try:
            text = _post(args.base_url, args.served_model, system, user, response_format, max_tokens, args.timeout_seconds)
            suspicion, verdict, found = _parse(text, found_key, allowed)
            return suspicion, verdict, found, None, text
        except (
            OSError, urllib.error.URLError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError,
        ) as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    return None, "UNKNOWN", [], last, ""


def _save(path: Path, prompt_sha: str, trajectory: str, saved: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gb._write_json(path, {
        "rubric_sha256": RUBRIC_SHA,
        "prompt_sha256": prompt_sha,
        "path": trajectory,
        **saved,
    })


def _resume(path: Path, prompt_sha: str, trajectory: str) -> bool:
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    if saved.get("rubric_sha256") != RUBRIC_SHA or saved.get("prompt_sha256") != prompt_sha or saved.get("path") != trajectory:
        raise SystemExit(f"graded checkpoint mismatch: {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("ledger", "raw", "conditioned"), required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--surface", choices=("calendar", "cua", "shade"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--ledger-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL)
    parser.add_argument("--served-model", default=MODEL_12)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    system, response_format, max_tokens, found_key, allowed = _reader(args.surface)
    prompt_sha = hashlib.sha256(system.encode()).hexdigest()
    entries = gb._load_manifest(args.manifest)
    if args.kind == "ledger":
        if args.source_dir is None:
            raise SystemExit("ledger reads need --source-dir")
        loader = _shade_ledgers if args.surface == "shade" else _neutral_ledgers
        ledgers = loader(args.source_dir)
        missing = [entry["path"] for entry in entries if entry["path"] not in ledgers]
        if missing:
            raise SystemExit(f"{args.system} is missing {len(missing)} ledgers")
    elif args.kind == "conditioned":
        if args.ledger_root is None:
            raise SystemExit("conditioned reads need --ledger-root")
        _format_template()
        ledgers_by_seed = _ledgers_for(args.surface, args.ledger_root)
    if args.check:
        print(f"rubric_sha256={RUBRIC_SHA} prompt_sha256={prompt_sha} n={len(entries)}", flush=True)
        return
    if not args.base_url:
        raise SystemExit("--base-url is required for a read")
    mine = [(i, entry) for i, entry in enumerate(entries, 1) if (i - 1) % args.num_chunks == args.chunk]

    def one(index: int, entry: dict) -> str:
        if args.kind == "conditioned":
            bits = []
            template = _format_template()
            for seed, table in ledgers_by_seed.items():
                dest = _checkpoint(args.output_dir / seed / "predictions", entry["path"])
                if _resume(dest, prompt_sha, entry["path"]):
                    bits.append("resume")
                    continue
                ledger = table[entry["path"]]
                if not ledger.strip():
                    _save(dest, prompt_sha, entry["path"], {
                        "seed": seed, "suspicion": None, "verdict": "UNKNOWN", "found": [], "error": "empty ledger", "raw": "",
                    })
                    bits.append("UNKNOWN")
                    continue
                try:
                    _source, trajectory = _trajectory(args.surface, entry, args.data_root)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    _save(dest, prompt_sha, entry["path"], {
                        "seed": seed, "suspicion": None, "verdict": "UNKNOWN", "found": [],
                        "error": f"{type(exc).__name__}: {exc}", "raw": "",
                    })
                    bits.append("UNKNOWN")
                    continue
                user = _user_message(template, ledger, trajectory)
                suspicion, verdict, found, error, raw = _one_call(
                    args, system, response_format, max_tokens, found_key, allowed, user,
                )
                _save(dest, prompt_sha, entry["path"], {
                    "seed": seed, "suspicion": suspicion, "verdict": verdict, "found": found, "error": error, "raw": raw,
                })
                bits.append(str(suspicion if suspicion is not None else "UNKNOWN"))
            return " ".join(bits)
        dest = _checkpoint(args.output_dir / "predictions", entry["path"])
        if _resume(dest, prompt_sha, entry["path"]):
            return "resume"
        if args.kind == "ledger":
            user = ledgers[entry["path"]]
            if args.surface == "calendar" and user.strip():
                user = "[NEUTRAL_LEDGER]\n" + user
            error_prefix = "empty ledger"
        else:
            try:
                user = _raw_user(args.surface, entry, args.data_root)
                error_prefix = "empty trajectory"
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                _save(dest, prompt_sha, entry["path"], {
                    "suspicion": None, "verdict": "UNKNOWN", "found": [],
                    "error": f"{type(exc).__name__}: {exc}", "raw": "",
                })
                return "UNKNOWN"
        if not user.strip():
            _save(dest, prompt_sha, entry["path"], {
                "suspicion": None, "verdict": "UNKNOWN", "found": [], "error": error_prefix, "raw": "",
            })
            return "UNKNOWN"
        suspicion, verdict, found, error, raw = _one_call(
            args, system, response_format, max_tokens, found_key, allowed, user,
        )
        _save(dest, prompt_sha, entry["path"], {
            "suspicion": suspicion, "verdict": verdict, "found": found, "error": error, "raw": raw,
        })
        return str(suspicion if suspicion is not None else "UNKNOWN")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, index, entry): index for index, entry in mine}
        for done, future in enumerate(as_completed(futures), 1):
            print(f"[{done}/{len(mine)}] #{futures[future]:03d} {future.result()}", flush=True)


if __name__ == "__main__":
    main()
