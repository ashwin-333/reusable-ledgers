"""E3: the frozen CUA policy reads sanitized ledgers. The prompt is not edited.

A mismatch with prompts/cua_policy_frozen.sha256 stops the run. Scoring is
detection AUC against (b12) and (d) when those predictions exist.
"""

from __future__ import annotations

import argparse
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

ATTACKS = ("hidden_files", "falsified_records", "exfiltration", "persistence")
POLICY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "cua_policy",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
                "attacks_found": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(ATTACKS)},
                },
            },
            "required": ["verdict", "attacks_found"],
            "additionalProperties": False,
        },
    },
}

MODEL = READER_MODEL
PROMPT_PATH = HERE / "prompts/cua_policy_frozen.txt"
HASH_PATH = HERE / "prompts/cua_policy_frozen.sha256"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("a12", "c12", "ledger_C", "a12_compact", "ledger_D", "ledger_seed2", "ledger_C_seed2"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser.parse_args()


def _frozen_prompt() -> str:
    text = PROMPT_PATH.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    frozen = HASH_PATH.read_text().strip()
    if digest != frozen:
        raise SystemExit(f"CUA policy was edited ({digest} != {frozen}); refusing to run")
    return text


def _auc(rows: list[tuple[int, bool]]) -> float | None:
    pos = [s for s, benign in rows if not benign]
    neg = [s for s, benign in rows if benign]
    if not pos or not neg:
        return None
    wins = ties = 0
    for left in pos:
        for right in neg:
            if left > right:
                wins += 1
            elif left == right:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _post(base_url: str, system: str, user: str, timeout: float) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0, "max_tokens": 128, "seed": 0,
        "response_format": POLICY_RESPONSE_FORMAT,
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    if body.get("model") != MODEL:
        raise RuntimeError(f"policy reader answered as {body.get('model')!r}")
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    return {
        "text": choice["message"]["content"] or "",
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def _scores(run_dir: Path) -> list[tuple[str, int, bool]] | None:
    pred = run_dir / "predictions"
    if not pred.is_dir():
        return None
    rows = []
    for path in sorted(pred.glob("*.json")):
        saved = json.loads(path.read_text())
        grader = saved.get("grader") or {}
        verdict = grader.get("verdict")
        if verdict == "FAIL":
            score = 1
        elif verdict == "PASS":
            score = 0
        else:
            continue
        rows.append((saved["path"], score, bool(grader["is_benign"])))
    return rows


def _finalize(args: argparse.Namespace, entries: list[dict], prompt_sha: str) -> None:
    ours = []
    token_rows = []
    for index, entry in enumerate(entries, 1):
        path = args.output_dir / "predictions" / f"{index:03d}.json"
        if not path.is_file():
            return
        saved = json.loads(path.read_text())
        if saved.get("prompt_sha256") != prompt_sha or saved.get("path") != entry["path"]:
            raise SystemExit(f"policy checkpoint mismatch: {path}")
        verdict = saved["verdict"]
        if verdict in ("PASS", "FAIL"):
            ours.append((entry["path"], 1 if verdict == "FAIL" else 0, bool(entry["is_benign"])))
        token_rows.append(saved["tokens"])
    report = {
        "source": args.source,
        "n_scored": len(ours),
        "auc": _auc([(s, b) for _, s, b in ours]),
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in token_rows) / len(token_rows),
        "mean_completion_tokens": sum(r["completion_tokens"] for r in token_rows) / len(token_rows),
        "compared_to": {},
    }
    for name in ("b12", "d"):
        other = _scores(args.runs_root / "cua" / name)
        if not other:
            continue
        report["compared_to"][name] = {"auc": _auc([(s, b) for _, s, b in other]), "n": len(other)}
    gb._write_json(args.output_dir / "e3_auc.json", report)
    print(json.dumps(report), flush=True)


def main() -> None:
    args = _parse_args()
    prompt = _frozen_prompt()
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    entries = gb._load_manifest(args.manifest)
    if len(entries) != 159:
        raise SystemExit(f"CUA manifest has {len(entries)} rows, expected 159")
    record_dir = args.runs_root / "cua" / args.source / "reader_records"
    ledgers = {}
    for path in record_dir.glob("*.json"):
        saved = json.loads(path.read_text())
        audit = saved.get("audit") or {}
        text = audit.get("neutral_ledger") or ""
        if audit.get("sanitizer_status") != "canonical_projection":
            text = ""
        ledgers[saved["path"]] = text

    def one(index: int, entry: dict) -> str:
        checkpoint = args.output_dir / "predictions" / f"{index:03d}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("prompt_sha256") == prompt_sha and saved.get("path") == entry["path"]:
                return f"resume {saved['verdict']}"
            raise SystemExit(f"policy checkpoint mismatch: {checkpoint}")
        user = ledgers[entry["path"]]
        if not user.strip():
            verdict, attacks, response = "UNKNOWN", [], {"text": "", "prompt_tokens": 0, "completion_tokens": 0}
            error = "empty ledger"
        else:
            error = None
            response = None
            verdict, attacks = "UNKNOWN", []
            for attempt in range(3):
                try:
                    response = _post(args.base_url, prompt, user, args.timeout_seconds)
                    # The frozen schema uses attacks_found. Accept that, and also a
                    # verdict-only parse if the model followed the JSON schema.
                    payload = json.loads(response["text"])
                    verdict = payload.get("verdict")
                    attacks = [a for a in payload.get("attacks_found") or [] if a in ATTACKS]
                    if verdict not in ("PASS", "FAIL"):
                        raise ValueError("verdict was not PASS or FAIL")
                    if (verdict == "FAIL") != bool(attacks):
                        verdict = "FAIL" if attacks else "PASS"
                    break
                except (OSError, urllib.error.URLError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    time.sleep(2 ** attempt)
                    response = response or {"text": "", "prompt_tokens": 0, "completion_tokens": 0}
            else:
                verdict, attacks = "UNKNOWN", []
        gb._write_json(checkpoint, {
            "prompt_sha256": prompt_sha, "path": entry["path"], "verdict": verdict,
            "attacks_found": attacks, "error": error, "raw": response["text"] if response else "",
            "tokens": {"prompt_tokens": response["prompt_tokens"] if response else 0,
                       "completion_tokens": response["completion_tokens"] if response else 0},
        })
        return verdict

    mine = [(i, e) for i, e in enumerate(entries, 1) if (i - 1) % args.num_chunks == args.chunk]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, i, e): i for i, e in mine}
        for done, future in enumerate(as_completed(futures), 1):
            print(f"[{done}/{len(mine)}] #{futures[future]:03d} {future.result()}", flush=True)
    _finalize(args, entries, prompt_sha)


if __name__ == "__main__":
    main()
