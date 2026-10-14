"""E5: each frozen policy is judged by the 12B reader. Policy text is not edited.

Widths are frozen. 1 is one call per policy, 5 is the first half then the
second half of policies.json, and 10 is every policy in one call. Direct-SFT
and rationale are not called here; score_e5.py applies their existing verdicts
to the policy labels.
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
from config import DATA_ROOT, OPENAI_BASE_URL, READER_MODEL, RUNS_DIR

HERE = Path(__file__).resolve().parent

from autoresearch import gemma_baseline as gb
from ledgers.run_eval import _json_user_prompt

POLICY_SHA = "41551599e87af48d3ab1822f2145281bbb763f298f75ff1a5d2c3fa575b0568a"
WIDTHS = (1, 5, 10)
MAX_TOKENS = {1: 64, 5: 192, 10: 320}
DATA_ROOTS = {
    "original": DATA_ROOT / "original_trajectories",
    "fresh": DATA_ROOT / "fresh_heldout_data",
}
BENCH_DIR = DATA_ROOT / "multi_policy"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_frozen() -> tuple[str, str, list[dict], list[dict]]:
    system_path = HERE / "prompts" / "e5_reader_system.txt"
    format_path = HERE / "prompts" / "e5_user_format.txt"
    system_sha = (HERE / "prompts" / "e5_reader_system.sha256").read_text().strip()
    format_sha = (HERE / "prompts" / "e5_user_format.sha256").read_text().strip()
    if _sha(system_path) != system_sha or _sha(format_path) != format_sha:
        raise SystemExit("E5 reader wrapper was edited; refusing to run")
    if not (BENCH_DIR / "READY_multi_policy").is_file():
        raise SystemExit(f"missing {BENCH_DIR / 'READY_multi_policy'}")
    policy_path = BENCH_DIR / "policies.json"
    if _sha(policy_path) != POLICY_SHA:
        raise SystemExit("policies.json hash does not match the frozen benchmark")
    spec = json.loads(policy_path.read_text())
    policies = spec["policies"]
    if [p["id"] for p in policies] != [
        "no_stripe_key_in_email", "no_external_contact_create", "no_phone_contacts",
        "no_spam_folder_changes", "no_test_mailbox", "no_email_to_new_external_contact",
        "no_bulk_recipients", "no_direct_send", "no_external_resolved_recipient",
        "no_credential_line_in_email",
    ]:
        raise SystemExit("policy ids or order changed")
    bench = json.loads((BENCH_DIR / "multi_policy_benchmark.json").read_text())
    if bench.get("policy_sha256") != POLICY_SHA or len(bench["trajectories"]) != 192:
        raise SystemExit("benchmark file does not match the frozen policies")
    return system_path.read_text(), format_path.read_text(), policies, bench["trajectories"]


def _groups(policies: list[dict], width: int) -> list[list[dict]]:
    if width == 1:
        return [[p] for p in policies]
    if width == 5:
        return [policies[:5], policies[5:]]
    if width == 10:
        return [policies]
    raise SystemExit(f"unsupported width {width}")


def _user_message(template: str, policies: list[dict], body: str) -> str:
    blocks = "\n\n".join(f"[{item['id']}]\n{item['text']}" for item in policies)
    head, rest = template.split("{blocks}")
    mid, tail = rest.split("{body}")
    message = head + blocks + mid + body + tail
    for item in policies:
        if item["text"] not in message:
            raise RuntimeError(f"policy text was not copied verbatim: {item['id']}")
    return message


def _schema(policies: list[dict]) -> dict:
    if len(policies) == 1:
        inner = {
            "type": "object",
            "properties": {"decision": {"type": "string", "enum": ["violates", "does_not"]}},
            "required": ["decision"],
            "additionalProperties": False,
        }
    else:
        inner = {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "minItems": len(policies),
                    "maxItems": len(policies),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "enum": [p["id"] for p in policies]},
                            "decision": {"type": "string", "enum": ["violates", "does_not"]},
                        },
                        "required": ["id", "decision"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["decisions"],
            "additionalProperties": False,
        }
    return {"type": "json_schema", "json_schema": {"name": "policy_decisions", "strict": True, "schema": inner}}


def _parse(policies: list[dict], text: str) -> dict[str, str]:
    payload = json.loads(text)
    if len(policies) == 1:
        decision = payload.get("decision")
        if decision not in ("violates", "does_not"):
            raise ValueError("decision was not violates or does_not")
        return {policies[0]["id"]: decision}
    found = {}
    for item in payload.get("decisions") or []:
        pid = item.get("id")
        decision = item.get("decision")
        if pid in found or decision not in ("violates", "does_not"):
            raise ValueError("duplicate or invalid decision")
        found[pid] = decision
    if set(found) != {p["id"] for p in policies}:
        raise ValueError("decisions did not cover the stated policies")
    return found


def _post(base_url: str, model: str, system: str, user: str, schema: dict, max_tokens: int, timeout: float) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 0,
        "response_format": schema,
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    if body.get("model") != model:
        raise RuntimeError(f"reader answered as {body.get('model')!r}, not {model}")
    usage = body.get("usage") or {}
    return {
        "text": body["choices"][0]["message"]["content"] or "",
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def _ledgers(runs_root: Path, system: str) -> dict[tuple[str, str], str]:
    found = {}
    for set_name in ("original", "fresh"):
        record_dir = runs_root / set_name / system / "reader_records"
        if not record_dir.is_dir():
            raise SystemExit(f"no ledgers at {record_dir}")
        for path in record_dir.glob("*.json"):
            saved = json.loads(path.read_text())
            audit = saved.get("audit") or {}
            text = audit.get("neutral_ledger") or ""
            if audit.get("sanitizer_status") != "canonical_projection":
                text = ""
            found[(set_name, saved["path"])] = text
    return found


def _record_body(system: str, row: dict, ledgers: dict[tuple[str, str], str]) -> tuple[str, str | None]:
    if system == "b12":
        return _json_user_prompt(row, DATA_ROOTS[row["set"]]), None
    text = ledgers.get((row["set"], row["path"]), "")
    if not text.strip():
        return "", "empty ledger"
    return text, None


def _one_width(base_url: str, model: str, system_prompt: str, template: str, policies: list[dict], body: str, width: int, timeout: float) -> dict:
    decisions = {}
    prompt_tokens = []
    completion_tokens = []
    error = None
    for group in _groups(policies, width):
        schema = _schema(group)
        user = _user_message(template, group, body)
        response = None
        parsed = None
        last = None
        for attempt in range(3):
            try:
                response = _post(base_url, model, system_prompt, user, schema, MAX_TOKENS[width], timeout)
                parsed = _parse(group, response["text"])
                last = None
                break
            except (OSError, urllib.error.URLError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                if "HTTP 400" in last:
                    break
                time.sleep(2 ** attempt)
        prompt_tokens.append(0 if response is None else response["prompt_tokens"])
        completion_tokens.append(0 if response is None else response["completion_tokens"])
        if parsed is None:
            error = last
            for item in group:
                decisions[item["id"]] = "unknown"
        else:
            decisions.update(parsed)
    return {
        "decisions": decisions,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
    }


def _finalize(output_dir: Path, rows: list[dict], system_sha: str) -> None:
    saved_rows = []
    for index, row in enumerate(rows, 1):
        path = output_dir / "predictions" / f"{index:03d}.json"
        if not path.is_file():
            return
        saved = json.loads(path.read_text())
        if saved.get("system_sha256") != system_sha or saved.get("path") != row["path"]:
            raise SystemExit(f"e5 checkpoint mismatch: {path}")
        saved_rows.append(saved)
    tokens = {}
    for width in WIDTHS:
        per_call = []
        per_traj = []
        for saved in saved_rows:
            calls = saved["widths"][str(width)]["prompt_tokens"]
            called = [n for n in calls if n]
            per_call.extend(called)
            per_traj.append(sum(calls))
        tokens[str(width)] = {
            "mean_prompt_tokens_per_call": (sum(per_call) / len(per_call)) if per_call else 0,
            "mean_prompt_tokens_per_trajectory": (sum(per_traj) / len(per_traj)) if per_traj else 0,
            "n_calls": len(per_call),
            "n_trajectories": len(per_traj),
        }
    gb._write_json(output_dir / "e5_tokens.json", tokens)
    print(json.dumps(tokens), flush=True)


def _preflight(runs_root: Path) -> None:
    _, _, policies, rows = _load_frozen()
    if len(policies) != 10 or len(rows) != 192:
        raise SystemExit("expected 10 policies and 192 trajectories")
    missing = []
    systems = ["a12", "c12", "ledger_C", "a12_compact"]
    if (runs_root / "original" / "ledger_D" / "reader_records").is_dir():
        systems.append("ledger_D")
    for system in systems:
        ledgers = _ledgers(runs_root, system)
        for row in rows:
            if (row["set"], row["path"]) not in ledgers:
                missing.append(f"{system} {row['set']} {row['path']}")
    if missing:
        raise SystemExit(f"{len(missing)} ledgers missing, first {missing[:3]}")
    print(json.dumps({"policies": len(policies), "trajectories": len(rows), "ledgers": "ok"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("a12", "c12", "b12", "ledger_C", "a12_compact", "ledger_D"))
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL)
    parser.add_argument("--served-model", default=READER_MODEL)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        _preflight(args.runs_root)
        return
    system_prompt, template, policies, rows = _load_frozen()
    system_sha = hashlib.sha256(system_prompt.encode()).hexdigest()
    ledgers = {} if args.system == "b12" else _ledgers(args.runs_root, args.system)

    def one(index: int, row: dict) -> str:
        checkpoint = args.output_dir / "predictions" / f"{index:03d}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("system_sha256") == system_sha and saved.get("path") == row["path"] and set(saved.get("widths") or {}) == {"1", "5", "10"}:
                return "resume"
            raise SystemExit(f"e5 checkpoint mismatch: {checkpoint}")
        body, skip = _record_body(args.system, row, ledgers)
        widths = {}
        if skip or not body:
            for width in WIDTHS:
                widths[str(width)] = {
                    "decisions": {p["id"]: "unknown" for p in policies},
                    "prompt_tokens": [],
                    "completion_tokens": [],
                    "error": skip or "empty record",
                }
        else:
            for width in WIDTHS:
                widths[str(width)] = _one_width(
                    args.base_url, args.served_model, system_prompt, template,
                    policies, body, width, args.timeout_seconds,
                )
        gb._write_json(checkpoint, {
            "system_sha256": system_sha,
            "policy_sha256": POLICY_SHA,
            "path": row["path"],
            "set": row["set"],
            "widths": widths,
        })
        return "ok"

    mine = [(i, row) for i, row in enumerate(rows, 1) if (i - 1) % args.num_chunks == args.chunk]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, i, row): i for i, row in mine}
        for done, future in enumerate(as_completed(futures), 1):
            print(f"[{done}/{len(mine)}] #{futures[future]:03d} {future.result()}", flush=True)
    _finalize(args.output_dir, rows, system_sha)


if __name__ == "__main__":
    main()
