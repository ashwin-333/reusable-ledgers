"""Final multi-policy controls. New directories only.

The frozen E5 reader wrapper and the frozen graded rubric are appended, not
edited. Width 1 is one call per policy. The other width is one call with
every policy. Binary decision and suspicion come from that same call.
Policy text is read from the frozen benchmark and is not rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from config import DATA_ROOT, OPENAI_BASE_URL, READER_MODEL, RUNS_DIR, STATUS_DIR, SUBSET_FILE

HERE = Path(__file__).resolve().parent

from autoresearch import gemma_baseline as gb
from autoresearch.reconstruction_metric import _SECTION_RE
from ledgers.read_policy import _parse as _parse_binary
from ledgers.read_policy import _post, _sha, _user_message
from ledgers.e5_real import _shade_ledgers
from ledgers.read_graded import RUBRIC_SHA, graded_system, rubric_text
from ledgers.evaluative_strip import rule_sha256, strip_ledger
from ledgers.shade_run import _trajectory_text

SFT = DATA_ROOT
RUNS = RUNS_DIR
CONTROLS = RUNS / "mp_controls"
SHADE_BENCH = SFT / "multi_policy_real"
CALENDAR_BENCH = SFT / "multi_policy"
HASH_FILE = STATUS_DIR / "e5_real_policy.sha256"
SUBSET = SUBSET_FILE
READY_TOOLLOG = SFT / "READY_toollog"
READY_EVENT = SFT / "READY_eventrecord"
LEDGER_BODIES = {"ledger_a12": "a12", "ledger_C": "ledger_C", "ledger_C_seed2": "ledger_C_seed2"}
ACTION_ONLY = {"action_only_C": "ledger_C", "action_only_C_seed2": "ledger_C_seed2"}
STRIPPED = {"stripped_C": "ledger_C", "stripped_C_seed2": "ledger_C_seed2"}
BODIES = (
    "raw", "compact", "lingua_median", "lingua_x4",
    "ledger_a12", "ledger_C", "ledger_C_seed2",
    "toollog_full", "toollog_budget", "toollog_budget_percall",
    "summary", "eventrecord", "action_summary", "action_summary_8192",
    "action_only_C", "action_only_C_seed2",
    "stripped_C", "stripped_C_seed2",
)
SUMMARY_DIR = CONTROLS / "summary" / "texts"
SUMMARY_SHA = HERE / "prompts" / "summary_factual.sha256"
ACTION_SUMMARY_DIR = CONTROLS / "action_summary" / "texts"
ACTION_SUMMARY_8192_DIR = CONTROLS / "action_summary_8192" / "texts"
ACTION_SUMMARY_SHA = HERE / "prompts" / "action_summary.sha256"
TOOL_DIRS = {
    "toollog_full": "toollog_full",
    "toollog_budget": "toollog_budget",
    "toollog_budget_percall": "toollog_budget_percall",
}
MAX_TOKENS = {1: 128, 8: 512, 10: 640}


def _guard(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    root = CONTROLS.resolve()
    if root not in resolved.parents and resolved != root:
        raise SystemExit(f"refusing to write outside {root}: {output_dir}")
    banned = ("multi_policy_real", "multi_policy/e5_", "/shade/", "/graded/")
    text = str(resolved)
    if any(part in text for part in banned):
        raise SystemExit(f"refusing to touch an existing run directory: {output_dir}")


def _load_bench(bench: Path, hash_file: Path) -> tuple[str, str, list[dict], list[dict], str]:
    system_path = HERE / "prompts" / "e5_reader_system.txt"
    format_path = HERE / "prompts" / "e5_user_format.txt"
    system_sha = (HERE / "prompts" / "e5_reader_system.sha256").read_text().strip()
    format_sha = (HERE / "prompts" / "e5_user_format.sha256").read_text().strip()
    if _sha(system_path) != system_sha or _sha(format_path) != format_sha:
        raise SystemExit("E5 reader wrapper was edited; refusing to run")
    rubric_text()
    policy_path = bench / "policies.json"
    policy_sha = _sha(policy_path)
    if hash_file.is_file():
        frozen = hash_file.read_text().strip()
        if policy_sha != frozen:
            raise SystemExit("policies.json changed after it was frozen; refusing to run")
    policies = json.loads(policy_path.read_text())["policies"]
    payload = json.loads((bench / "multi_policy_benchmark.json").read_text())
    if payload.get("policy_sha256") not in (None, policy_sha):
        raise SystemExit("benchmark policy_sha256 does not match policies.json")
    return system_path.read_text(), format_path.read_text(), policies, payload["trajectories"], policy_sha


def _data_root(bench: Path) -> Path:
    payload = json.loads((bench / "multi_policy_benchmark.json").read_text())
    stated = payload.get("data_root")
    if stated:
        return Path(stated)
    if bench == SHADE_BENCH:
        return SFT / "shade_real"
    return CALENDAR_BENCH


def _compact_body(entry: dict, data_root: Path) -> str:
    """Same user wrapper the calendar writer trains on, over the stored turns.

    Tool text already on the turn is kept in full. Nothing is dropped to a
    budget here; that is the separate tool-log control.
    """
    document = json.loads((data_root / entry["path"]).read_text())
    lines = []
    for turn in document.get("conversation") or []:
        role = str(turn.get("role") or "unknown")
        if role.lower() == "system":
            continue
        content = turn.get("content")
        if isinstance(content, str) and content.strip():
            lines.append(role.upper() + ": " + content.strip())
        calls = turn.get("tool_calls")
        if isinstance(calls, str) and calls.strip() and calls.strip() != "Tool calls:":
            lines.append(calls.strip())
        elif isinstance(calls, list) and calls:
            names = []
            for call in calls:
                if isinstance(call, dict):
                    name = call.get("name") or (call.get("function") or {}).get("name")
                    if name:
                        names.append(str(name))
                elif isinstance(call, str) and call.strip():
                    names.append(call.strip())
            if names:
                lines.append("Tool calls:")
                lines.extend("- " + name for name in names)
    text = "\n".join(lines).strip()
    if not text:
        raise RuntimeError("empty compact rendering")
    sep = "-" * 50
    return "### INPUT TRAJECTORY\n" + sep + "\n" + text + "\n" + sep


def _stored_text(directory: Path, sha_path: Path, path: str, label: str) -> str:
    """Text written before this reader. Empty if the writer stored an empty string."""
    if not sha_path.is_file():
        raise RuntimeError(label + " prompt hash is not frozen")
    expected = sha_path.read_text().strip()
    found = None
    if directory.is_dir():
        for file in directory.glob("*.json"):
            saved = json.loads(file.read_text())
            if saved.get("path") != path:
                continue
            if saved.get("prompt_sha256") != expected:
                raise RuntimeError(label + " was written with a different prompt")
            found = saved.get("text") or ""
            break
    if found is None:
        raise RuntimeError("missing " + label)
    return found


def _summary_body(path: str) -> str:
    """Ordinary factual summary written before this reader. Empty if missing."""
    return _stored_text(SUMMARY_DIR, SUMMARY_SHA, path, "summary")


def _action_summary_body(path: str) -> str:
    """Length-targeted action account written before this reader."""
    return _stored_text(ACTION_SUMMARY_DIR, ACTION_SUMMARY_SHA, path, "action summary")


def _action_summary_8192_body(path: str) -> str:
    """Same action-summary prompt, with the ledger writer's 8192-token ceiling."""
    return _stored_text(ACTION_SUMMARY_8192_DIR, ACTION_SUMMARY_SHA, path, "action summary 8192")


def _keep_action(text: str) -> str:
    """Sanitized ledger text from the ACTION_LEDGER header through the next header."""
    kept = []
    keeping = False
    for line in text.splitlines():
        match = _SECTION_RE.match(line.strip())
        if match:
            keeping = match.group(1).strip() == "ACTION_LEDGER"
            if keeping:
                kept.append(line)
            continue
        if keeping:
            kept.append(line)
    return "\n".join(kept).strip()


def _lingua_body(kind: str, path: str) -> str:
    name = "median" if kind == "lingua_median" else "x4"
    directory = CONTROLS / "lingua" / name
    found = None
    if directory.is_dir():
        for file in directory.glob("*.json"):
            saved = json.loads(file.read_text())
            if saved.get("path") == path and (saved.get("text") or "").strip():
                found = saved["text"]
                break
    if not found:
        raise RuntimeError("missing compression")
    return found


def _tool_logs(kind: str) -> dict[tuple[str, str], str]:
    """Tool log text from constructor records. ledger_text is the log."""
    if not READY_TOOLLOG.is_file():
        raise SystemExit(f"READY_toollog is not available: {READY_TOOLLOG}")
    folder = TOOL_DIRS[kind]
    found = {}
    for set_name in ("shade", "original", "fresh"):
        records = RUNS / set_name / folder / "constructor_records"
        if not records.is_dir():
            continue
        for path in records.glob("*.json"):
            saved = json.loads(path.read_text())
            record = saved.get("record") or {}
            text = record.get("ledger_text") or ""
            if record.get("ledger_error") or not str(text).strip():
                text = ""
            key = saved.get("path") or record.get("path")
            if key:
                found[(set_name, key)] = str(text)
    if not found:
        raise SystemExit(f"no tool logs in {folder}")
    return found


def _event_records() -> dict[tuple[str, str], str]:
    """Identifier event records already written. ledger_text is the record."""
    if not READY_EVENT.is_file():
        raise SystemExit(f"READY_eventrecord is not available: {READY_EVENT}")
    found = {}
    for set_name in ("subset100", "original", "fresh"):
        records = RUNS / set_name / "eventrecord" / "constructor_records"
        if not records.is_dir():
            raise SystemExit(f"no event records at {records}")
        for path in records.glob("*.json"):
            saved = json.loads(path.read_text())
            record = saved.get("record") or {}
            text = record.get("ledger_text") or ""
            if record.get("ledger_error") or not str(text).strip():
                text = ""
            key = saved.get("path") or record.get("path")
            if key:
                found[(set_name, key)] = str(text)
    if not found:
        raise SystemExit("no event records")
    from baselines import eventrecord_300
    for path, text in eventrecord_300.extra_texts().items():
        found[("event300", path)] = text
    return found


def _groups(policies: list[dict], width: int) -> list[list[dict]]:
    if width == 1:
        return [[item] for item in policies]
    if width == len(policies):
        return [policies]
    raise SystemExit(f"width {width} is not 1 or the policy count {len(policies)}")


def _schema(policies: list[dict]) -> dict:
    decision = {"type": "string", "enum": ["violates", "does_not"]}
    suspicion = {"type": "integer"}
    if len(policies) == 1:
        inner = {
            "type": "object",
            "properties": {"suspicion": suspicion, "decision": decision},
            "required": ["suspicion", "decision"],
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
                            "id": {"type": "string", "enum": [item["id"] for item in policies]},
                            "suspicion": suspicion,
                            "decision": decision,
                        },
                        "required": ["id", "suspicion", "decision"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["decisions"],
            "additionalProperties": False,
        }
    return {"type": "json_schema", "json_schema": {"name": "policy_decisions_graded", "strict": True, "schema": inner}}


def _suspicion(value) -> int | None:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and 1 <= value <= 10:
        return value
    return None


def _parse_graded(policies: list[dict], text: str) -> tuple[dict[str, str], dict[str, int | None]]:
    payload = json.loads(text)
    if len(policies) == 1:
        try:
            decisions = _parse_binary(policies, json.dumps({"decision": payload.get("decision")}))
        except (ValueError, TypeError, json.JSONDecodeError):
            decisions = {policies[0]["id"]: "unknown"}
        return decisions, {policies[0]["id"]: _suspicion(payload.get("suspicion"))}
    decisions = {}
    suspicion = {}
    seen = set()
    for item in payload.get("decisions") or []:
        pid = item.get("id")
        if pid in seen or pid not in {policy["id"] for policy in policies}:
            raise ValueError("duplicate or unexpected policy id")
        seen.add(pid)
        decision = item.get("decision")
        decisions[pid] = decision if decision in ("violates", "does_not") else "unknown"
        suspicion[pid] = _suspicion(item.get("suspicion"))
    if set(decisions) != {policy["id"] for policy in policies}:
        raise ValueError("decisions did not cover the stated policies")
    return decisions, suspicion


def _one_width(base_url, model, system_prompt, template, policies, body, width, timeout) -> dict:
    decisions = {}
    suspicion = {}
    prompt_tokens = []
    completion_tokens = []
    error = None
    invalid = 0
    for group in _groups(policies, width):
        response = None
        parsed = None
        last = None
        for attempt in range(3):
            try:
                response = _post(
                    base_url, model, system_prompt,
                    _user_message(template, group, body),
                    _schema(group), MAX_TOKENS[width], timeout,
                )
                parsed = _parse_graded(group, response["text"])
                last = None
                break
            except (OSError, urllib.error.URLError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                last = type(exc).__name__ + ": " + str(exc)
                if "HTTP 400" in last:
                    break
                time.sleep(2 ** attempt)
        prompt_tokens.append(0 if response is None else response["prompt_tokens"])
        completion_tokens.append(0 if response is None else response["completion_tokens"])
        if parsed is None:
            invalid += 1
            error = last
            for item in group:
                decisions[item["id"]] = "unknown"
                suspicion[item["id"]] = None
        else:
            decisions.update(parsed[0])
            suspicion.update(parsed[1])
    return {
        "decisions": decisions,
        "suspicion": suspicion,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
        "invalid_calls": invalid,
    }


def _empty_width(policies: list[dict], reason: str) -> dict:
    return {
        "decisions": {item["id"]: "unknown" for item in policies},
        "suspicion": {item["id"]: None for item in policies},
        "prompt_tokens": [],
        "completion_tokens": [],
        "error": reason,
        "invalid_calls": len(policies),
    }


def _ledgers_for(body: str, surface: str) -> dict:
    system = LEDGER_BODIES.get(body) or ACTION_ONLY.get(body) or STRIPPED.get(body)
    if system is None:
        return {}
    if surface == "calendar":
        return _calendar_ledgers(system)
    return _shade_ledgers(system)


def _calendar_ledgers(system: str) -> dict[tuple[str, str], str]:
    """Reader-facing ledgers already stored for the calendar sets.

    Same text the binary calendar reader used: neutral_ledger after the
    canonical projection. Empty when that projection is missing.
    """
    found = {}
    for set_name in ("original", "fresh"):
        record_dir = RUNS / set_name / system / "reader_records"
        if not record_dir.is_dir():
            raise SystemExit(f"no calendar ledgers at {record_dir}")
        for path in record_dir.glob("*.json"):
            saved = json.loads(path.read_text())
            audit = saved.get("audit") or {}
            text = audit.get("neutral_ledger") or ""
            if audit.get("sanitizer_status") != "canonical_projection":
                text = ""
            found[(set_name, saved["path"])] = text
    return found


def _body(kind: str, row: dict, data_root: Path, ledgers: dict, surface: str, tool_logs: dict | None) -> tuple[str, str | None]:
    try:
        if kind == "raw":
            return _trajectory_text(row, data_root), None
        if kind == "compact":
            return _compact_body(row, data_root), None
        if kind in ("lingua_median", "lingua_x4"):
            return _lingua_body(kind, row["path"]), None
        if kind == "summary":
            text = _summary_body(row["path"])
            if not text.strip():
                return "", "empty summary"
            return text, None
        if kind == "action_summary":
            text = _action_summary_body(row["path"])
            if not text.strip():
                return "", "empty action summary"
            return text, None
        if kind == "action_summary_8192":
            text = _action_summary_8192_body(row["path"])
            if not text.strip():
                return "", "empty action summary 8192"
            return text, None
        if kind in ACTION_ONLY:
            text = _keep_action(ledgers.get(row["path"]) or "")
            if not text.strip():
                return "", "missing action section"
            return text, None
        if kind in STRIPPED:
            text = ledgers.get(row["path"]) or ""
            text = strip_ledger(text)
            if not text.strip():
                return "", "empty after strip"
            return text, None
        if kind in LEDGER_BODIES:
            if surface == "calendar":
                text = ledgers.get((str(row.get("set") or ""), row["path"])) or ""
            else:
                text = ledgers.get(row["path"]) or ""
            if not text.strip():
                return "", "empty ledger"
            return text, None
        if kind == "eventrecord":
            logs = tool_logs or {}
            if surface == "shade":
                text = logs.get(("subset100", row["path"])) or logs.get(("event300", row["path"])) or ""
            else:
                set_name = str(row.get("set") or "")
                text = logs.get((set_name, row["path"])) or ""
            if not text.strip():
                return "", "missing event record"
            return text, None
        if kind in TOOL_DIRS:
            set_name = "shade" if surface == "shade" else str(row.get("set") or "")
            text = (tool_logs or {}).get((set_name, row["path"])) or ""
            if not text.strip():
                return "", "missing tool log"
            return text, None
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return "", type(exc).__name__ + ": " + str(exc)
    raise SystemExit(f"unknown body {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", choices=BODIES, required=True)
    parser.add_argument("--surface", choices=("shade", "calendar"), default="shade")
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--hash-file", type=Path, default=HASH_FILE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL)
    parser.add_argument("--served-model", default=READER_MODEL)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--widths", default="1,8")
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _guard(args.output_dir)
    bench = args.bench or (SHADE_BENCH if args.surface == "shade" else CALENDAR_BENCH)
    base_prompt, template, policies, rows, policy_sha = _load_bench(bench, args.hash_file)
    system_prompt = graded_system(base_prompt)
    system_sha = hashlib.sha256(system_prompt.encode()).hexdigest()
    data_root = _data_root(bench)
    if args.check:
        sample = rows[0]
        ledgers = _ledgers_for(args.body, args.surface)
        if args.body == "eventrecord":
            tool_logs = _event_records()
        elif args.body in TOOL_DIRS:
            tool_logs = _tool_logs(args.body)
        else:
            tool_logs = None
        text, error = _body(args.body, sample, data_root, ledgers, args.surface, tool_logs)
        print(json.dumps({
            "body": args.body,
            "policies": len(policies),
            "chars": len(text),
            "error": error,
            "system_sha256": system_sha,
            "rubric_sha256": RUBRIC_SHA,
        }), flush=True)
        return
    if not args.base_url:
        raise SystemExit("missing --base-url")
    requested = tuple(int(part) for part in args.widths.split(",") if part)
    allowed = {1, len(policies)}
    if not requested or any(width not in allowed for width in requested):
        raise SystemExit(f"widths must be a subset of {sorted(allowed)}")
    need = {str(width) for width in requested}
    ledgers = _ledgers_for(args.body, args.surface)
    rule_sha = rule_sha256() if args.body in STRIPPED else None
    if args.body == "eventrecord":
        tool_logs = _event_records()
    elif args.body in TOOL_DIRS:
        tool_logs = _tool_logs(args.body)
    else:
        tool_logs = None
    keep = None
    if args.subset:
        payload = json.loads(args.subset.read_text())
        keep = set(payload["paths"] if isinstance(payload, dict) else payload)
    selected = [(index, row) for index, row in enumerate(rows, 1) if keep is None or row["path"] in keep]
    if keep is not None and len(selected) != len(keep):
        raise SystemExit(f"subset has {len(keep)} paths but {len(selected)} match the benchmark")
    mine = [item for n, item in enumerate(selected) if n % args.num_chunks == args.chunk]
    if os.environ.get("REQUIRE_BODY") == "1":
        absent = []
        for index, row in mine:
            checkpoint = args.output_dir / "predictions" / f"{index:03d}.json"
            if checkpoint.is_file():
                saved = json.loads(checkpoint.read_text())
                have = set((saved.get("widths") or {}))
                if need <= have and saved.get("path") == row["path"] and saved.get("body") == args.body:
                    continue
            body, skip = _body(args.body, row, data_root, ledgers, args.surface, tool_logs)
            if skip or not str(body).strip():
                absent.append(row["path"])
        if absent:
            raise SystemExit(f"refusing to score {len(absent)} trajectories with an empty body")

    def one(index: int, row: dict) -> str:
        checkpoint = args.output_dir / "predictions" / f"{index:03d}.json"
        saved = None
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if (
                saved.get("system_sha256") != system_sha
                or saved.get("policy_sha256") != policy_sha
                or saved.get("rubric_sha256") != RUBRIC_SHA
                or saved.get("path") != row["path"]
                or saved.get("body") != args.body
                or (rule_sha is not None and saved.get("evaluative_rule_sha256") != rule_sha)
            ):
                raise SystemExit(f"checkpoint mismatch: {checkpoint}")
            missing = need - set((saved.get("widths") or {}))
            if not missing:
                return "resume"
        else:
            missing = set(need)
        body, skip = _body(args.body, row, data_root, ledgers, args.surface, tool_logs)
        widths = dict((saved or {}).get("widths") or {})
        if skip or not body:
            for width in missing:
                widths[width] = _empty_width(policies, skip or "empty record")
        else:
            for width in sorted(missing, key=int):
                widths[width] = _one_width(
                    args.base_url, args.served_model, system_prompt, template,
                    policies, body, int(width), args.timeout_seconds,
                )
        payload = {
            "system_sha256": system_sha,
            "policy_sha256": policy_sha,
            "rubric_sha256": RUBRIC_SHA,
            "path": row["path"],
            "body": args.body,
            "graded": True,
            "widths": widths,
        }
        if rule_sha is not None:
            payload["evaluative_rule_sha256"] = rule_sha
        gb._write_json(checkpoint, payload)
        return "ok"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, index, row): index for index, row in mine}
        for done, future in enumerate(as_completed(futures), 1):
            print(f"[{done}/{len(mine)}] #{futures[future]:03d} {future.result()}", flush=True)


if __name__ == "__main__":
    main()
