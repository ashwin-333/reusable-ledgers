#!/usr/bin/env python3
"""Build tau2 telecom EventRecords from tau2_eventrecord_rules.json (hash-checked).

No model calls. Reads trajectory paths and trajectories only; no policies or labels.
New directory only.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from config import DATA_ROOT, RUNS_DIR

HERE = Path(__file__).resolve().parent
from baselines import eventrecord as be

RULES = HERE / "tau2_eventrecord_rules.json"
RULES_SHA = (HERE / "tau2_eventrecord_rules.sha256").read_text().strip()
EXP = DATA_ROOT / "tau2"
T2 = EXP / "tau2_telecom"
OUT = RUNS_DIR / "mp_controls/oct3/tau2_eventrecord/constructor_records"
REPORT = OUT.parent / "build_report.json"
SAMPLE = OUT.parent / "sample_record.txt"

if hashlib.sha256(RULES.read_bytes()).hexdigest() != RULES_SHA:
    raise SystemExit("rule file does not match its hash")
SPEC = json.loads(RULES.read_text())
KEYS = set(SPEC["telecom_keys"])
STATUS_RES = [re.compile(p) for p in SPEC["status_res"]]
be.BUDGET = int(SPEC["budget_tokens"])
base_identifiers = be.identifiers


def parse(text):
    if not isinstance(text, str):
        return text
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            pass
    return None


def fields(obj, out):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in KEYS:
                for item in value if isinstance(value, list) else [value]:
                    if item is not None and not isinstance(item, (dict, list)):
                        out.append((key, str(item)))
            if isinstance(value, (dict, list)):
                fields(value, out)
    elif isinstance(obj, list):
        for item in obj:
            fields(item, out)


def calls_of(conv):
    """(name, args_text, result_text, error) per call; positional pairing with the following results."""
    found = []
    i = 0
    while i < len(conv):
        turn = conv[i]
        calls = turn.get("calls") or []
        if not calls:
            i += 1
            continue
        result_role = "user_tool" if turn.get("role") == "user" else "tool"
        results = []
        j = i + 1
        while j < len(conv) and conv[j].get("role") == result_role:
            results.append(conv[j])
            j += 1
        paired = []
        for k, call in enumerate(calls):
            name = str(call.get("name") or "unknown")
            if call.get("actor") == "user":
                name += " (user action)"
            args = call.get("arguments") or {}
            args_text = args if isinstance(args, str) else json.dumps(args, sort_keys=True)
            res = results[k] if k < len(results) else None
            content = (res.get("content") or "") if res else ""
            error = bool(res and (res.get("error") is True or content.lstrip().startswith("Error")))
            paired.append([name, args_text, content, error])
        if len(results) > len(calls) and paired:
            extra = results[len(calls):]
            paired[-1][2] = (paired[-1][2] + "\n" + "\n".join(r.get("content") or "" for r in extra)).strip()
            paired[-1][3] = paired[-1][3] or any(r.get("error") is True for r in extra)
        found.extend(tuple(p) for p in paired)
        i = j if j > i + 1 else i + 1
    return found


ERRORS: dict[tuple[str, str], bool] = {}


def identifiers(call_text: str, result: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if ERRORS.get((call_text, result)):
        out.append(("status", "error"))
    fields(parse(call_text), out)
    obj = parse(result)
    if obj is not None and isinstance(obj, (dict, list)):
        fields(obj, out)
    else:
        for pattern in STATUS_RES:
            out.extend(("status", " ".join(m.group(0).split())) for m in pattern.finditer(result))
    out.extend(base_identifiers(call_text, result))
    seen, kept = set(), []
    for typ, value in out:
        key = be._key(typ, value)
        if key in seen:
            continue
        seen.add(key)
        kept.append((typ, value))
    return kept


be.identifiers = identifiers


def main() -> None:
    if OUT.exists() or REPORT.exists():
        raise SystemExit("refusing to overwrite")
    ntok = be.load_tokenizer()
    paths = [row["path"] for row in json.loads((T2 / "manifest.json").read_text())]
    if len(paths) != 300 or len(set(paths)) != 300:
        raise SystemExit("expected 300 unique paths")
    OUT.mkdir(parents=True)
    config = hashlib.sha256((be.rule_hash() + RULES_SHA).encode()).hexdigest()
    metas = []
    for index, rel in enumerate(paths):
        conv = json.loads((T2 / "trajectories" / rel).read_text()).get("conversation") or []
        calls = calls_of(conv)
        ERRORS.clear()
        for name, args_text, content, error in calls:
            if error:
                ERRORS[(args_text, content)] = True
        triples = [(name, args_text, content) for name, args_text, content, _e in calls]
        text, meta = be.build_record(triples, ntok)
        meta["n_user_actions"] = sum(name.endswith("(user action)") for name, *_ in calls)
        metas.append(meta)
        (OUT / f"{index:03d}.json").write_text(json.dumps({
            "config_sha256": config,
            "rules_sha256": RULES_SHA,
            "path": rel,
            "record": {"path": rel, "ledger_text": text, "ledger_error": None, "representation": "eventrecord",
                       "n_calls": meta["n_calls"], "token_len_full": meta["token_len_full"],
                       "token_len": meta["token_len"], "budget_tokens": be.BUDGET, "trim": meta["trim"]},
        }) + "\n")
        if index == 0:
            SAMPLE.write_text(f"path: {rel}\ntokens: {meta['token_len']} (full {meta['token_len_full']}, trim {meta['trim']})\n\n{text}\n")
    report = {
        "rules_sha256": RULES_SHA,
        "config_sha256": config,
        "budget_tokens": be.BUDGET,
        "dir": str(OUT),
        "n": len(metas),
        "full": be.dist([m["token_len_full"] for m in metas]),
        "final": be.dist([m["token_len"] for m in metas]),
        "trim": {s: sum(m["trim"] == s for m in metas) for s in ("none", "repeat", "even")},
        "over_budget": sum(m["over_budget"] for m in metas),
        "empty": sum(m["n_calls"] == 0 for m in metas),
        "calls_median": be.dist([m["n_calls"] for m in metas])["median"],
        "user_actions_median": be.dist([m["n_user_actions"] for m in metas])["median"],
    }
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
