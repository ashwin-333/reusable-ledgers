"""Identifier-preserving event records. No model calls.

Reads trajectory paths only. Does not read policy text or labels.
For each tool call, in order: the tool name and identifiers matched by the
fixed regexes below, deduplicated within the call. If the record is longer
than 1174 Gemma tokens, repeated identifiers are dropped (first call keeps
each one), then each call's identifiers are trimmed to an equal token share.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit
from config import DATA_ROOT, READER_MODEL_LARGE, RUNS_DIR, SUBSET_FILE, tokenizer_file

BUDGET = 1174
SFT = DATA_ROOT
OUT_ROOT = RUNS_DIR
READY = SFT / "READY_eventrecord"
SUMMARY = SFT / "eventrecord_summary.json"

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\bhttps?://[^\s<>\"')\]]+", re.I)
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
LABELED_ID_RE = re.compile(
    r"(?i)\b(?:iban|account(?:\s+number)?|acct|routing(?:\s+number)?|"
    r"confirmation(?:\s+(?:code|number))?|booking(?:\s+id)?|"
    r"reservation(?:\s+id)?|ticket(?:\s+number)?|invoice(?:\s+number)?|"
    r"mrn|ssn)\b\s*[#:=]?\s*([A-Za-z0-9][A-Za-z0-9-]{4,})"
)
SECRET_RE = re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]+\b")
AMOUNT_RE = re.compile(
    r"[$€£]\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"
    r"|\b(?:USD|EUR|GBP|CAD|MXN|AUD)\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"
    r"|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\s*(?:USD|EUR|GBP|CAD|MXN|AUD|dollars|euros)\b",
    re.I,
)
PATH_RE = re.compile(r"(?:(?:~)?/(?:[\w.@+-]+/)*[\w.@+-]+)")
FILE_RE = re.compile(
    r"\b[\w.-]+\.(?:csv|xlsx|xls|pdf|json|txt|md|png|jpg|jpeg|docx|html|py|xml)\b",
    re.I,
)
DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
)
NAME_RES = (
    re.compile(r"(?i)\bName:\s*([A-Z][a-z]+(?:[ '-][A-Z][a-z]+)*)"),
    re.compile(r"(?im)^\s*name:\s*([A-Z][a-z]+(?:[ '-][A-Z][a-z]+)*)"),
    re.compile(r"(?i)\bContact\s+([A-Z][a-z]+(?:[ '-][A-Z][a-z]+)*)\s+added"),
    re.compile(r"(?i)\bAdded contact:\s*([A-Z][a-z]+(?:[ '-][A-Z][a-z]+)*)"),
)
RECIPIENTS_RE = re.compile(r"(?i)\bRecipients:\s*\[([^\]]*)\]")
QUOTED_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")
DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|uk|de|fr|nl|app|dev|ai|me|info|biz)\b",
    re.I,
)
# Longer / more specific types win ties. Domain is last so an email or URL
# consumes its own host; the host is then added as a domain identifier.
SCAN = (
    ("email", EMAIL_RE, 0),
    ("url", URL_RE, 0),
    ("account", IBAN_RE, 0),
    ("account", LABELED_ID_RE, 1),
    ("account", SECRET_RE, 0),
    ("amount", AMOUNT_RE, 0),
    ("path", PATH_RE, 0),
    ("path", FILE_RE, 0),
    ("date", DATE_RE, 0),
    ("domain", DOMAIN_RE, 0),
)


def rule_hash() -> str:
    blob = [f"budget={BUDGET}", "representation=eventrecord"]
    for typ, pattern, group in SCAN:
        blob.append(f"{typ}\t{group}\t{pattern.pattern}")
    for pattern in NAME_RES:
        blob.append(f"name\t1\t{pattern.pattern}")
    blob.append(f"name\trecipients\t{RECIPIENTS_RE.pattern}")
    return hashlib.sha256("\n".join(blob).encode()).hexdigest()


def paths_only(path: Path) -> list[str]:
    """Keep trajectory paths. Discard every other manifest field."""
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("paths"), list):
        return [str(item) for item in payload["paths"]]
    rows = payload if isinstance(payload, list) else payload.get("entries") or []
    out = []
    for row in rows:
        if isinstance(row, dict):
            out.append(str(row["path"]))
        else:
            out.append(str(row))
    return out


def tool_calls(conv: list[dict]) -> list[tuple[str, str, str]]:
    """(name, call text, result) in order. end_task with no result is kept."""
    found = []
    i = 0
    while i < len(conv):
        msg = conv[i]
        raw = msg.get("tool_calls")
        names: list[tuple[str, str]] = []
        if msg.get("role") == "assistant" and isinstance(raw, str):
            lines = raw.splitlines()
            idx = 0
            while idx < len(lines):
                match = re.match(r"^- ([A-Za-z0-9_]+)\s*$", lines[idx])
                if not match:
                    idx += 1
                    continue
                name = match.group(1)
                blob = [lines[idx]]
                idx += 1
                while idx < len(lines) and not re.match(r"^- [A-Za-z0-9_]+", lines[idx]):
                    blob.append(lines[idx])
                    idx += 1
                names.append((name, "\n".join(blob)))
        elif msg.get("role") == "assistant" and isinstance(raw, list):
            for call in raw:
                if isinstance(call, dict):
                    name = call.get("name") or (call.get("function") or {}).get("name") or "unknown"
                    args = call.get("arguments") or (call.get("function") or {}).get("arguments") or ""
                    if not isinstance(args, str):
                        args = json.dumps(args, sort_keys=True)
                    names.append((str(name), args))
                elif isinstance(call, str) and call.strip():
                    names.append((call.strip(), call.strip()))
        if not names:
            i += 1
            continue
        results = []
        j = i + 1
        while j < len(conv) and conv[j].get("role") == "tool":
            results.append(conv[j].get("content") or "")
            j += 1
        r = 0
        paired = []
        for name, blob in names:
            if name == "end_task" and r >= len(results):
                paired.append((name, blob, ""))
                continue
            if r < len(results):
                paired.append((name, blob, results[r]))
                r += 1
            else:
                paired.append((name, blob, ""))
        if r < len(results) and paired:
            name, blob, result = paired[-1]
            extra = "\n".join(results[r:])
            paired[-1] = (name, blob, (result + "\n" + extra).strip())
        found.extend(paired)
        i = j
    return found


def _clean(value: str) -> str:
    return value.strip().strip(".,;:)]}>\"'")


def _key(typ: str, value: str) -> tuple[str, str]:
    text = " ".join(value.split())
    if typ in ("email", "domain", "name"):
        text = text.casefold()
    return typ, text


def _host(url: str) -> str:
    host = urlsplit(url if "://" in url else "http://" + url).hostname or ""
    return host.lower().rstrip(".")


def identifiers(call_text: str, result: str) -> list[tuple[str, str]]:
    text = call_text + "\n" + result
    spans: list[tuple[int, int, str, str]] = []
    for typ, pattern, group in SCAN:
        for match in pattern.finditer(text):
            raw = match.group(group) if group else match.group(0)
            value = _clean(raw)
            if len(value) < 2:
                continue
            spans.append((match.start(), match.end(), typ, value))
    for pattern in NAME_RES:
        for match in pattern.finditer(text):
            value = _clean(match.group(1))
            if re.search(r"[A-Za-z]", value):
                spans.append((match.start(1), match.end(1), "name", value))
    for match in RECIPIENTS_RE.finditer(text):
        for quoted in QUOTED_RE.finditer(match.group(1)):
            value = _clean(quoted.group(1) or quoted.group(2) or "")
            if value and "@" not in value and re.search(r"[A-Za-z]", value):
                spans.append((quoted.start(), match.start() + quoted.end(), "name", value))
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[str, str]] = []
    occupied_until = -1
    seen = set()
    for start, end, typ, value in spans:
        if start < occupied_until:
            continue
        occupied_until = end
        extras = [(typ, value)]
        if typ == "email" and "@" in value:
            extras.append(("domain", value.split("@", 1)[1].lower()))
        elif typ == "url":
            host = _host(value)
            if host and "." in host:
                extras.append(("domain", host))
        for extra_type, extra_value in extras:
            identity = _key(extra_type, extra_value)
            if identity in seen:
                continue
            seen.add(identity)
            kept.append((extra_type, " ".join(extra_value.split())))
    return kept


def render(calls: list[tuple[str, list[tuple[str, str]]]]) -> str:
    blocks = []
    for index, (name, idents) in enumerate(calls):
        lines = [f"[{index}] {name}"]
        lines.extend(f"{typ}:{value}" for typ, value in idents)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def drop_repeats(
    calls: list[tuple[str, list[tuple[str, str]]]],
) -> list[tuple[str, list[tuple[str, str]]]]:
    seen = set()
    out = []
    for name, idents in calls:
        kept = []
        for typ, value in idents:
            identity = _key(typ, value)
            if identity in seen:
                continue
            seen.add(identity)
            kept.append((typ, value))
        out.append((name, kept))
    return out


def _ident_text(idents: list[tuple[str, str]]) -> str:
    return "\n".join(f"{typ}:{value}" for typ, value in idents)


def trim_even(calls, ntok) -> list[tuple[str, list[tuple[str, str]]]]:
    skeleton = render([(name, []) for name, _idents in calls])
    room = BUDGET - ntok(skeleton)
    share = max(0, room // len(calls)) if calls else 0
    trimmed = []
    for name, idents in calls:
        kept = []
        for ident in idents:
            trial = kept + [ident]
            if ntok(_ident_text(trial)) <= share:
                kept = trial
            else:
                break
        trimmed.append((name, kept))
    while ntok(render(trimmed)) > BUDGET and any(idents for _name, idents in trimmed):
        lengths = []
        for index, (_name, idents) in enumerate(trimmed):
            if idents:
                lengths.append((ntok(_ident_text(idents)), index))
        _length, index = max(lengths)
        name, idents = trimmed[index]
        trimmed[index] = (name, idents[:-1])
    return trimmed


def build_record(calls: list[tuple[str, str, str]], ntok) -> tuple[str, dict]:
    extracted = [(name, identifiers(blob, result)) for name, blob, result in calls]
    full_text = render(extracted)
    full_len = ntok(full_text)
    stage = "none"
    current = extracted
    if full_len > BUDGET:
        current = drop_repeats(extracted)
        stage = "repeat"
        if ntok(render(current)) > BUDGET:
            current = trim_even(current, ntok)
            stage = "even"
    text = render(current)
    final_len = ntok(text)
    return text, {
        "n_calls": len(calls),
        "token_len_full": full_len,
        "token_len": final_len,
        "trim": stage,
        "over_budget": final_len > BUDGET,
    }


def dist(values: list[int]) -> dict:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return {"n": 0}

    def at(q: float) -> int:
        return ordered[min(n - 1, int(q * n))]

    return {
        "n": n,
        "min": ordered[0],
        "p10": at(0.10),
        "median": ordered[n // 2],
        "p90": at(0.90),
        "max": ordered[-1],
        "mean": round(sum(ordered) / n, 1),
    }


def load_tokenizer():
    from tokenizers import Tokenizer

    found = tokenizer_file(READER_MODEL_LARGE)
    if found is None:
        raise SystemExit("missing Gemma tokenizer")
    tokenizer = Tokenizer.from_file(str(found))
    load_tokenizer.found = found

    def ntok(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return ntok


def write_set(name: str, root: Path, rel_paths: list[str], ntok, config: str) -> dict:
    directory = OUT_ROOT / name / "eventrecord" / "constructor_records"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, rel in enumerate(rel_paths):
        document = json.loads((root / rel).read_text())
        calls = tool_calls(document.get("conversation") or [])
        text, meta = build_record(calls, ntok)
        payload = {
            "config_sha256": config,
            "path": rel,
            "record": {
                "path": rel,
                "ledger_text": text,
                "ledger_error": None,
                "representation": "eventrecord",
                "n_calls": meta["n_calls"],
                "token_len_full": meta["token_len_full"],
                "token_len": meta["token_len"],
                "budget_tokens": BUDGET,
                "trim": meta["trim"],
            },
        }
        (directory / f"{index:03d}.json").write_text(json.dumps(payload) + "\n")
        rows.append(meta)
    return {
        "dir": str(directory),
        "n": len(rows),
        "full": dist([row["token_len_full"] for row in rows]),
        "final": dist([row["token_len"] for row in rows]),
        "trim": {
            "none": sum(1 for row in rows if row["trim"] == "none"),
            "repeat": sum(1 for row in rows if row["trim"] == "repeat"),
            "even": sum(1 for row in rows if row["trim"] == "even"),
        },
        "over_budget": sum(1 for row in rows if row["over_budget"]),
        "calls_median": dist([row["n_calls"] for row in rows])["median"],
    }


def main() -> None:
    ntok = load_tokenizer()
    config = rule_hash()
    sets = [
        (
            "subset100",
            SFT / "shade_real",
            paths_only(SUBSET_FILE),
        ),
        (
            "original",
            DATA_ROOT / "original_trajectories",
            paths_only(SFT / "manifests/manifest_mt_heldout.json"),
        ),
        (
            "fresh",
            SFT / "fresh_heldout_data",
            paths_only(SFT / "manifests/manifest_fresh_heldout.json"),
        ),
    ]
    expected = {"subset100": 100, "original": 96, "fresh": 96}
    report = {"budget_tokens": BUDGET, "config_sha256": config, "tokenizer": str(load_tokenizer.found), "sets": {}}
    pooled_full = []
    pooled_final = []
    for name, root, rel_paths in sets:
        if len(rel_paths) != expected[name]:
            raise SystemExit(f"{name} has {len(rel_paths)} paths, expected {expected[name]}")
        if len(set(rel_paths)) != len(rel_paths):
            raise SystemExit(f"{name} has duplicate paths")
        summary = write_set(name, root, rel_paths, ntok, config)
        report["sets"][name] = summary
        # Re-read lengths from the files just written so the report matches disk.
        directory = Path(summary["dir"])
        full, final = [], []
        for path in sorted(directory.glob("*.json")):
            saved = json.loads(path.read_text())
            record = saved["record"]
            if "is_benign" in saved or "is_benign" in record or "label" in record:
                raise SystemExit(f"label field written: {path}")
            full.append(record["token_len_full"])
            final.append(record["token_len"])
        pooled_full.extend(full)
        pooled_final.extend(final)
    report["pooled"] = {"full": dist(pooled_full), "final": dist(pooled_final)}
    SUMMARY.write_text(json.dumps(report, indent=1) + "\n")
    lines = [
        "READY eventrecord",
        "created: 2026-09-30",
        "no model calls",
        "policies and labels were not read",
        f"budget_tokens: {BUDGET}",
        f"config_sha256: {config}",
        f"tokenizer: {load_tokenizer.found}",
        "rule: per-call dedup; if over budget, drop identifiers already kept on an earlier call; then trim each call to an equal identifier-token share",
        f"summary: {SUMMARY}",
    ]
    for name, summary in report["sets"].items():
        final = summary["final"]
        full = summary["full"]
        lines.append(
            f"{name}: n={summary['n']} dir={summary['dir']} "
            f"full min/p10/median/p90/max={full['min']}/{full['p10']}/{full['median']}/{full['p90']}/{full['max']} "
            f"final min/p10/median/p90/max={final['min']}/{final['p10']}/{final['median']}/{final['p90']}/{final['max']} "
            f"trim none/repeat/even={summary['trim']['none']}/{summary['trim']['repeat']}/{summary['trim']['even']} "
            f"over_budget={summary['over_budget']} calls_median={summary['calls_median']}"
        )
    pooled = report["pooled"]
    lines.append(
        "pooled final min/p10/median/p90/max="
        f"{pooled['final']['min']}/{pooled['final']['p10']}/{pooled['final']['median']}/{pooled['final']['p90']}/{pooled['final']['max']}"
    )
    READY.write_text("\n".join(lines) + "\n")
    print(READY.read_text())


def _self_test() -> None:
    def ntok(text: str) -> int:
        return len(text.split())

    sample = "Email user@Gmail.com about https://files.example.com/a/b.csv on 2026-03-16. Total cost: $175.89. IBAN GB29NWBK60161331926819. Name: Sarah Chen"
    found = identifiers("- send_email", sample)
    kinds = {typ for typ, _value in found}
    assert "email" in kinds and "domain" in kinds and "url" in kinds
    assert "amount" in kinds and "account" in kinds and "date" in kinds and "name" in kinds
    assert ("domain", "gmail.com") in [(typ, value.casefold()) for typ, value in found]
    # A prose name with no contact field is not extracted.
    prose = identifiers("- send_email", "Sarah Chen sent the note.")
    assert not any(typ == "name" for typ, _value in prose)
    calls = [
        ("send_email", "- send_email", "to user@gmail.com"),
        ("send_email", "- send_email", "again user@gmail.com"),
    ]
    global BUDGET
    saved = BUDGET
    try:
        BUDGET = 100000
        text, meta = build_record(calls, ntok)
        assert meta["trim"] == "none", meta
        assert text.count("email:user@gmail.com") == 2
        extracted = [(name, identifiers(blob, result)) for name, blob, result in calls]
        repeated = drop_repeats(extracted)
        full_n = ntok(render(extracted))
        repeat_n = ntok(render(repeated))
        assert repeat_n < full_n
        BUDGET = repeat_n
        text, meta = build_record(calls, ntok)
        assert meta["trim"] == "repeat", meta
        assert text.count("email:user@gmail.com") == 1
        BUDGET = repeat_n - 1
        text, meta = build_record(calls, ntok)
        assert meta["trim"] == "even", meta
        assert "email:" not in text
    finally:
        BUDGET = saved
    print("self-test ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
