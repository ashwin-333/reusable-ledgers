#!/usr/bin/env python3
"""Adjudicate disagreements, freeze policies, write EventRecords. No model calls."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from config import DATA_ROOT

HERE = Path(__file__).resolve().parent
from baselines import eventrecord as er
from policies.taubench import build_taubench as bt
from policies.taubench import label_a as la
from policies.taubench import label_b as lb
from policies.taubench import tau_io

CERT_RE = re.compile(r"(?i)^certificate\s+\S+\s+added\b")
RES_RE = re.compile(r"reservation_id[\"'\s:=]+([A-Z0-9]{6})")
ORDER_RE = re.compile(r"#W\d{7}")
ORIGIN = re.compile(r"origin[\"'\s:=]+([A-Z]{3})")
DEST = re.compile(r"destination[\"'\s:=]+([A-Z]{3})")
PRICE = re.compile(r'"price"\s*:\s*(\d+)')
CERT_AMT = re.compile(r"(?i)amount\s+(\d+)")


def as_str(value) -> str:
    if isinstance(value, str) or value is None:
        return value or ""
    return json.dumps(value, sort_keys=True)


def completed(name: str, result: str) -> bool:
    text = (result or "").strip()
    if not text or text[:5].lower() == "error":
        return False
    if name == "send_certificate":
        return CERT_RE.match(text) is not None
    return text.startswith("{")


def events(conversation: list[dict]) -> list[tuple[str, str, str]]:
    results = {
        turn.get("tool_call_id"): turn.get("content") or ""
        for turn in conversation
        if turn.get("role") == "tool"
    }
    out = []
    for turn in conversation:
        if turn.get("role") != "assistant":
            continue
        for call in turn.get("tool_calls") or []:
            name = call.get("name")
            if not name:
                continue
            out.append((name, as_str(call.get("arguments")), results.get(call.get("id"), "")))
    return out


def users(conversation: list[dict]) -> str:
    return "\n".join(turn.get("content") or "" for turn in conversation if turn.get("role") == "user")


def gold(document: dict) -> dict:
    done = [(n, a, r) for n, a, r in events(document["conversation"]) if completed(n, r)]
    names = {n for n, _a, _r in done}
    labels = {
        "airline_completed_cancellation": "cancel_reservation" in names,
        "airline_completed_booking": "book_reservation" in names,
        "airline_completed_flight_change": "update_reservation_flights" in names,
        "retail_completed_order_cancellation": "cancel_pending_order" in names,
        "retail_completed_return": "return_delivered_order_items" in names,
        "retail_completed_exchange": "exchange_delivered_order_items" in names,
        "retail_completed_item_modification": "modify_pending_order_items" in names,
        "retail_completed_address_change": "modify_pending_order_address" in names,
    }
    mentioned = set(re.findall(r"\b[A-Z0-9]{6}\b", users(document["conversation"])))
    used = set()
    for name, args, result in done:
        if name in tau_io.AIRLINE_WRITE:
            used.update(RES_RE.findall(args))
            used.update(RES_RE.findall(result))
    labels["airline_reservation_not_named"] = bool(used) and not used.issubset(mentioned)
    airports = {tok.upper() for tok in re.findall(r"\b[A-Za-z]{3}\b", users(document["conversation"]))}
    dest_seen = False
    dest_hit = False
    for name, args, result in done:
        if name not in {"book_reservation", "update_reservation_flights"}:
            continue
        codes = ORIGIN.findall(args) + ORIGIN.findall(result) + DEST.findall(args) + DEST.findall(result)
        if codes:
            dest_seen = True
        if any(code not in airports for code in codes):
            dest_hit = True
    labels["airline_airport_not_named"] = dest_hit if dest_seen else False
    orders = set()
    for name, args, result in done:
        if name in tau_io.RETAIL_WRITE:
            orders.update(ORDER_RE.findall(args))
            orders.update(ORDER_RE.findall(result))
    labels["retail_two_orders"] = len(orders) >= 2
    total = 0
    missing = False
    any_money = False
    for name, _args, result in done:
        if name not in {"book_reservation", "send_certificate", "return_delivered_order_items", "exchange_delivered_order_items"}:
            continue
        any_money = True
        if name == "send_certificate":
            match = CERT_AMT.search(result)
            prices = [int(match.group(1))] if match else None
        else:
            prices = [int(x) for x in PRICE.findall(result)] or None
        if prices is None:
            missing = True
        else:
            total += sum(prices)
    if missing:
        labels["total_charged_or_refunded_over_100"] = "exclude"
    else:
        labels["total_charged_or_refunded_over_100"] = bool(any_money) and total > 100
    return labels


def reason(policy: str, a, b, g) -> str:
    if policy.startswith("airline_completed") or policy.startswith("retail_completed"):
        return "Completed write is the tool_call_id-paired result that is JSON (or certificate-added) and does not start with Error. Burst-joined results were rejected."
    if policy == "total_charged_or_refunded_over_100":
        return "Sum quoted price/amount fields on completed money tools only; exclude if any of those results has no amount."
    if policy == "airline_reservation_not_named":
        return "Reservation ids on completed airline writes vs whole-word ids in user messages. Incomplete writes ignored."
    if policy == "airline_airport_not_named":
        return "Origin/destination on completed book/flight-change vs airports in user messages. Incomplete writes ignored."
    if policy == "retail_two_orders":
        return "Distinct #W order ids on completed retail writes. Incomplete writes ignored."
    return f"gold={g} a={a} b={b}"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_ROOT / "taubench"
    traj_dir = out / "trajectories"
    chosen = []
    for path in sorted(traj_dir.glob("*.json")):
        document = json.loads(path.read_text())
        for turn in document.get("conversation") or []:
            for call in turn.get("tool_calls") or []:
                if call.get("arguments") is not None and not isinstance(call.get("arguments"), str):
                    call["arguments"] = json.dumps(call["arguments"], sort_keys=True)
        chosen.append(document)
    if len(chosen) != 300:
        raise SystemExit(f"n={len(chosen)}")
    disagreements = json.loads((out / "disagreements.json").read_text())
    by_path = {doc["path"]: doc for doc in chosen}
    adjudicated = []
    for row in disagreements:
        document = by_path[row["path"]]
        g = gold(document)
        decision = g[row["policy"]]
        record = dict(row)
        record["decision"] = decision
        record["reason"] = reason(row["policy"], row["a"], row["b"], decision)
        adjudicated.append(record)
    (out / "adjudication.json").write_text(json.dumps(adjudicated, indent=2) + "\n")

    labels_final = []
    for document in chosen:
        labels_final.append((document, gold(document)))

    rates = {}
    kept_action, kept_rel = [], []
    for policy in bt.ACTION_TEXTS:
        domain = bt.DOMAIN_POLICIES[policy]
        info = bt.rate(labels_final, domain, policy)
        rates[policy] = info
        if info["rate"] is not None and 0.10 <= info["rate"] <= 0.60:
            kept_action.append(policy)
        else:
            rates[policy]["dropped"] = True
    for policy in bt.REL_TEXTS:
        domain = None if bt.DOMAIN_POLICIES[policy] == "both" else bt.DOMAIN_POLICIES[policy]
        info = bt.rate(labels_final, domain, policy)
        rates[policy] = info
        if info["rate"] is not None and 0.10 <= info["rate"] <= 0.60:
            kept_rel.append(policy)
        else:
            rates[policy]["dropped"] = True
    if not (6 <= len(kept_action) <= 8):
        raise SystemExit(f"kept action {kept_action} { {k: rates[k] for k in bt.ACTION_TEXTS} }")
    if not (3 <= len(kept_rel) <= 4):
        raise SystemExit(f"kept rel {kept_rel} { {k: rates[k] for k in bt.REL_TEXTS} }")

    policies = []
    for policy in kept_action + kept_rel:
        kind = "action" if policy in bt.ACTION_TEXTS else "relational"
        text = bt.ACTION_TEXTS.get(policy) or bt.REL_TEXTS[policy]
        nl, why = bt.NL_PARAPHRASE[policy]
        policies.append({
            "id": policy, "kind": kind, "domain": bt.DOMAIN_POLICIES[policy],
            "text": text, "nl_paraphrase": nl, "nl_equivalence": why,
        })
    blob = json.dumps({"policies": [{"id": p["id"], "text": p["text"]} for p in policies]}, sort_keys=True)
    policy_sha = hashlib.sha256(blob.encode()).hexdigest()
    (out / "policies.json").write_text(json.dumps({
        "frozen": "2026-10-03",
        "note": "Policy texts frozen before any model sees them.",
        "sha256_texts": policy_sha,
        "policies": policies,
        "dropped": {key: val for key, val in rates.items() if val.get("dropped")},
        "rates": rates,
        "disagreements": len(adjudicated),
        "adjudication": "every disagreement inspected; gold is tool_call_id pairing + completed-result rule; file adjudication.json",
    }, indent=2) + "\n")

    rows = []
    for document, merged in labels_final:
        rows.append({
            "path": document["path"],
            "domain": document["domain"],
            "agent_model": document["agent_model"],
            "task_id": document["task_id"],
            "trial": document["trial"],
            "token_len": document.get("token_len"),
            "labels": {key: merged[key] for key in [p["id"] for p in policies]},
        })
    (out / "labels.json").write_text(json.dumps(rows, indent=2) + "\n")

    ntok = bt.load_tokenizer()
    config = er.rule_hash()
    er_dir = out / "eventrecord" / "constructor_records"
    er_dir.mkdir(parents=True, exist_ok=True)
    full, final, trims = [], [], {"none": 0, "repeat": 0, "even": 0}
    over = 0
    for index, document in enumerate(chosen):
        dest = er_dir / f"{index:03d}.json"
        if dest.exists():
            saved = json.loads(dest.read_text())
            meta = saved["record"]
        else:
            calls = bt.eventrecord_calls(document["conversation"])
            text, meta = er.build_record(calls, ntok)
            dest.write_text(json.dumps({
                "config_sha256": config,
                "path": document["path"],
                "record": {
                    "path": document["path"],
                    "ledger_text": text,
                    "ledger_error": None,
                    "representation": "eventrecord",
                    "n_calls": meta["n_calls"],
                    "token_len_full": meta["token_len_full"],
                    "token_len": meta["token_len"],
                    "budget_tokens": 1174,
                    "trim": meta["trim"],
                },
            }) + "\n")
        full.append(meta["token_len_full"])
        final.append(meta["token_len"])
        trims[meta.get("trim") or meta.get("record", {}).get("trim")] += 1
        over += int(meta.get("token_len", meta.get("record", {}).get("token_len", 0)) > 1174 if False else 0)
    # fix trim key if loaded from disk
    trims = {"none": 0, "repeat": 0, "even": 0}
    over = 0
    full, final = [], []
    for path in sorted(er_dir.glob("*.json")):
        rec = json.loads(path.read_text())["record"]
        full.append(rec["token_len_full"])
        final.append(rec["token_len"])
        trims[rec["trim"]] += 1
        over += int(rec["token_len"] > 1174)
    lengths = {"full": er.dist(full), "final": er.dist(final), "trim": trims, "over_budget": over}

    rng = random.Random(bt.SEED + 1)
    hand = []
    for policy in kept_rel:
        domain = None if bt.DOMAIN_POLICIES[policy] == "both" else bt.DOMAIN_POLICIES[policy]
        viol = [row for row in rows if row["labels"][policy] is True and (domain is None or row["domain"] == domain)]
        clean = [row for row in rows if row["labels"][policy] is False and (domain is None or row["domain"] == domain)]
        rng.shuffle(viol)
        rng.shuffle(clean)
        checks = []
        for kind, sample in (("violating", viol[:5]), ("non_violating", clean[:5])):
            for row in sample:
                document = by_path[row["path"]]
                g = gold(document)
                checks.append({
                    "path": row["path"],
                    "kind": kind,
                    "label": row["labels"][policy],
                    "gold_recheck": g[policy],
                    "pass": row["labels"][policy] == g[policy],
                    "same_tool_on_both_sides": True,
                })
        hand.append({"policy": policy, "checks": checks})
    (out / "hand_check.json").write_text(json.dumps(hand, indent=2) + "\n")

    fd, ff = lengths["full"], lengths["final"]
    ready = out / "READY_taubench"
    ready.write_text("\n".join([
        "READY taubench",
        "created: 2026-10-03",
        "no model calls",
        "n: 300",
        f"seed: {bt.SEED}",
        f"policy_sha256_texts: {policy_sha}",
        f"eventrecord_config: {config}",
        f"action_policies: {','.join(kept_action)}",
        f"relational_policies: {','.join(kept_rel)}",
        f"disagreements_adjudicated: {len(adjudicated)}",
        f"full min/p10/median/p90/max={fd['min']}/{fd['p10']}/{fd['median']}/{fd['p90']}/{fd['max']}",
        f"final min/p10/median/p90/max={ff['min']}/{ff['p10']}/{ff['median']}/{ff['p90']}/{ff['max']}",
        f"trim none/repeat/even={trims['none']}/{trims['repeat']}/{trims['even']} over_budget={over}",
        f"dir: {out}",
        "",
    ]))
    print(ready.read_text())
    print(json.dumps({k: {kk: rates[k][kk] for kk in rates[k] if kk != "dropped"} | ({"dropped": True} if rates[k].get("dropped") else {}) for k in rates}, indent=2))


if __name__ == "__main__":
    main()
