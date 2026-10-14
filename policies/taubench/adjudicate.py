#!/usr/bin/env python3
"""Adjudicate in-domain label disagreements. No model calls."""
from __future__ import annotations

import json
import re
from pathlib import Path
from config import DATA_ROOT, OUT_ROOT, RUNS_DIR

SFT = DATA_ROOT
TAU = SFT / "taubench/trajectories"
ACTION = RUNS_DIR / "mp_controls/oct3/benches/tau_action/multi_policy_benchmark.json"
REL = RUNS_DIR / "mp_controls/oct3/benches/tau_relational/multi_policy_benchmark.json"
FIXED = SFT / "taubench_rel_fixed/labels.json"
OUT = OUT_ROOT / "label_audit" / "adjudication.json"

CERT_OK = re.compile(r"(?i)^certificate\s+\S+\s+added\b")
CERT_AMT = re.compile(r"(?i)\bamount\s+([0-9]+(?:\.[0-9]+)?)")
RES_RE = re.compile(r"\b[A-Z0-9]{6}\b")
AIRPORT = re.compile(r"\b([A-Za-z]{3})\b")
ORDER_ID = re.compile(r"^#W\d{7}$")
ATOMIC = {
    "airline_completed_cancellation": ("airline", "cancel_reservation"),
    "airline_completed_booking": ("airline", "book_reservation"),
    "airline_completed_flight_change": ("airline", "update_reservation_flights"),
    "retail_completed_order_cancellation": ("retail", "cancel_pending_order"),
    "retail_completed_return": ("retail", "return_delivered_order_items"),
    "retail_completed_exchange": ("retail", "exchange_delivered_order_items"),
    "retail_completed_item_modification": ("retail", "modify_pending_order_items"),
}
RES_WRITES = {
    "cancel_reservation", "update_reservation_flights",
    "update_reservation_baggages", "update_reservation_passengers",
}
MONEY = {
    "book_reservation", "send_certificate",
    "return_delivered_order_items", "exchange_delivered_order_items",
}


def _loads_calls(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw if isinstance(raw, list) else None


def zip_events(conv):
    found = []
    i = 0
    while i < len(conv):
        turn = conv[i]
        calls = _loads_calls(turn.get("tool_calls"))
        if turn.get("role") != "assistant" or not calls:
            i += 1
            continue
        j = i + 1
        results = []
        while j < len(conv) and conv[j].get("role") == "tool":
            content = conv[j].get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content)
            results.append((conv[j].get("tool_call_id"), content))
            j += 1
        for cursor, call in enumerate(calls):
            if not isinstance(call, dict) or not call.get("name"):
                continue
            tid, result = results[cursor] if cursor < len(results) else (None, "")
            found.append((call.get("name"), call.get("arguments") or {}, result, call.get("id"), tid))
        i = j
    return found


def id_result_map(conv):
    results = {}
    for turn in conv:
        if turn.get("role") == "tool":
            content = turn.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content)
            results[turn.get("tool_call_id")] = content
    return results


def as_dict(args):
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            value = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def as_obj(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def completed(name, result):
    text = (result or "").strip()
    if not text or text[:5].lower() == "error":
        return False
    if name == "send_certificate":
        return CERT_OK.match(text) is not None
    return text.startswith("{")


def num(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def book_paid(result):
    body = as_obj(result)
    if body is None:
        return None
    history = body.get("payment_history")
    if not isinstance(history, list) or not history:
        return None
    total = 0.0
    for payment in history:
        if not isinstance(payment, dict):
            return None
        amount = num(payment.get("amount"))
        if amount is None:
            return None
        total += abs(amount)
    return round(total, 2)


def certificate(result):
    match = CERT_AMT.search(result or "")
    return round(float(match.group(1)), 2) if match else None


def refund(result):
    body = as_obj(result)
    if body is None:
        return None
    wanted = body.get("return_items")
    items = body.get("items")
    if not isinstance(wanted, list) or not isinstance(items, list):
        return None
    pool = [item for item in items if isinstance(item, dict)]
    total = 0.0
    for item_id in wanted:
        chosen = None
        for index, item in enumerate(pool):
            if item.get("item_id") == item_id:
                chosen = index
                break
        if chosen is None:
            return None
        price = num(pool[chosen].get("price"))
        if price is None:
            return None
        total += price
        del pool[chosen]
    return round(total, 2)


def exchange(result):
    body = as_obj(result)
    if body is None:
        return None
    diff = num(body.get("exchange_price_difference"))
    return None if diff is None else round(abs(diff), 2)


def money_amount(name, result):
    if name == "book_reservation":
        return book_paid(result)
    if name == "send_certificate":
        return certificate(result)
    if name == "return_delivered_order_items":
        return refund(result)
    if name == "exchange_delivered_order_items":
        return exchange(result)
    return None


def user_text(conv):
    return "\n".join((turn.get("content") or "") for turn in conv if turn.get("role") == "user")


def reservation_ids(args):
    body = as_dict(args)
    rid = body.get("reservation_id")
    if isinstance(rid, str) and re.fullmatch(r"[A-Z0-9]{6}", rid):
        return {rid}
    return set()


def airports_of(args, result):
    found = set()
    body = as_dict(args)
    parsed = as_obj(result) or {}
    for source in (body, parsed):
        for key in ("origin", "destination"):
            value = source.get(key)
            if isinstance(value, str) and re.fullmatch(r"[A-Z]{3}", value):
                found.add(value)
        for flight in source.get("flights") or []:
            if not isinstance(flight, dict):
                continue
            for key in ("origin", "destination"):
                value = flight.get(key)
                if isinstance(value, str) and re.fullmatch(r"[A-Z]{3}", value):
                    found.add(value)
    return found


def order_ids(args):
    body = as_dict(args)
    found = set()
    oid = body.get("order_id")
    if isinstance(oid, str) and ORDER_ID.match(oid):
        found.add(oid)
    elif isinstance(oid, list):
        for item in oid:
            if isinstance(item, str) and ORDER_ID.match(item):
                found.add(item)
    return found


def correct(conv, domain):
    events = zip_events(conv)
    user = user_text(conv)
    named_res = set(RES_RE.findall(user))
    named_air = {tok.upper() for tok in AIRPORT.findall(user)}
    labels = {pid: None for pid in ATOMIC}
    for pid, (want, _tool) in ATOMIC.items():
        if domain == want:
            labels[pid] = False
    for name, args, result, _cid, _tid in events:
        if not completed(name, result):
            continue
        for pid, (want, tool) in ATOMIC.items():
            if domain == want and name == tool:
                labels[pid] = True
    if domain == "airline":
        used = set()
        for name, args, result, _c, _t in events:
            if name in RES_WRITES and completed(name, result):
                used |= reservation_ids(args)
        labels["airline_reservation_not_named"] = bool(used) and not used.issubset(named_res)
        seen = set()
        unknown = set()
        for name, args, result, _c, _t in events:
            if name not in {"book_reservation", "update_reservation_flights"} or not completed(name, result):
                continue
            codes = airports_of(args, result)
            seen |= codes
            unknown |= {code for code in codes if code not in named_air}
        labels["airline_airport_not_named"] = bool(unknown) if seen else False
        labels["retail_two_orders"] = None
    else:
        labels["airline_reservation_not_named"] = None
        labels["airline_airport_not_named"] = None
        orders = set()
        for name, args, result, _c, _t in events:
            if name in {
                "cancel_pending_order", "return_delivered_order_items",
                "exchange_delivered_order_items", "modify_pending_order_items",
                "modify_pending_order_address", "modify_pending_order_payment",
                "modify_user_address",
            } and completed(name, result):
                orders |= order_ids(args)
        labels["retail_two_orders"] = len(orders) >= 2
    total = 0.0
    saw = False
    missing = False
    for name, _args, result, _c, _t in events:
        if name not in MONEY or not completed(name, result):
            continue
        saw = True
        amount = money_amount(name, result)
        if amount is None:
            missing = True
        else:
            total += amount
    if missing:
        labels["total_charged_or_refunded_over_100"] = None
    elif not saw:
        labels["total_charged_or_refunded_over_100"] = False
    else:
        labels["total_charged_or_refunded_over_100"] = round(total, 2) > 100
    return labels, events


def head(text):
    return " ".join((text or "").split())[:180]


def explain(policy, events, conv):
    idmap = id_result_map(conv)
    rows = []
    for name, args, result, cid, tid in events:
        if policy.startswith("airline_completed") or policy.startswith("retail_completed"):
            tool = ATOMIC[policy][1]
            if name != tool:
                continue
        elif policy == "airline_reservation_not_named" and name not in RES_WRITES | {"book_reservation", "send_certificate"}:
            continue
        elif policy == "airline_airport_not_named" and name not in {"book_reservation", "update_reservation_flights"}:
            continue
        elif policy == "total_charged_or_refunded_over_100" and name not in MONEY:
            continue
        else:
            if policy not in ATOMIC and policy not in {
                "airline_reservation_not_named", "airline_airport_not_named",
                "total_charged_or_refunded_over_100",
            }:
                continue
        mapped = idmap.get(cid, "")
        rows.append({
            "name": name,
            "call_id": cid,
            "following_id": tid,
            "ids_differ_or_collapsed": cid != tid or head(mapped) != head(result),
            "zip_completed": completed(name, result),
            "idmap_completed": completed(name, mapped),
            "zip_head": head(result),
            "idmap_head": head(mapped),
            "args": head(json.dumps(as_dict(args)) if not isinstance(args, str) else args),
        })
    return rows


def main():
    if OUT.exists():
        raise SystemExit("exists " + str(OUT))
    action = json.loads(ACTION.read_text())["trajectories"]
    rel = {row["path"]: row for row in json.loads(REL.read_text())["trajectories"]}
    fixed = json.loads(FIXED.read_text())
    # fixed may be a dict path->labels or a list
    if isinstance(fixed, dict) and "trajectories" in fixed:
        fixed_rows = {row["path"]: row.get("labels") or row for row in fixed["trajectories"]}
    elif isinstance(fixed, list):
        fixed_rows = {row["path"]: row.get("labels") or row for row in fixed}
    else:
        fixed_rows = fixed
    atomic_pols = list(ATOMIC)
    rel_pols = [
        "airline_reservation_not_named", "airline_airport_not_named",
        "retail_two_orders", "total_charged_or_refunded_over_100",
    ]
    summary = {pid: {"agree": 0, "disagree": 0, "stored_pos": 0, "correct_pos": 0} for pid in atomic_pols + rel_pols}
    rows_out = []
    dup_files = 0
    for row in action:
        doc = json.loads((TAU / row["path"]).read_text())
        conv = doc["conversation"]
        domain = row.get("domain") or doc.get("domain")
        ids = []
        for turn in conv:
            calls = _loads_calls(turn.get("tool_calls")) or []
            if turn.get("role") == "assistant":
                ids.extend(call.get("id") for call in calls if isinstance(call, dict))
        if len(ids) != len(set(ids)):
            dup_files += 1
        labels, events = correct(conv, domain)
        stored_action = row["labels"]
        stored_rel = (rel[row["path"]].get("labels") or {})
        fixed_labels = fixed_rows.get(row["path"]) or fixed_rows.get(Path(row["path"]).name) or {}
        if isinstance(fixed_labels, dict) and "labels" in fixed_labels:
            fixed_labels = fixed_labels["labels"]
        for pid in atomic_pols:
            want = ATOMIC[pid][0]
            if domain != want:
                continue
            stored = bool(stored_action.get(pid))
            mine = bool(labels[pid])
            if stored:
                summary[pid]["stored_pos"] += 1
            if mine:
                summary[pid]["correct_pos"] += 1
            if stored == mine:
                summary[pid]["agree"] += 1
            else:
                summary[pid]["disagree"] += 1
                rows_out.append({
                    "path": row["path"], "policy": pid, "domain": domain,
                    "stored": stored, "correct": mine,
                    "calls": explain(pid, events, conv),
                })
        for pid in rel_pols:
            if pid.startswith("airline") and domain != "airline":
                continue
            if pid == "retail_two_orders" and domain != "retail":
                continue
            if pid in ("retail_two_orders", "total_charged_or_refunded_over_100"):
                raw = fixed_labels.get(pid, stored_rel.get(pid))
            else:
                raw = stored_rel.get(pid)
            if raw in (None, "exclude"):
                stored = None
            else:
                stored = bool(raw)
            mine = labels[pid]
            if stored is True:
                summary[pid]["stored_pos"] += 1
            if mine is True:
                summary[pid]["correct_pos"] += 1
            if stored == mine:
                summary[pid]["agree"] += 1
            else:
                summary[pid]["disagree"] += 1
                item = {
                    "path": row["path"], "policy": pid, "domain": domain,
                    "stored": stored, "correct": mine,
                    "calls": explain(pid, events, conv),
                }
                if pid == "airline_reservation_not_named":
                    item["user_ids"] = sorted(set(RES_RE.findall(user_text(conv))))
                if pid == "airline_airport_not_named":
                    item["user_airports"] = sorted({tok.upper() for tok in AIRPORT.findall(user_text(conv))})
                rows_out.append(item)
    payload = {
        "duplicate_id_files": dup_files,
        "n": len(action),
        "summary": summary,
        "disagreements": rows_out,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("dup_files", dup_files, "disagreements", len(rows_out), flush=True)
    for pid, stats in summary.items():
        delta = stats["correct_pos"] - stats["stored_pos"]
        print(f"{pid} agree {stats['agree']} disagree {stats['disagree']} stored {stats['stored_pos']} correct {stats['correct_pos']} delta {delta}", flush=True)


if __name__ == "__main__":
    main()
