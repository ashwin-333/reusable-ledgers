#!/usr/bin/env python3
"""Recount reservation and corrected money labels. No model calls."""
from __future__ import annotations

import json
import re
from pathlib import Path
from config import DATA_ROOT, OUT_ROOT, RUNS_DIR

SFT = DATA_ROOT
TAU = SFT / "taubench/trajectories"
REL = RUNS_DIR / "mp_controls/oct3/benches/tau_relational/multi_policy_benchmark.json"
FIXED = SFT / "taubench_rel_fixed/labels.json"
OUT = OUT_ROOT / "label_audit" / "recount.json"

# Reuse the zip walker by import would pull adjudicate; copy the small helpers.
from policies.taubench import adjudicate as A

RES_WRITES = {
    "cancel_reservation", "update_reservation_flights",
    "update_reservation_baggages", "update_reservation_passengers",
    "book_reservation",
}


def reservation_label(conv):
    user = A.user_text(conv)
    named = set(A.RES_RE.findall(user))
    used = set()
    notes = []
    for name, args, result, cid, tid in A.zip_events(conv):
        if not A.completed(name, result):
            continue
        if name not in RES_WRITES:
            continue
        if name == "book_reservation":
            body = A.as_obj(result) or {}
            rid = body.get("reservation_id")
            if isinstance(rid, str) and re.fullmatch(r"[A-Z0-9]{6}", rid):
                used.add(rid)
                notes.append({"name": name, "id": rid, "named": rid in named, "head": A.head(result)})
        else:
            for rid in A.reservation_ids(args):
                used.add(rid)
                notes.append({"name": name, "id": rid, "named": rid in named, "head": A.head(json.dumps(A.as_dict(args))[:160])})
    label = bool(used) and not used.issubset(named)
    return label, notes, sorted(named)


def money_label(conv):
    total = 0.0
    saw = False
    parts = []
    for name, args, result, cid, tid in A.zip_events(conv):
        if name not in A.MONEY or not A.completed(name, result):
            continue
        saw = True
        amount = A.money_amount(name, result)
        parts.append({"name": name, "amount": amount, "head": A.head(result)})
        if amount is None:
            return None, parts
        total += amount
    if not saw:
        return False, parts
    return round(total, 2) > 100, parts


def main():
    fixed = json.loads(FIXED.read_text())
    labels = fixed["labels"]
    print("labels type", type(labels).__name__, flush=True)
    if isinstance(labels, dict):
        key = next(iter(labels))
        print("key", key, "val", str(labels[key])[:240], flush=True)
        by_path = labels
    else:
        print("row", str(labels[0])[:300], flush=True)
        by_path = {}
        for row in labels:
            by_path[row.get("path") or row.get("id")] = row.get("labels", row)
    rel_rows = json.loads(REL.read_text())["trajectories"]
    res_dis = []
    money_dis = []
    two_dis = []
    res_stored_pos = res_ok_pos = res_n = 0
    money_stored_pos = money_ok_pos = money_n = 0
    two_stored_pos = two_ok_pos = two_n = 0
    for row in rel_rows:
        doc = json.loads((TAU / row["path"]).read_text())
        conv = doc["conversation"]
        domain = row.get("domain") or doc.get("domain")
        fixed_row = by_path.get(row["path"]) or by_path.get(Path(row["path"]).name) or {}
        if isinstance(fixed_row, dict) and "labels" in fixed_row and "retail_two_orders" not in fixed_row:
            fixed_row = fixed_row["labels"]
        if domain == "airline":
            res_n += 1
            stored = bool(row["labels"]["airline_reservation_not_named"])
            ok, notes, named = reservation_label(conv)
            if stored:
                res_stored_pos += 1
            if ok:
                res_ok_pos += 1
            if stored != ok:
                res_dis.append({
                    "path": row["path"], "stored": stored, "correct": ok,
                    "named": named, "writes": notes,
                })
        if domain == "retail":
            two_n += 1
            raw = fixed_row.get("retail_two_orders")
            stored = None if raw in (None, "exclude") else bool(raw)
            orders = set()
            for name, args, result, _c, _t in A.zip_events(conv):
                if name in {
                    "cancel_pending_order", "return_delivered_order_items",
                    "exchange_delivered_order_items", "modify_pending_order_items",
                    "modify_pending_order_address", "modify_pending_order_payment",
                    "modify_user_address",
                } and A.completed(name, result):
                    orders |= A.order_ids(args)
            ok = len(orders) >= 2
            if stored is True:
                two_stored_pos += 1
            if ok:
                two_ok_pos += 1
            if stored != ok:
                two_dis.append({"path": row["path"], "stored": stored, "correct": ok, "orders": sorted(orders)})
        money_n += 1
        raw = fixed_row.get("total_charged_or_refunded_over_100")
        stored = None if raw in (None, "exclude") else bool(raw)
        ok, parts = money_label(conv)
        if stored is True:
            money_stored_pos += 1
        if ok is True:
            money_ok_pos += 1
        if stored != ok:
            money_dis.append({"path": row["path"], "domain": domain, "stored": stored, "correct": ok, "parts": parts})
    payload = {
        "reservation": {"n": res_n, "agree": res_n - len(res_dis), "disagree": len(res_dis),
                        "stored_pos": res_stored_pos, "correct_pos": res_ok_pos,
                        "delta": res_ok_pos - res_stored_pos, "rows": res_dis},
        "two_orders": {"n": two_n, "agree": two_n - len(two_dis), "disagree": len(two_dis),
                       "stored_pos": two_stored_pos, "correct_pos": two_ok_pos,
                       "delta": two_ok_pos - two_stored_pos, "rows": two_dis},
        "money": {"n": money_n, "agree": money_n - len(money_dis), "disagree": len(money_dis),
                  "stored_pos": money_stored_pos, "correct_pos": money_ok_pos,
                  "delta": (money_ok_pos - money_stored_pos), "rows": money_dis},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("reservation", payload["reservation"]["agree"], payload["reservation"]["disagree"], "delta", payload["reservation"]["delta"], flush=True)
    print("two", payload["two_orders"]["agree"], payload["two_orders"]["disagree"], "delta", payload["two_orders"]["delta"], flush=True)
    print("money", payload["money"]["agree"], payload["money"]["disagree"], "delta", payload["money"]["delta"], flush=True)


if __name__ == "__main__":
    main()
