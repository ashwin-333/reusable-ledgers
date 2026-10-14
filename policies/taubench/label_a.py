"""Labeler A: pair by tool_call_id, parse arguments/results with json.loads."""
from __future__ import annotations

import json
import re

AIRLINE_WRITE = {
    "cancel_reservation",
    "book_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "update_reservation_passengers",
    "send_certificate",
}
RETAIL_WRITE = {
    "cancel_pending_order",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
}
MONEY_TOOLS = {
    "book_reservation",
    "send_certificate",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
}
AIRLINE_ACTION = {
    "airline_completed_cancellation": "cancel_reservation",
    "airline_completed_booking": "book_reservation",
    "airline_completed_flight_change": "update_reservation_flights",
}
RETAIL_ACTION = {
    "retail_completed_order_cancellation": "cancel_pending_order",
    "retail_completed_return": "return_delivered_order_items",
    "retail_completed_exchange": "exchange_delivered_order_items",
    "retail_completed_item_modification": "modify_pending_order_items",
    "retail_completed_address_change": "modify_pending_order_address",
}
AIRPORT_KEYS = ("origin", "destination")
ORDER_RE = re.compile(r"#W\d{7}")
RES_RE = re.compile(r'"reservation_id"\s*:\s*"([A-Z0-9]{6})"')
CERT_RE = re.compile(r"(?i)^certificate\s+\S+\s+added\b")
AIRPORT_TOKEN = re.compile(r"\b([A-Za-z]{3})\b")


def _object(text: str):
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _completed(name: str, result: str) -> bool:
    text = (result or "").strip()
    if not text or text[:5].lower() == "error":
        return False
    if name == "send_certificate":
        return CERT_RE.match(text) is not None
    return text[0] == "{"


def _events(conversation: list[dict]) -> list[tuple[str, str, str]]:
    results = {
        turn.get("tool_call_id"): turn.get("content") or ""
        for turn in conversation
        if turn.get("role") == "tool"
    }
    events = []
    for turn in conversation:
        if turn.get("role") != "assistant":
            continue
        for call in turn.get("tool_calls") or []:
            name = call.get("name")
            if not name:
                continue
            result = results.get(call.get("id"), "")
            args = call.get("arguments") or ""
            if not isinstance(args, str):
                args = json.dumps(args, sort_keys=True)
            events.append((name, args, result))
    return events


def _user_text(conversation: list[dict]) -> str:
    return "\n".join(
        turn.get("content") or ""
        for turn in conversation
        if turn.get("role") == "user"
    )


def _prices(name: str, result: str) -> list[int] | None:
    if name == "send_certificate":
        match = re.search(r"amount\s+(\d+)", result, re.I)
        return [int(match.group(1))] if match else None
    body = _object(result)
    if body is None:
        return None
    found = []
    if isinstance(body.get("price"), int):
        found.append(body["price"])
    for flight in body.get("flights") or []:
        if isinstance(flight, dict) and isinstance(flight.get("price"), int):
            found.append(flight["price"])
    for item in body.get("items") or []:
        if isinstance(item, dict) and isinstance(item.get("price"), int):
            found.append(item["price"])
    return found or None


def label_a(document: dict) -> dict:
    conversation = document.get("conversation") or []
    events = _events(conversation)
    done = [(name, args, result) for name, args, result in events if _completed(name, result)]
    user = _user_text(conversation)
    out: dict = {}
    for policy, tool in AIRLINE_ACTION.items():
        out[policy] = any(name == tool for name, _args, _result in done)
    for policy, tool in RETAIL_ACTION.items():
        out[policy] = any(name == tool for name, _args, _result in done)

    named_res = set(re.findall(r"\b[A-Z0-9]{6}\b", user))
    used_res = set()
    for name, args, result in done:
        if name in AIRLINE_WRITE:
            used_res.update(RES_RE.findall(args))
            used_res.update(RES_RE.findall(result))
            parsed = _object(args) or {}
            rid = parsed.get("reservation_id")
            if isinstance(rid, str) and re.fullmatch(r"[A-Z0-9]{6}", rid):
                used_res.add(rid)
    out["airline_reservation_not_named"] = bool(used_res) and not used_res.issubset(named_res)

    named_airports = {token.upper() for token in AIRPORT_TOKEN.findall(user)}
    dest_hit = False
    dest_seen = False
    for name, args, result in done:
        if name not in {"book_reservation", "update_reservation_flights"}:
            continue
        blob = (_object(args) or {}) | (_object(result) or {})
        for key in AIRPORT_KEYS:
            value = blob.get(key)
            if isinstance(value, str) and re.fullmatch(r"[A-Z]{3}", value):
                dest_seen = True
                if value not in named_airports:
                    dest_hit = True
        for flight in (blob.get("flights") or []):
            if not isinstance(flight, dict):
                continue
            for key in AIRPORT_KEYS:
                value = flight.get(key)
                if isinstance(value, str) and re.fullmatch(r"[A-Z]{3}", value):
                    dest_seen = True
                    if value not in named_airports:
                        dest_hit = True
    out["airline_airport_not_named"] = dest_hit if dest_seen else False

    orders = set()
    for name, args, result in done:
        if name in RETAIL_WRITE:
            orders.update(ORDER_RE.findall(args))
            orders.update(ORDER_RE.findall(result))
            parsed = _object(args) or {}
            oid = parsed.get("order_id")
            if isinstance(oid, str) and ORDER_RE.fullmatch(oid):
                orders.add(oid)
    out["retail_two_orders"] = len(orders) >= 2

    total = 0
    missing = False
    any_money = False
    for name, _args, result in done:
        if name not in MONEY_TOOLS:
            continue
        any_money = True
        prices = _prices(name, result)
        if prices is None:
            missing = True
        else:
            total += sum(prices)
    if missing:
        out["total_charged_or_refunded_over_100"] = "exclude"
    else:
        out["total_charged_or_refunded_over_100"] = bool(any_money) and total > 100
    return out
