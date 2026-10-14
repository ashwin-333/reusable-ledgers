"""Labeler B: burst-zip tool messages, field values via regex only."""
from __future__ import annotations

import json
import re

AIR_CANCEL = "cancel_reservation"
AIR_BOOK = "book_reservation"
AIR_FLIGHT = "update_reservation_flights"
AIR_WRITES = {
    AIR_CANCEL, AIR_BOOK, AIR_FLIGHT,
    "update_reservation_baggages", "update_reservation_passengers", "send_certificate",
}
RET_CANCEL = "cancel_pending_order"
RET_RETURN = "return_delivered_order_items"
RET_EXCHANGE = "exchange_delivered_order_items"
RET_ITEMS = "modify_pending_order_items"
RET_ADDR = "modify_pending_order_address"
RET_WRITES = {
    RET_CANCEL, RET_RETURN, RET_EXCHANGE, RET_ITEMS, RET_ADDR,
    "modify_pending_order_payment", "modify_user_address",
}
MONEY = {AIR_BOOK, "send_certificate", RET_RETURN, RET_EXCHANGE}
ERROR_HEAD = re.compile(r"(?i)^error\b")
CERT_OK = re.compile(r"(?i)^certificate\b.+\badded\b")
RES_FIELD = re.compile(r"reservation_id[\"'\s:=]+([A-Z0-9]{6})")
ORDER_FIELD = re.compile(r"#W\d{7}")
ORIGIN = re.compile(r"origin[\"'\s:=]+([A-Z]{3})")
DEST = re.compile(r"destination[\"'\s:=]+([A-Z]{3})")
PRICE = re.compile(r"price[\"'\s:=]+(\d+)")
CERT_AMT = re.compile(r"(?i)amount\s+(\d+)")
WORD6 = re.compile(r"\b[A-Z0-9]{6}\b")
WORD3 = re.compile(r"\b[A-Za-z]{3}\b")


def _ok(name: str, result: str) -> bool:
    text = (result or "").strip()
    if not text or ERROR_HEAD.match(text):
        return False
    if name == "send_certificate":
        return CERT_OK.match(text) is not None
    return text.startswith("{")


def _bursts(conversation: list[dict]) -> list[tuple[str, str, str]]:
    events = []
    i = 0
    while i < len(conversation):
        turn = conversation[i]
        names = []
        if turn.get("role") == "assistant":
            for call in turn.get("tool_calls") or []:
                name = call.get("name")
                if name:
                    args = call.get("arguments") or ""
                    if not isinstance(args, str):
                        args = json.dumps(args, sort_keys=True)
                    names.append((name, args))
        if not names:
            i += 1
            continue
        results = []
        j = i + 1
        while j < len(conversation) and conversation[j].get("role") == "tool":
            results.append(conversation[j].get("content") or "")
            j += 1
        if len(names) == len(results):
            paired = list(zip(names, results))
        else:
            joined = "\n".join(results)
            paired = [((name, args), joined) for name, args in names]
        for (name, args), result in paired:
            events.append((name, args, result))
        i = j
    return events


def _user_blob(conversation: list[dict]) -> str:
    parts = []
    for turn in conversation:
        if turn.get("role") == "user":
            parts.append(turn.get("content") or "")
    return "\n".join(parts)


def _money(name: str, result: str) -> list[int] | None:
    if name == "send_certificate":
        match = CERT_AMT.search(result or "")
        return [int(match.group(1))] if match else None
    found = [int(x) for x in PRICE.findall(result or "")]
    return found or None


def label_b(document: dict) -> dict:
    conversation = document.get("conversation") or []
    events = _bursts(conversation)
    done = [row for row in events if _ok(row[0], row[2])]
    users = _user_blob(conversation)
    labels = {
        "airline_completed_cancellation": any(row[0] == AIR_CANCEL for row in done),
        "airline_completed_booking": any(row[0] == AIR_BOOK for row in done),
        "airline_completed_flight_change": any(row[0] == AIR_FLIGHT for row in done),
        "retail_completed_order_cancellation": any(row[0] == RET_CANCEL for row in done),
        "retail_completed_return": any(row[0] == RET_RETURN for row in done),
        "retail_completed_exchange": any(row[0] == RET_EXCHANGE for row in done),
        "retail_completed_item_modification": any(row[0] == RET_ITEMS for row in done),
        "retail_completed_address_change": any(row[0] == RET_ADDR for row in done),
    }

    mentioned = set(WORD6.findall(users))
    used = set()
    for name, args, result in done:
        if name in AIR_WRITES:
            used.update(RES_FIELD.findall(args))
            used.update(RES_FIELD.findall(result))
    labels["airline_reservation_not_named"] = bool(used - mentioned)

    airports = {token.upper() for token in WORD3.findall(users)}
    unseen = False
    present = False
    for name, args, result in done:
        if name not in {AIR_BOOK, AIR_FLIGHT}:
            continue
        blob = f"{args}\n{result}"
        codes = ORIGIN.findall(blob) + DEST.findall(blob)
        if codes:
            present = True
        if any(code not in airports for code in codes):
            unseen = True
    labels["airline_airport_not_named"] = unseen if present else False

    orders = set()
    for name, args, result in done:
        if name in RET_WRITES:
            orders.update(ORDER_FIELD.findall(args))
            orders.update(ORDER_FIELD.findall(result))
    labels["retail_two_orders"] = len(orders) >= 2

    total = 0
    saw = False
    missing = False
    for name, _args, result in done:
        if name not in MONEY:
            continue
        saw = True
        amounts = _money(name, result)
        if amounts is None:
            missing = True
        else:
            total += sum(amounts)
    if missing:
        labels["total_charged_or_refunded_over_100"] = "exclude"
    else:
        labels["total_charged_or_refunded_over_100"] = bool(saw) and total > 100
    return labels
