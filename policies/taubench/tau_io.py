"""Shared tau-bench I/O. Strip evaluation fields. No labels here."""
from __future__ import annotations

import json
import re
from pathlib import Path

KEEP_ROLES = {"user", "assistant", "tool"}
SOURCE_FILES = (
    ("airline", "gpt-4o", "gpt-4o-airline.json"),
    ("retail", "gpt-4o", "gpt-4o-retail.json"),
    ("airline", "sonnet-35-new", "sonnet-35-new-airline.json"),
    ("retail", "sonnet-35-new", "sonnet-35-new-retail.json"),
)
AIRLINE_WRITE = (
    "cancel_reservation",
    "book_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "update_reservation_passengers",
    "send_certificate",
)
RETAIL_WRITE = (
    "cancel_pending_order",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
)
ORDER_RE = re.compile(r"#W\d{7}")
RES_ID_RE = re.compile(r"\b[A-Z0-9]{6}\b")
AIRPORT_RE = re.compile(r"\b[A-Z]{3}\b")


def source_stem(domain: str, agent: str, task_id: int, trial: int) -> str:
    return f"{domain}__{agent}__task{task_id:03d}__trial{trial}"


def strip_turn(turn: dict) -> dict | None:
    role = turn.get("role")
    if role not in KEEP_ROLES:
        return None
    out: dict = {"role": role}
    if role == "assistant":
        content = turn.get("content")
        if isinstance(content, str) and content.strip():
            out["content"] = content
        calls = []
        for call in turn.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = fn.get("name") or call.get("name")
            args = fn.get("arguments") or call.get("arguments") or ""
            if not isinstance(args, str):
                args = json.dumps(args, sort_keys=True)
            if not name:
                continue
            item = {"id": call.get("id"), "name": name, "arguments": args}
            calls.append(item)
        if calls:
            out["tool_calls"] = calls
        if "content" not in out and "tool_calls" not in out:
            return None
        return out
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": turn.get("tool_call_id"),
            "name": turn.get("name"),
            "content": turn.get("content") or "",
        }
    content = turn.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    return {"role": "user", "content": content}


def strip_record(raw: dict, domain: str, agent: str) -> dict:
    conversation = []
    for turn in raw.get("traj") or []:
        if isinstance(turn, dict):
            kept = strip_turn(turn)
            if kept is not None:
                conversation.append(kept)
    return {
        "path": source_stem(domain, agent, int(raw["task_id"]), int(raw["trial"])) + ".json",
        "domain": domain,
        "agent_model": agent,
        "task_id": int(raw["task_id"]),
        "trial": int(raw["trial"]),
        "conversation": conversation,
    }


def render_observed(conversation: list[dict]) -> str:
    lines = []
    for turn in conversation:
        role = turn["role"]
        if role == "user":
            lines.append("USER: " + turn["content"])
        elif role == "assistant":
            if turn.get("content"):
                lines.append("ASSISTANT: " + turn["content"])
            for call in turn.get("tool_calls") or []:
                lines.append(f"TOOL_CALL {call['name']} {call.get('arguments') or ''}")
        elif role == "tool":
            lines.append(f"TOOL_RESULT {turn.get('name') or ''} {turn.get('content') or ''}")
    return "\n".join(lines)


def load_sources(raw_dir: Path) -> list[dict]:
    rows = []
    for domain, agent, name in SOURCE_FILES:
        payload = json.loads((raw_dir / name).read_text())
        if not isinstance(payload, list):
            raise SystemExit(f"{name} is not a list")
        for raw in payload:
            rows.append(strip_record(raw, domain, agent))
    return rows
