"""Trajectory loading and the compact rendering every writer and raw reader sees."""

import json
import re
from types import SimpleNamespace


def _ellipsize(s, max_len=None):
    if s is None: return "<none>"
    if max_len is None or len(s) <= max_len: return s
    keep = max_len // 2
    return s[:keep] + " … " + s[-(max_len - keep - 3):]

def _normalize_tool_calls(raw):
    """Convert tool_calls to a string regardless of source format."""
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        lines = []
        for tc in raw:
            if isinstance(tc, str):
                # "function='foo' args={...} id='...'"
                m = re.match(r"function='([^']+)'", tc)
                name = m.group(1) if m else tc[:60]
                args_m = re.search(r"args=(\{.*?\})(?:\s|$)", tc)
                args_str = args_m.group(1) if args_m else ""
                lines.append(f"- {name}" + (f" {args_str}" if args_str else ""))
            elif isinstance(tc, dict):
                name = tc.get("function") or tc.get("name") or "unknown"
                args = tc.get("args") or tc.get("arguments") or {}
                lines.append(f"- {name} {json.dumps(args)}" if args else f"- {name}")
            else:
                lines.append(f"- {str(tc)[:120]}")
        return "\n".join(lines)
    return str(raw)

def load_trace_auto(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    conv = obj.get("conversation") or obj.get("messages")
    if conv is None:
        raise ValueError("No 'conversation' or 'messages' in JSON")
    turns, last_assistant_with_tool = [], None
    for idx, m in enumerate(conv):
        if m.get("role") == "system": continue
        role = m.get("role", "user")
        text = m.get("content", "") or ""
        tool_calls_str = _normalize_tool_calls(m.get("tool_calls"))
        if role == "tool":
            if last_assistant_with_tool is not None:
                if not hasattr(last_assistant_with_tool, "tool_out"):
                    last_assistant_with_tool.tool_out = text
                    last_assistant_with_tool.tool_out_short = text
                else:
                    last_assistant_with_tool.tool_out = f"{last_assistant_with_tool.tool_out}\n\n{text}"
                    last_assistant_with_tool.tool_out_short = last_assistant_with_tool.tool_out
            continue
        turn = SimpleNamespace(index=idx, role=role, text=text, tool_name=None,
                               tool_args=None, tool_args_short=None, tool_out=None, tool_out_short=None)
        if tool_calls_str:
            for line in tool_calls_str.splitlines():
                line = line.strip()
                if line.startswith("- ") and line[2:].strip():
                    turn.tool_name = line[2:].strip()
                    turn.tool_args = tool_calls_str
                    turn.tool_args_short = tool_calls_str
                    last_assistant_with_tool = turn
                    break
        turns.append(turn)
    return SimpleNamespace(turns=turns)

def render_turns_compact(turns, *, max_tool_chars=None, max_msg_chars=None):
    lines = []
    for t in turns:
        role = getattr(t, "role", "unknown")
        if role.lower() == "system": continue
        tool = getattr(t, "tool_name", None)
        full_out = getattr(t, "tool_out", None)
        out_short = getattr(t, "tool_out_short", None)
        msg = getattr(t, "text", None)
        ru = role.upper()
        if role.lower() in ["assistant", "user"]:
            if msg: lines.append(f"{ru}: {_ellipsize(msg, max_msg_chars)}")
            if tool:
                lines.append("Tool calls:")
                lines.append(f"- {tool}")
            if full_out is not None:
                to = full_out if max_tool_chars is None else (out_short or _ellipsize(full_out, max_tool_chars))
                lines.append(f"TOOL: {to}")
            elif out_short:
                lines.append(f"TOOL: {out_short}")
    return "\n".join(lines)
