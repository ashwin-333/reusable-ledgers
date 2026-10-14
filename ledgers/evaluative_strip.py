"""Frozen evaluative-field strip. The text file is the rule; this module applies it.

The superseded clause-deletion rule is not implemented here.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULE_PATH = HERE / "prompts" / "evaluative_strip.txt"
RULE_SHA_PATH = HERE / "prompts" / "evaluative_strip.sha256"
SUPERSEDED_SHA = "d063b248d31e080988d183e618dfb991a99478fa533a080d2ebf059c7e68f7aa"
SEPARATOR = " | "
AUTHORIZED = "Authorized:"
ACTION = "Tool:"
_SENTENCE = re.compile(r"(?<=[.!?;])\s+")
_SENTENCE_PHRASES = (
    "not authorized",
    "unauthorized",
    "However,",
    "should not",
    "sensitive",
)


def rule_text() -> str:
    return RULE_PATH.read_text()


def rule_sha256() -> str:
    digest = hashlib.sha256(RULE_PATH.read_bytes()).hexdigest()
    frozen = RULE_SHA_PATH.read_text().strip()
    if digest != frozen:
        raise SystemExit("evaluative strip rule changed after it was frozen")
    if SUPERSEDED_SHA not in rule_text():
        raise SystemExit("superseded rule hash is missing from the log")
    phrases = _sentence_phrases()
    if phrases != _SENTENCE_PHRASES:
        raise SystemExit("strip rule substrings are not the frozen list")
    return digest


def _sentence_phrases() -> tuple[str, ...]:
    lines = rule_text().splitlines()
    try:
        start = lines.index("SUBSTRINGS:") + 1
    except ValueError:
        raise SystemExit("strip rule has no SUBSTRINGS") from None
    found = []
    for line in lines[start:]:
        if line == "":
            break
        found.append(line)
    return tuple(found)


def _strip_authorized(line: str) -> str:
    while AUTHORIZED in line:
        start = line.find(AUTHORIZED)
        rest = line[start + len(AUTHORIZED):]
        sep_at = rest.find(SEPARATOR)
        if sep_at == -1:
            prefix = line[:start]
            if prefix.endswith(SEPARATOR):
                prefix = prefix[: -len(SEPARATOR)]
            line = prefix
            break
        end = start + len(AUTHORIZED) + sep_at
        line = line[:start] + line[end + len(SEPARATOR):]
    if line.endswith(SEPARATOR):
        line = line[: -len(SEPARATOR)]
    if line.startswith(SEPARATOR):
        line = line[len(SEPARATOR):]
    return line


def _strip_sentences(line: str) -> str:
    phrases = _sentence_phrases()
    kept = []
    for sentence in _SENTENCE.split(line):
        if sentence and not any(phrase in sentence for phrase in phrases):
            kept.append(sentence)
    return " ".join(kept)


def strip_ledger(text: str) -> str:
    """Apply the frozen field rule. Action lines lose only Authorized fields."""
    rule_sha256()
    kept = []
    for line in text.splitlines():
        if not line.strip():
            kept.append(line)
            continue
        updated = _strip_authorized(line)
        if ACTION not in line:
            updated = _strip_sentences(updated)
        if updated.strip():
            kept.append(updated)
    return "\n".join(kept)


def removal_counts(text: str) -> dict:
    stripped = strip_ledger(text)
    before_lines = text.splitlines()
    after_lines = stripped.splitlines()
    return {
        "chars_before": len(text),
        "chars_after": len(stripped),
        "chars_removed": len(text) - len(stripped),
        "lines_before": len(before_lines),
        "lines_after": len(after_lines),
        "lines_removed": len(before_lines) - len(after_lines),
    }
