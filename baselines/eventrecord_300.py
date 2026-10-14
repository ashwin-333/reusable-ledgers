"""Event records published with READY_eventrecord_300.

The subset records stay where they are. This module only notices a later
300-trajectory drop, and only after that marker exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from config import DATA_ROOT, RUNS_DIR

SFT = DATA_ROOT
MARKERS = (
    SFT / "READY_eventrecord_300",
    SFT / "eventrecord" / "READY_eventrecord_300",
    RUNS_DIR / "READY_eventrecord_300",
)
SUBSET_RECORDS = RUNS_DIR / "subset100/eventrecord/constructor_records"
BENCH = SFT / "multi_policy_real/multi_policy_benchmark.json"


def marker() -> Path | None:
    for path in MARKERS:
        if path.is_file():
            return path
    return None


def _record_dirs(path: Path) -> list[Path]:
    found = []
    for line in path.read_text().splitlines():
        raw = None
        if "dir=" in line:
            raw = line.split("dir=", 1)[1].split()[0]
        elif line.strip().startswith("dir:"):
            raw = line.split("dir:", 1)[1].strip().split()[0]
        if raw is None:
            continue
        candidate = Path(raw)
        if candidate.name != "constructor_records":
            nested = candidate / "constructor_records"
            alt = candidate / "eventrecord" / "constructor_records"
            if nested.is_dir():
                candidate = nested
            elif alt.is_dir():
                candidate = alt
        if candidate.is_dir():
            found.append(candidate)
    if found:
        return found
    root = RUNS_DIR
    return sorted(item for item in root.glob("*/eventrecord/constructor_records") if item.is_dir())


def _texts(directory: Path) -> dict[str, str]:
    found = {}
    if not directory.is_dir():
        return found
    for file in directory.glob("*.json"):
        try:
            saved = json.loads(file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        record = saved.get("record") if isinstance(saved.get("record"), dict) else {}
        if record.get("ledger_error") or saved.get("ledger_error"):
            continue
        text = record.get("ledger_text") or saved.get("ledger_text") or ""
        if not str(text).strip():
            continue
        key = saved.get("path") or record.get("path")
        if key:
            found[str(key)] = str(text)
    return found


def extra_texts() -> dict[str, str]:
    """Records named by the 300 marker, other than the existing subset directory."""
    path = marker()
    if path is None:
        return {}
    found = {}
    subset = SUBSET_RECORDS.resolve()
    for directory in _record_dirs(path):
        try:
            if directory.resolve() == subset:
                continue
        except OSError:
            continue
        found.update(_texts(directory))
    return found


def shade_paths() -> set[str]:
    rows = json.loads(BENCH.read_text())["trajectories"]
    return {row["path"] for row in rows}


def ready() -> bool:
    """True when the marker exists and all 300 SHADE trajectories have a record."""
    if marker() is None:
        return False
    covered = set(_texts(SUBSET_RECORDS))
    covered.update(extra_texts())
    return len(covered & shade_paths()) >= 300
