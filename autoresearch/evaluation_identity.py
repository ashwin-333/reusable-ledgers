"""Content identity for LDS-v2 validation, evaluation, and matched baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import AUTORESEARCH_DIR, REPO_ROOT


def evaluation_runtime_paths() -> list[Path]:
    """Sources that can change constructor inputs, parsing, or LDS-v2 scores."""

    paths = [
        AUTORESEARCH_DIR / "candidate_config.py",
        AUTORESEARCH_DIR / "evaluation_identity.py",
        AUTORESEARCH_DIR / "evidence.py",
        AUTORESEARCH_DIR / "ledger_score.py",
        AUTORESEARCH_DIR / "lds_v2.py",
        AUTORESEARCH_DIR / "recompute_metrics.py",
        REPO_ROOT / "ledgers" / "render.py",
        REPO_ROOT / "ledgers" / "prompt_multipart.py",
    ]
    paths.extend(sorted((AUTORESEARCH_DIR / "tools").glob("*.py")))
    return paths


def build_evaluation_identity(
    *,
    manifest_path: Path,
    data_root: Path,
    provider: str,
    ledger_model: str,
    ledger_max_tokens: int,
    unknown_policy: str,
) -> dict[str, Any]:
    """Hash the exact evaluation target plus its provider-specific execution."""

    manifest_path = Path(manifest_path)
    data_root = Path(data_root)
    manifest_bytes = manifest_path.read_bytes()
    manifest_rows = json.loads(manifest_bytes)
    if not isinstance(manifest_rows, list):
        raise TypeError(f"manifest must be a JSON list: {manifest_path}")

    trajectory_hasher = hashlib.sha256()
    trajectory_hasher.update(manifest_bytes)
    for row in manifest_rows:
        relative = str(row.get("path") or row.get("trajectory_path"))
        if not relative or relative == "None":
            raise ValueError(f"manifest row has no trajectory path: {row!r}")
        raw_path = Path(relative)
        trajectory_path = raw_path if raw_path.is_absolute() else data_root / raw_path
        trajectory_hasher.update(relative.encode("utf-8"))
        trajectory_hasher.update(
            hashlib.sha256(trajectory_path.read_bytes()).digest()
        )

    runtime_hasher = hashlib.sha256()
    for path in evaluation_runtime_paths():
        runtime_hasher.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        runtime_hasher.update(hashlib.sha256(path.read_bytes()).digest())

    target = {
        "schema": "lds-v2",
        "manifest": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "trajectory_bundle_sha256": trajectory_hasher.hexdigest(),
        "ledger_model": ledger_model,
        "ledger_max_tokens": int(ledger_max_tokens),
        "unknown_policy": unknown_policy,
        "runtime_sha256": runtime_hasher.hexdigest(),
    }
    target_fingerprint = hashlib.sha256(
        json.dumps(target, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = {
        **target,
        "provider": provider,
        "validation_target_fingerprint": target_fingerprint,
    }
    identity["fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity
