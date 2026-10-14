"""Paths and model IDs. Every value can be overridden with an environment variable of the same name.

DATA_ROOT     benchmark trajectories, benchmark files, and sampled-ID lists (see README.md, Data)
RESULTS_ROOT  directory holding the released results/ and labels/ read by eval/reproduce_main.py
OUT_ROOT      where writers, readers, and label builders put new outputs
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DATA_ROOT = _path("DATA_ROOT", REPO_ROOT / "data")
RESULTS_ROOT = _path("RESULTS_ROOT", REPO_ROOT)
OUT_ROOT = _path("OUT_ROOT", REPO_ROOT / "outputs")

WRITER_MODEL = os.environ.get("WRITER_MODEL", "google/gemma-4-12B-it")
READER_MODEL = os.environ.get("READER_MODEL", "google/gemma-4-12B-it")
READER_MODEL_LARGE = os.environ.get("READER_MODEL_LARGE", "google/gemma-4-31B-it")
QWEN_READER_MODEL = os.environ.get("QWEN_READER_MODEL", "Qwen/Qwen3.5-9B")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")

MODEL_REVISIONS = {
    "google/gemma-4-12B-it": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "google/gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
}

SUBSET_FILE = DATA_ROOT / "e5_w1_subset.json"
STATUS_DIR = OUT_ROOT / "status"
RUNS_DIR = OUT_ROOT / "runs"


def model_revision(model: str) -> str | None:
    return MODEL_REVISIONS.get(model)


def snapshot_dir(model: str, *, weights: bool = False) -> Path:
    """Local Hugging Face snapshot for a model, downloaded into the standard Hugging Face cache if missing."""
    from huggingface_hub import snapshot_download

    patterns = None if weights else ["*.json", "tokenizer*", "*.model", "*.txt"]
    return Path(snapshot_download(model, revision=model_revision(model), allow_patterns=patterns))


def tokenizer_file(model: str) -> Path:
    return snapshot_dir(model) / "tokenizer.json"
