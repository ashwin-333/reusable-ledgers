"""Shared constants, paths, and defaults for the autoresearch loop."""

from __future__ import annotations


# ── Paths ────────────────────────────────────────────────────────────────────
from config import OUT_ROOT, REPO_ROOT

AUTORESEARCH_DIR = REPO_ROOT / "autoresearch"
RESULTS_DIR = OUT_ROOT / "autoresearch"
RUN_LOG_PATH = RESULTS_DIR / "run_log.jsonl"
BUDGET_LEDGER_PATH = RESULTS_DIR / "budget_ledger.jsonl"
API_CALL_LEDGER_PATH = RESULTS_DIR / "api_call_ledger.jsonl"
LLM_RESPONSE_CACHE_PATH = RESULTS_DIR / "llm_response_cache.jsonl"
PIPELINES_DIR = RESULTS_DIR / "pipelines"

PROMPTS_DIR = AUTORESEARCH_DIR / "prompts"

# ── Budget ───────────────────────────────────────────────────────────────────

# The per-iteration value is a soft target; the global value is a hard admission cap.
DEFAULT_GLOBAL_BUDGET_USD = 20.0
DEFAULT_PER_ITER_BUDGET_USD = 1.0
OVERSHOOT_TOLERANCE = 1.2

DEFAULT_LLM_MAX_TOKENS = 32
DEFAULT_LLM_CONCURRENCY = 4

# ── Evaluation ───────────────────────────────────────────────────────────────

CONSTRUCTS = ["c1", "c2", "c3", "c4"]
BUCKET_ORDER = ["0-10k", "10k-20k", "20k-30k", "30k-40k", "40k-50k", "50k+"]

# Trajectory label mapping (from directory structure)
MALICIOUS_POSITIONS = {"beginning", "middle", "end"}
BENIGN_POSITION = "benign"

# ── Dev/held-out split ───────────────────────────────────────────────────────

DEFAULT_DEV_FRACTION = 0.7
DEFAULT_SPLIT_SEED = 42
