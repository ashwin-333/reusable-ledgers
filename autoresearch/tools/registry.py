"""Tool registry — maps tool IDs to their implementing classes."""

from __future__ import annotations

from .t03_toolcall_extractor import ToolCallExtractor
from .t04_trajectory_table import TrajectoryTableRenderer
from .t05_toolcall_stats import ToolCallStats
from .t06_ngram_detector import NgramDetector
from .t07_action_entropy import ActionEntropy
from .t08_cost_profiler import CostProfiler
from .t09_isolation_forest import IsolationForestTool
from .t10_lof import LocalOutlierFactor
from .t11_zscore_threshold import ZScoreThreshold
from .t12_markov_surprise import MarkovSurprise
from .t17_salience_windows import SalienceWindows
from .t20_state_diff import StateDiffDetector
from .t21_permission_scanner import PermissionScanner
from .t22_secret_scanner import SecretScanner
from .t31_turn_selector import TurnFieldSelector
from .judge import ThresholdScorer, AugmentedJudge
from .linear_scorer import LinearScorer
from .population_anomaly import PopulationAnomalyScorer

TOOL_REGISTRY: dict[str, type] = {
    # LLM-based tools (require .set_llm(adapter))
    # Pure-compute tools
    "T03": ToolCallExtractor,
    "T04": TrajectoryTableRenderer,
    "T05": ToolCallStats,
    "T06": NgramDetector,
    "T07": ActionEntropy,
    "T08": CostProfiler,
    "T09": IsolationForestTool,
    "T10": LocalOutlierFactor,
    "T11": ZScoreThreshold,
    "T12": MarkovSurprise,
    "T17": SalienceWindows,
    # LLM-based tools (require .set_llm(adapter))
    # Scanners
    "T20": StateDiffDetector,
    "T21": PermissionScanner,
    "T22": SecretScanner,
    "T31": TurnFieldSelector,
    # Scorers
    "threshold_scorer": ThresholdScorer,
    "linear_scorer": LinearScorer,
    "population_anomaly": PopulationAnomalyScorer,
    "judge": AugmentedJudge,
}


def get_tool(tool_id: str, params: dict | None = None):
    """Instantiate a tool by ID."""
    cls = TOOL_REGISTRY.get(tool_id)
    if cls is None:
        raise KeyError(f"Unknown tool ID: {tool_id!r}. Available: {sorted(TOOL_REGISTRY)}")
    return cls(params or {})
