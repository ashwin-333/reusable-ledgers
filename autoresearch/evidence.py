"""Validated evidence-stage graphs for ledger construction.

The ledger autoresearch search space may combine deterministic tools before the
ledger-writing model sees a trajectory.  This module keeps that composition
separate from the ROC-AUC ``Pipeline``: an evidence graph may end in text,
features, flags, or a score, and every stage is rendered as bounded evidence for
the ledger prompt.

Dependencies name *stage IDs*, not tool IDs.  That lets a candidate use the
same tool more than once with different parameters without overwriting an
earlier result.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate_config import (
    CandidateConfigError,
    load_tool_catalog,
    materialize_tool_params,
)
from .tools.base import ToolResult
from .tools.registry import TOOL_REGISTRY, get_tool


DEFAULT_MAX_OUTPUT_CHARS = 6_000
MAX_OUTPUT_CHARS = 50_000
# This is a declared-budget cap, not a claim that every stage will fill its
# allocation.  Per-stage bounds alone would otherwise allow a candidate to add
# an arbitrary number of 50k stages and overwhelm the constructor context.  A
# 100k aggregate still permits the largest legacy action block plus several
# ordinary evidence stages while keeping evidence growth finite.
MAX_TOTAL_EVIDENCE_CHARS = 100_000

# Match the immutable candidate snapshot's safe-ID contract so a graph accepted
# by the outer loop cannot fail merely because the executor uses a narrower ID
# alphabet or length.
_STAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STAGE_FIELDS = {
    "stage_id",
    "tool_id",
    "params",
    "input_from",
    "max_output_chars",
}


class EvidenceConfigError(ValueError):
    """An evidence graph is malformed or cannot be executed as declared."""


@dataclass(frozen=True)
class EvidenceStage:
    """One ordered evidence stage.

    ``input_from`` is either ``"trajectory"`` or the ``stage_id`` of an
    earlier stage.  ``max_output_chars`` bounds this stage's complete rendered
    block.
    """

    stage_id: str
    tool_id: str
    params: dict[str, Any]
    input_from: str = "trajectory"
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not _STAGE_ID_RE.fullmatch(
            self.stage_id
        ):
            raise EvidenceConfigError(
                f"stage_id must match {_STAGE_ID_RE.pattern!r}"
            )
        if self.stage_id == "trajectory":
            raise EvidenceConfigError(
                "stage_id 'trajectory' is reserved for the raw input"
            )
        if not isinstance(self.tool_id, str) or not self.tool_id:
            raise EvidenceConfigError("tool_id must be a non-empty string")
        if not isinstance(self.input_from, str) or not self.input_from:
            raise EvidenceConfigError(
                "input_from must be 'trajectory' or an earlier stage_id"
            )
        if not isinstance(self.params, dict):
            raise EvidenceConfigError("params must be a JSON object")
        if isinstance(self.max_output_chars, bool) or not isinstance(
            self.max_output_chars, int
        ):
            raise EvidenceConfigError("max_output_chars must be an integer")
        if not 1 <= self.max_output_chars <= MAX_OUTPUT_CHARS:
            raise EvidenceConfigError(
                f"max_output_chars must be between 1 and {MAX_OUTPUT_CHARS}"
            )
        try:
            json.dumps(self.params, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise EvidenceConfigError(
                f"params must contain only finite JSON values: {exc}"
            ) from exc
        try:
            # Use the candidate snapshot's single source of truth. It validates
            # supported knobs and expands every implicit runtime default before
            # this stage can be executed or serialized into run metadata.
            effective_params = materialize_tool_params(self.tool_id, self.params)
        except CandidateConfigError as exc:
            raise EvidenceConfigError(str(exc)) from exc
        object.__setattr__(self, "params", effective_params)

    @classmethod
    def from_dict(cls, raw: Any, *, index: int) -> "EvidenceStage":
        where = f"stage {index}"
        if not isinstance(raw, dict):
            raise EvidenceConfigError(f"{where} must be a JSON object")

        unknown = sorted(set(raw) - _STAGE_FIELDS)
        if unknown:
            raise EvidenceConfigError(
                f"{where} has unknown field(s) {unknown}; allowed: "
                f"{sorted(_STAGE_FIELDS)}"
            )

        missing = [key for key in ("stage_id", "tool_id") if key not in raw]
        if missing:
            raise EvidenceConfigError(f"{where} is missing required field(s) {missing}")

        stage_id = raw["stage_id"]
        tool_id = raw["tool_id"]
        input_from = raw.get("input_from", "trajectory")
        params = raw.get("params", {})
        max_chars = raw.get("max_output_chars", DEFAULT_MAX_OUTPUT_CHARS)

        if not isinstance(stage_id, str) or not _STAGE_ID_RE.fullmatch(stage_id):
            raise EvidenceConfigError(
                f"{where}.stage_id must match {_STAGE_ID_RE.pattern!r}"
            )
        if stage_id == "trajectory":
            raise EvidenceConfigError(
                f"{where}.stage_id 'trajectory' is reserved for the raw input"
            )
        if not isinstance(tool_id, str) or not tool_id:
            raise EvidenceConfigError(f"{where}.tool_id must be a non-empty string")
        if not isinstance(input_from, str) or not input_from:
            raise EvidenceConfigError(
                f"{where}.input_from must be 'trajectory' or an earlier stage_id"
            )
        if not isinstance(params, dict):
            raise EvidenceConfigError(f"{where}.params must be a JSON object")
        if isinstance(max_chars, bool) or not isinstance(max_chars, int):
            raise EvidenceConfigError(f"{where}.max_output_chars must be an integer")
        if not 1 <= max_chars <= MAX_OUTPUT_CHARS:
            raise EvidenceConfigError(
                f"{where}.max_output_chars must be between 1 and {MAX_OUTPUT_CHARS}"
            )

        # Programmatic callers can pass values that JSON config files cannot.
        # Refuse them here so the candidate snapshot remains reproducible.
        try:
            json.dumps(params, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise EvidenceConfigError(
                f"{where}.params must contain only finite JSON values: {exc}"
            ) from exc

        return cls(
            stage_id=stage_id,
            tool_id=tool_id,
            params=copy.deepcopy(params),
            input_from=input_from,
            max_output_chars=max_chars,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "tool_id": self.tool_id,
            "params": copy.deepcopy(self.params),
            "input_from": self.input_from,
            "max_output_chars": self.max_output_chars,
        }


@dataclass(frozen=True)
class EvidenceStageResult:
    """The typed result and rendered prompt block from one stage."""

    stage: EvidenceStage
    result: ToolResult
    block: str


@dataclass(frozen=True)
class EvidenceRun:
    """Ordered outputs from running one evidence graph on one trajectory."""

    stages: tuple[EvidenceStageResult, ...]

    @property
    def blocks(self) -> list[str]:
        return [stage.block for stage in self.stages]

    @property
    def prompt_text(self) -> str:
        return "\n".join(self.blocks)

    @property
    def outputs(self) -> dict[str, ToolResult]:
        return {stage.stage.stage_id: stage.result for stage in self.stages}


class EvidencePlan:
    """A validated, ordered graph of evidence tools."""

    def __init__(self, stages: list[EvidenceStage] | tuple[EvidenceStage, ...]):
        self.stages = tuple(stages)
        self._declared_types = self._validate()

    @classmethod
    def from_data(cls, raw: Any) -> "EvidencePlan":
        """Build a plan from ``{"stages": [...]}`` or a bare stage list."""

        if isinstance(raw, dict):
            unknown = sorted(set(raw) - {"stages"})
            if unknown:
                raise EvidenceConfigError(
                    f"evidence config has unknown top-level field(s) {unknown}; "
                    "only 'stages' is allowed"
                )
            if "stages" not in raw:
                raise EvidenceConfigError("evidence config is missing required 'stages'")
            raw_stages = raw["stages"]
        elif isinstance(raw, list):
            raw_stages = raw
        else:
            raise EvidenceConfigError(
                "evidence config must be an object with 'stages' or a bare stage list"
            )

        if not isinstance(raw_stages, list):
            raise EvidenceConfigError("evidence config 'stages' must be a JSON list")
        stages = [
            EvidenceStage.from_dict(stage, index=i)
            for i, stage in enumerate(raw_stages, start=1)
        ]
        return cls(stages)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "EvidencePlan":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise EvidenceConfigError(f"evidence config does not exist: {path}") from exc
        except OSError as exc:
            raise EvidenceConfigError(f"cannot read evidence config {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise EvidenceConfigError(
                f"invalid JSON in evidence config {path}: {exc}"
            ) from exc
        return cls.from_data(raw)

    def to_dict(self) -> dict[str, Any]:
        return {"stages": [stage.to_dict() for stage in self.stages]}

    def _validate(self) -> dict[str, str]:
        total_output_chars = sum(stage.max_output_chars for stage in self.stages)
        if total_output_chars > MAX_TOTAL_EVIDENCE_CHARS:
            raise EvidenceConfigError(
                "evidence graph declares "
                f"{total_output_chars} aggregate output characters, exceeding "
                f"the {MAX_TOTAL_EVIDENCE_CHARS} character total maximum"
            )

        seen: dict[str, str] = {}
        catalog = load_tool_catalog().by_id
        for index, stage in enumerate(self.stages, start=1):
            where = f"stage {index} ({stage.stage_id!r})"
            if stage.stage_id in seen:
                raise EvidenceConfigError(f"duplicate stage_id {stage.stage_id!r}")
            catalog_entry = catalog.get(stage.tool_id)
            if catalog_entry is None:
                raise EvidenceConfigError(
                    f"{where} names unknown tool {stage.tool_id!r}; "
                    "register a versioned implementation before using new tool ideas"
                )
            if catalog_entry.status != "runnable":
                raise EvidenceConfigError(
                    f"{where} cannot use {stage.tool_id}: status="
                    f"{catalog_entry.status}; {catalog_entry.status_reason}"
                )
            if stage.tool_id not in TOOL_REGISTRY:
                raise EvidenceConfigError(
                    f"{where} uses catalog-runnable tool {stage.tool_id}, but it "
                    "is absent from the runtime registry"
                )
            if not re.fullmatch(r"T\d{2}", stage.tool_id):
                raise EvidenceConfigError(
                    f"{where} names {stage.tool_id!r}, which is a scorer rather than "
                    "a candidate evidence tool"
                )

            try:
                tool = get_tool(stage.tool_id, stage.params)
            except Exception as exc:  # constructor-level parameter validation
                raise EvidenceConfigError(
                    f"{where} could not initialize {stage.tool_id}: {exc}"
                ) from exc
            input_type = getattr(tool, "input_type", None)
            output_type = getattr(tool, "output_type", None)
            if not isinstance(input_type, str) or not isinstance(output_type, str):
                raise EvidenceConfigError(
                    f"{where} tool {stage.tool_id} has no valid input/output type"
                )

            if stage.input_from == "trajectory":
                source_type = "trajectory"
            elif stage.input_from in seen:
                source_type = seen[stage.input_from]
            else:
                raise EvidenceConfigError(
                    f"{where}.input_from={stage.input_from!r} must name 'trajectory' "
                    "or an earlier stage_id"
                )

            if input_type != "any" and source_type != "any" and input_type != source_type:
                raise EvidenceConfigError(
                    f"{where} expects input_type={input_type!r}, but "
                    f"{stage.input_from!r} produces {source_type!r}"
                )
            seen[stage.stage_id] = output_type
        return seen

    @property
    def requires_llm(self) -> bool:
        """Whether a currently supported stage exposes an LLM setter."""

        return any(
            hasattr(get_tool(stage.tool_id, stage.params), "set_llm")
            for stage in self.stages
        )

    @staticmethod
    def _new_context() -> dict[str, Any]:
        # Never pass evaluator labels (construct/is_benign) into evidence tools.
        return {"manifest": {}, "outputs": {}}

    @staticmethod
    def _resolve_input(
        stage: EvidenceStage, trajectory: dict, context: dict[str, Any]
    ) -> Any:
        if stage.input_from == "trajectory":
            return trajectory
        # Dependencies were checked at construction; retain a clear runtime
        # error in case a caller mutates a context outside this class.
        try:
            return context["outputs"][stage.input_from].output
        except KeyError as exc:
            raise EvidenceConfigError(
                f"stage {stage.stage_id!r} cannot resolve input_from "
                f"{stage.input_from!r}"
            ) from exc

    @staticmethod
    def _check_result(stage: EvidenceStage, result: Any, expected_type: str) -> ToolResult:
        if not isinstance(result, ToolResult):
            raise TypeError(
                f"evidence stage {stage.stage_id!r} ({stage.tool_id}) returned "
                f"{type(result).__name__}, expected ToolResult"
            )
        if result.output_type != expected_type:
            raise TypeError(
                f"evidence stage {stage.stage_id!r} ({stage.tool_id}) declared "
                f"output_type={expected_type!r} but returned {result.output_type!r}"
            )
        return result

    def run(self, trajectory: dict, *, llm_adapter: Any | None = None) -> EvidenceRun:
        """Run the graph synchronously (sufficient for current evidence tools)."""

        context = self._new_context()
        completed: list[EvidenceStageResult] = []
        for stage in self.stages:
            tool = get_tool(stage.tool_id, stage.params)
            if hasattr(tool, "set_llm"):
                if llm_adapter is None:
                    raise EvidenceConfigError(
                        f"evidence stage {stage.stage_id!r} requires an LLM adapter"
                    )
                tool.set_llm(llm_adapter)
            if callable(getattr(tool, "process_async", None)):
                raise RuntimeError(
                    f"evidence stage {stage.stage_id!r} is async; use run_async()"
                )
            input_data = self._resolve_input(stage, trajectory, context)
            result = self._check_result(
                stage,
                tool.process(input_data, context),
                self._declared_types[stage.stage_id],
            )
            result.metadata.setdefault("tool_name", getattr(tool, "tool_name", stage.tool_id))
            context["outputs"][stage.stage_id] = result
            completed.append(
                EvidenceStageResult(stage, result, render_evidence(stage, result))
            )
        return EvidenceRun(tuple(completed))

    async def run_async(
        self, trajectory: dict, *, llm_adapter: Any | None = None
    ) -> EvidenceRun:
        """Run the graph in order, awaiting an async stage when necessary."""

        context = self._new_context()
        completed: list[EvidenceStageResult] = []
        for stage in self.stages:
            tool = get_tool(stage.tool_id, stage.params)
            if hasattr(tool, "set_llm"):
                if llm_adapter is None:
                    raise EvidenceConfigError(
                        f"evidence stage {stage.stage_id!r} requires an LLM adapter"
                    )
                tool.set_llm(llm_adapter)
            input_data = self._resolve_input(stage, trajectory, context)
            process_async = getattr(tool, "process_async", None)
            if callable(process_async):
                raw_result = await process_async(input_data, context)
            else:
                raw_result = tool.process(input_data, context)
            result = self._check_result(
                stage, raw_result, self._declared_types[stage.stage_id]
            )
            result.metadata.setdefault("tool_name", getattr(tool, "tool_name", stage.tool_id))
            context["outputs"][stage.stage_id] = result
            completed.append(
                EvidenceStageResult(stage, result, render_evidence(stage, result))
            )
        return EvidenceRun(tuple(completed))


def legacy_evidence_plan(
    tool_ids: list[str] | tuple[str, ...] | None = None,
    *,
    inject_actions: bool = False,
) -> EvidencePlan:
    """Translate old injection flags into independent trajectory stages.

    Legacy tools intentionally do not depend on each other.  A features-only
    tool such as T05 therefore fails validation here instead of being silently
    handed the wrong input; use an evidence config to declare T03 -> T05.
    """

    normalized = [str(tool).strip().upper() for tool in (tool_ids or ()) if str(tool).strip()]
    if inject_actions and "T03" not in normalized:
        normalized.insert(0, "T03")

    counts: dict[str, int] = {}
    stages: list[EvidenceStage] = []
    for tool_id in normalized:
        counts[tool_id] = counts.get(tool_id, 0) + 1
        suffix = "" if counts[tool_id] == 1 else f"_{counts[tool_id]}"
        # Preserve the old T04 6k bound.  T03's action list was unbounded, so
        # use the global safety maximum rather than the smaller generic default.
        max_chars = MAX_OUTPUT_CHARS if tool_id == "T03" else DEFAULT_MAX_OUTPUT_CHARS
        stages.append(
            EvidenceStage(
                stage_id=f"{tool_id}{suffix}",
                tool_id=tool_id,
                params={},
                input_from="trajectory",
                max_output_chars=max_chars,
            )
        )
    return EvidencePlan(stages)


def render_evidence(stage: EvidenceStage, result: ToolResult) -> str:
    """Render one result as a readable, size-bounded prompt block."""

    output = result.output
    specialized = True
    if stage.tool_id == "T03" and isinstance(output, dict):
        text = _render_t03(output)
    elif stage.tool_id == "T04":
        text = "### EVIDENCE: TRANSCRIPT AS STEP TABLE\n\n" + str(output) + "\n"
    elif stage.tool_id == "T20" and isinstance(output, dict):
        text = _render_t20(output)
    elif stage.tool_id in {"T21", "T22"} and isinstance(output, dict):
        text = _render_scanner(stage.tool_id, output)
    else:
        specialized = False
        safe = _json_safe(output)
        payload = json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=True)
        text = (
            f"### EVIDENCE STAGE: {stage.stage_id} ({stage.tool_id})\n\n"
            f"Input: {stage.input_from}  Output type: {result.output_type}\n\n"
            f"{payload}\n"
        )
    if specialized and stage.stage_id != stage.tool_id:
        text = (
            f"### EVIDENCE STAGE: {stage.stage_id} ({stage.tool_id})\n\n"
            f"Input: {stage.input_from}  Output type: {result.output_type}\n\n"
            f"{text}"
        )
    return _truncate(text, stage.max_output_chars)


def extracted_action_block(
    trajectory: dict, *, max_output_chars: int = MAX_OUTPUT_CHARS
) -> str:
    """Backward-compatible T03 action-list helper used by existing tests."""

    plan = EvidencePlan(
        [
            EvidenceStage(
                stage_id="T03",
                tool_id="T03",
                params={},
                input_from="trajectory",
                max_output_chars=max_output_chars,
            )
        ]
    )
    return plan.run(trajectory).blocks[0]


def _render_t03(extracted: dict[str, Any]) -> str:
    records = extracted.get("records") or []
    lines = [
        "### EXTRACTED ACTION LIST",
        "",
        "Every tool call in this trajectory, extracted deterministically. This "
        "list is complete and ordered; treat it as authoritative for WHAT was "
        "called, and the transcript below as the source for arguments, results "
        "and intent.",
        "",
        f"Total tool calls: {extracted.get('n_tool_calls', len(records))}"
        f"  (state-changing: {extracted.get('n_state_changing', 0)},"
        f" read-only: {extracted.get('n_read_only', 0)})",
        "",
    ]
    for record in records:
        args_text = (record.get("args_text") or "").strip().replace("\n", " ")
        if len(args_text) > 200:
            args_text = args_text[:200] + " ...[truncated]"
        step = record.get("step")
        step_text = str(step) if step is not None else "?"
        lines.append(
            f"  {step_text:>4}. {record.get('tool_name', '?')}"
            + (f"  args: {args_text}" if args_text else "")
        )
    lines.append("")
    return "\n".join(lines)


def _render_t20(output: dict[str, Any]) -> str:
    lines = [
        "### EVIDENCE: STATE-CHANGE SUMMARY",
        "",
        f"State-changing operations detected: {output.get('n_changes', 0)} "
        f"(consequential: {output.get('n_consequential', 0)})",
        f"By type: {output.get('change_type_counts', {})}",
    ]
    for change in (output.get("changes") or [])[:25]:
        lines.append(
            f"  step {change.get('step')}: {change.get('type')} "
            f"({change.get('match')})"
        )
    for entity in (output.get("entities") or [])[:15]:
        lines.append(
            f"  entity at step {entity.get('step')}: {entity.get('type')} = "
            f"{entity.get('value')}"
        )
    lines.append("")
    return "\n".join(lines)


def _render_scanner(tool_id: str, output: dict[str, Any]) -> str:
    if tool_id == "T22":
        title = "### EVIDENCE: SENSITIVE-DATA FLAGS"
        caveat = "Matches are not proof of misuse; routine emails/IBANs raise counts."
    else:
        title = "### EVIDENCE: PERMISSION / PRIVILEGE FLAGS"
        caveat = "Matches identify possible access changes; verify them against context."
    lines = [
        title,
        "",
        f"Pattern matches: {output.get('n_flags', 0)}  "
        f"by type: {output.get('type_counts', {})}",
        caveat,
    ]
    for flag in (output.get("flags") or [])[:20]:
        lines.append(
            f"  step {flag.get('step')} ({flag.get('role')}): "
            f"{flag.get('type')} = {flag.get('match')}"
        )
    lines.append("")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n...[evidence truncated]"
    if max_chars <= len(marker):
        return text[:max_chars]
    return text[: max_chars - len(marker)] + marker


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert arbitrary tool output to a finite, visibly bounded JSON shape."""

    if depth >= 8:
        return "...[maximum nesting depth reached]"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        items = list(value.items())
        rendered = {
            str(key): _json_safe(item, depth=depth + 1)
            for key, item in items[:100]
        }
        if len(items) > 100:
            rendered["...[truncated entries]"] = len(items) - 100
        return rendered
    if isinstance(value, (list, tuple)):
        rendered = [_json_safe(item, depth=depth + 1) for item in value[:100]]
        if len(value) > 100:
            rendered.append(f"...[{len(value) - 100} more items]")
        return rendered
    if isinstance(value, set):
        items = sorted((str(item) for item in value))
        return _json_safe(items, depth=depth + 1)
    return str(value)
