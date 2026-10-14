"""Canonical, reproducible configurations for ledger-constructor candidates.

The autoresearch agent is allowed to change several things at once.  That
freedom only remains scientifically useful if every evaluated candidate is a
fully materialised, immutable snapshot.  This module owns that snapshot format
and validates an evidence-stage graph before any trajectory or model call is
started.

The candidate-tool catalog is a checked-in snapshot of the team's Google Sheet.
It distinguishes ideas in the search space from tools that are actually safe to
execute in the current runtime.  A tool is accepted in an evidence graph only
when the snapshot marks it ``runnable`` *and* a matching registry class exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


CANDIDATE_SCHEMA_VERSION = 1
DEFAULT_MAX_OUTPUT_CHARS = 6_000
DEFAULT_TOOL_CATALOG_PATH = Path(__file__).with_name("candidate_tools.json")

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STAGE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_TOOL_ID = re.compile(r"^T[0-9]{2}$")
_RUNTIME_STATUSES = {"runnable", "unavailable", "unimplemented", "reserved"}
_STRUCTURES = {"single", "two_pass"}

# Defaults the current evidence runtime actually uses. The spreadsheet's
# parameter column is aspirational; accepting a name that code ignores, or
# hashing params={} while relying on mutable code defaults, would make the
# search irreproducible. Every candidate expands these defaults before hashing.
RUNTIME_TOOL_DEFAULTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "T03": MappingProxyType({}),
        "T04": MappingProxyType(
            {"include_output": False, "max_output_chars": 120, "format": "markdown"}
        ),
        "T05": MappingProxyType({"normalize_by_length": True}),
        "T06": MappingProxyType({"n_values": (2, 3, 4), "min_repeat_count": 2}),
        "T07": MappingProxyType({"window_size": 10}),
        # Evidence graphs do not provide a fitted population context, so only
        # the heuristic threshold changes T09/T10 behavior in this runtime.
        "T09": MappingProxyType({"heuristic_threshold": 0.4}),
        "T10": MappingProxyType({"heuristic_threshold": 0.4}),
        "T11": MappingProxyType({"z_threshold": 2.0}),
        "T12": MappingProxyType({"order": 1, "smoothing": 0.1}),
        "T17": MappingProxyType(
            {"terms": ("requested", "only", "never", "must not", "recipient", "target", "amount", "permission", "created", "updated", "deleted", "sent"),
             "window_turns": 8, "stride_turns": 4, "top_k": 4, "excerpt_chars": 240}
        ),
        "T20": MappingProxyType({}),
        "T21": MappingProxyType({}),
        "T22": MappingProxyType(
            {"include_pii": True, "include_secrets": True, "include_exfil": True}
        ),
        "T31": MappingProxyType(
            {"roles": ("user", "assistant", "tool"),
             "fields": ("content", "tool_calls"), "contains_any": (),
             "max_turns": 30, "max_chars_per_field": 300}
        ),
    }
)
RUNTIME_TOOL_PARAMS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        tool_id: frozenset(defaults)
        for tool_id, defaults in RUNTIME_TOOL_DEFAULTS.items()
    }
)

# Machine-readable descriptions shown to the autonomous researcher. Validation
# below remains authoritative; this map makes the same supported knobs usable
# without trial-and-error rejections.
RUNTIME_TOOL_CONSTRAINTS: Mapping[str, Mapping[str, Mapping[str, Any]]] = (
    MappingProxyType(
        {
            "T03": MappingProxyType({}),
            "T04": MappingProxyType(
                {
                    "include_output": MappingProxyType({"type": "boolean"}),
                    "max_output_chars": MappingProxyType(
                        {"type": "integer", "minimum": 1}
                    ),
                    "format": MappingProxyType(
                        {"type": "string", "enum": ("markdown", "csv")}
                    ),
                }
            ),
            "T05": MappingProxyType(
                {"normalize_by_length": MappingProxyType({"type": "boolean"})}
            ),
            "T06": MappingProxyType(
                {
                    "n_values": MappingProxyType(
                        {
                            "type": "array[integer]",
                            "min_items": 1,
                            "item_minimum": 1,
                        }
                    ),
                    "min_repeat_count": MappingProxyType(
                        {"type": "integer", "minimum": 1}
                    ),
                }
            ),
            "T07": MappingProxyType(
                {"window_size": MappingProxyType({"type": "integer", "minimum": 1})}
            ),
            "T09": MappingProxyType(
                {"heuristic_threshold": MappingProxyType({"type": "number"})}
            ),
            "T10": MappingProxyType(
                {"heuristic_threshold": MappingProxyType({"type": "number"})}
            ),
            "T11": MappingProxyType(
                {
                    "z_threshold": MappingProxyType(
                        {"type": "number", "exclusive_minimum": 0}
                    )
                }
            ),
            "T12": MappingProxyType(
                {
                    "order": MappingProxyType({"type": "integer", "minimum": 1}),
                    "smoothing": MappingProxyType(
                        {"type": "number", "exclusive_minimum": 0}
                    ),
                }
            ),
            "T17": MappingProxyType(
                {
                    "terms": MappingProxyType({"type": "array[string]", "min_items": 1}),
                    "window_turns": MappingProxyType({"type": "integer", "minimum": 1}),
                    "stride_turns": MappingProxyType({"type": "integer", "minimum": 1}),
                    "top_k": MappingProxyType({"type": "integer", "minimum": 1}),
                    "excerpt_chars": MappingProxyType({"type": "integer", "minimum": 1}),
                }
            ),
            "T20": MappingProxyType({}),
            "T21": MappingProxyType({}),
            "T22": MappingProxyType(
                {
                    "include_pii": MappingProxyType({"type": "boolean"}),
                    "include_secrets": MappingProxyType({"type": "boolean"}),
                    "include_exfil": MappingProxyType({"type": "boolean"}),
                }
            ),
            "T31": MappingProxyType(
                {
                    "roles": MappingProxyType({"type": "array[string]"}),
                    "fields": MappingProxyType({"type": "array[string]"}),
                    "contains_any": MappingProxyType({"type": "array[string]"}),
                    "max_turns": MappingProxyType({"type": "integer", "minimum": 1}),
                    "max_chars_per_field": MappingProxyType({"type": "integer", "minimum": 1}),
                }
            ),
        }
    )
)


class CandidateConfigError(ValueError):
    """Raised when a candidate cannot be reproduced or safely executed."""


def _require_safe_id(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise CandidateConfigError(
            f"{field_name} must match {_SAFE_ID.pattern!r}; got {value!r}"
        )


def _freeze_json(value: Any, *, path: str) -> Any:
    """Validate a JSON value and recursively make it immutable."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CandidateConfigError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        keys = list(value)
        for key in keys:
            if not isinstance(key, str):
                raise CandidateConfigError(f"{path} has a non-string key {key!r}")
        for key in sorted(keys):
            frozen[key] = _freeze_json(value[key], path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise CandidateConfigError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def materialize_tool_params(
    tool_id: str, params: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return explicit effective params for a runnable evidence tool.

    Explicit supported values win over defaults. Unsupported parameters fail
    closed instead of becoming no-op search dimensions. Availability, graph
    placement, and I/O types are validated separately by
    :func:`validate_evidence_stages`.
    """
    if not isinstance(tool_id, str) or not _TOOL_ID.fullmatch(tool_id):
        raise CandidateConfigError(
            f"tool_id must match {_TOOL_ID.pattern!r}; got {tool_id!r}"
        )
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise CandidateConfigError("tool params must be an object")

    supplied = _thaw_json(_freeze_json(params, path=f"params[{tool_id}]"))
    validate_tool_params(tool_id, supplied)
    defaults = RUNTIME_TOOL_DEFAULTS.get(tool_id, {})
    materialized: dict[str, Any] = {
        key: _thaw_json(_freeze_json(value, path=f"defaults[{tool_id}].{key}"))
        for key, value in defaults.items()
    }
    materialized.update(supplied)
    validate_tool_params(tool_id, materialized)
    # Return canonical key order and fresh nested containers.  EvidenceStage
    # freezes this value again, so neither defaults nor caller input can mutate
    # a materialized candidate after hashing.
    return {
        key: _thaw_json(
            _freeze_json(materialized[key], path=f"materialized[{tool_id}].{key}")
        )
        for key in sorted(materialized)
    }


def validate_tool_params(tool_id: str, params: Mapping[str, Any]) -> None:
    """Validate only knobs that have an effect in the registered implementation."""
    allowed = RUNTIME_TOOL_PARAMS.get(tool_id)
    if allowed is None:
        return  # availability/registration is validated separately
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise CandidateConfigError(
            f"tool {tool_id} does not implement parameter(s) {unknown}; "
            f"supported runtime parameters: {sorted(allowed)}"
        )

    def require_bool(name: str) -> None:
        if name in params and not isinstance(params[name], bool):
            raise CandidateConfigError(f"{tool_id}.{name} must be boolean")

    def require_positive_int(name: str) -> None:
        if name not in params:
            return
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CandidateConfigError(f"{tool_id}.{name} must be a positive integer")

    def require_number(name: str, *, positive: bool = False) -> None:
        if name not in params:
            return
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CandidateConfigError(f"{tool_id}.{name} must be numeric")
        if not math.isfinite(float(value)) or (positive and float(value) <= 0):
            qualifier = "positive and finite" if positive else "finite"
            raise CandidateConfigError(f"{tool_id}.{name} must be {qualifier}")

    for name in (
        "include_output",
        "normalize_by_length",
        "include_pii",
        "include_secrets",
        "include_exfil",
    ):
        require_bool(name)
    for name in ("max_output_chars", "min_repeat_count", "window_size", "order", "window_turns", "stride_turns", "top_k", "excerpt_chars", "max_turns", "max_chars_per_field"):
        require_positive_int(name)
    for name in ("z_threshold", "smoothing"):
        require_number(name, positive=True)
    require_number("heuristic_threshold")

    if "n_values" in params and (
        not isinstance(params["n_values"], (list, tuple))
        or not params["n_values"]
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in params["n_values"]
        )
    ):
        raise CandidateConfigError(
            f"{tool_id}.n_values must be a non-empty list of positive integers"
        )
    if "format" in params and params["format"] not in {"markdown", "csv"}:
        raise CandidateConfigError(f"{tool_id}.format must be 'markdown' or 'csv'")
    if "terms" in params and (
        not isinstance(params["terms"], (list, tuple))
        or not 1 <= len(params["terms"]) <= 32
        or any(not isinstance(term, str) or not 1 <= len(term) <= 80 for term in params["terms"])
    ):
        raise CandidateConfigError(f"{tool_id}.terms must have 1–32 literal strings of 1–80 characters")
    if "roles" in params and (
        not isinstance(params["roles"], (list, tuple))
        or not params["roles"]
        or any(role not in {"system", "user", "assistant", "tool"} for role in params["roles"])
    ):
        raise CandidateConfigError(f"{tool_id}.roles must list valid conversation roles")
    if "fields" in params and (
        not isinstance(params["fields"], (list, tuple))
        or not params["fields"]
        or any(field not in {"content", "tool_calls"} for field in params["fields"])
    ):
        raise CandidateConfigError(f"{tool_id}.fields must list content/tool_calls")
    if "contains_any" in params and (
        not isinstance(params["contains_any"], (list, tuple))
        or len(params["contains_any"]) > 32
        or any(not isinstance(term, str) or not 1 <= len(term) <= 80 for term in params["contains_any"])
    ):
        raise CandidateConfigError(f"{tool_id}.contains_any must have at most 32 literal strings")


@dataclass(frozen=True)
class PromptSnapshot:
    """Exact prompt bytes plus a verified SHA-256 digest."""

    name: str
    content: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise CandidateConfigError("prompt name must be a non-empty string")
        if len(self.name) > 256 or any(ord(char) < 32 for char in self.name):
            raise CandidateConfigError("prompt name is too long or has control characters")
        if not isinstance(self.content, str):
            raise CandidateConfigError("prompt content must be a string")
        expected = _sha256_text(self.content)
        if self.sha256 != expected:
            raise CandidateConfigError(
                f"prompt {self.name!r} hash mismatch: expected {expected}, "
                f"got {self.sha256!r}"
            )

    @classmethod
    def from_content(cls, *, name: str, content: str) -> "PromptSnapshot":
        return cls(name=name, content=content, sha256=_sha256_text(content))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromptSnapshot":
        if not isinstance(data, Mapping):
            raise CandidateConfigError("prompt snapshot must be an object")
        try:
            name = data["name"]
            content = data["content"]
        except KeyError as exc:
            raise CandidateConfigError(
                f"prompt snapshot is missing {exc.args[0]!r}"
            ) from exc
        if not isinstance(content, str):
            raise CandidateConfigError("prompt content must be a string")
        supplied_hash = data.get("sha256", data.get("hash"))
        if supplied_hash is None:
            supplied_hash = _sha256_text(content)
        return cls(name=name, content=content, sha256=supplied_hash)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "content": self.content, "sha256": self.sha256}


@dataclass(frozen=True)
class EvidenceStage:
    """One node in an ordered evidence graph.

    ``stage_id`` is the graph identity; ``tool_id`` selects an implementation.
    Consequently, the same tool can be used more than once with different
    parameters as long as each use has a distinct stage ID.
    """

    stage_id: str
    tool_id: str
    params: Mapping[str, Any] = field(default_factory=dict)
    input_from: str = "trajectory"
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not _STAGE_ID.fullmatch(
            self.stage_id
        ):
            raise CandidateConfigError(
                f"stage_id must match {_STAGE_ID.pattern!r}; got {self.stage_id!r}"
            )
        if self.stage_id == "trajectory":
            raise CandidateConfigError(
                "stage_id 'trajectory' is reserved for the raw input"
            )
        if not isinstance(self.tool_id, str) or not _TOOL_ID.fullmatch(self.tool_id):
            raise CandidateConfigError(
                f"tool_id must match {_TOOL_ID.pattern!r}; got {self.tool_id!r}"
            )
        if self.input_from != "trajectory":
            if not isinstance(self.input_from, str) or not _STAGE_ID.fullmatch(
                self.input_from
            ):
                raise CandidateConfigError(
                    "input_from must be 'trajectory' or a safe stage_id"
                )
        if (
            isinstance(self.max_output_chars, bool)
            or not isinstance(self.max_output_chars, int)
            or not 1 <= self.max_output_chars <= 50_000
        ):
            raise CandidateConfigError(
                "max_output_chars must be an integer between 1 and 50,000"
            )
        if not isinstance(self.params, Mapping):
            raise CandidateConfigError("stage params must be an object")
        object.__setattr__(
            self, "params", _freeze_json(self.params, path=f"stage[{self.stage_id}].params")
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceStage":
        if not isinstance(data, Mapping):
            raise CandidateConfigError("each evidence stage must be an object")
        allowed = {
            "stage_id",
            "tool_id",
            "tool",  # narrow compatibility alias
            "params",
            "input_from",
            "max_output_chars",
        }
        unknown = set(data) - allowed
        if unknown:
            raise CandidateConfigError(
                f"unknown evidence-stage fields: {sorted(unknown)}"
            )
        tool_id = data.get("tool_id", data.get("tool"))
        if "stage_id" not in data:
            raise CandidateConfigError("evidence stage is missing 'stage_id'")
        if tool_id is None:
            raise CandidateConfigError("evidence stage is missing 'tool_id'")
        return cls(
            stage_id=data["stage_id"],
            tool_id=tool_id,
            params=data.get("params", {}),
            input_from=data.get("input_from", "trajectory"),
            max_output_chars=data.get(
                "max_output_chars", DEFAULT_MAX_OUTPUT_CHARS
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "tool_id": self.tool_id,
            "params": _thaw_json(self.params),
            "input_from": self.input_from,
            "max_output_chars": self.max_output_chars,
        }


@dataclass(frozen=True)
class ToolCatalogEntry:
    """One spreadsheet idea plus its current local-runtime contract."""

    tool_id: str
    name: str
    category: str
    parameter_hints: tuple[str, ...]
    sheet: Mapping[str, Any]
    runtime_status: str
    runtime_reason: str
    input_type: str | None
    output_type: str | None

    # Short aliases are the public names used by the loop-facing catalog API.
    @property
    def status(self) -> str:
        return self.runtime_status

    @property
    def status_reason(self) -> str:
        return self.runtime_reason

    @property
    def key_parameters(self) -> str:
        return "; ".join(self.parameter_hints)

    @property
    def runtime_parameter_names(self) -> tuple[str, ...]:
        return tuple(sorted(RUNTIME_TOOL_PARAMS.get(self.tool_id, ())))

    @property
    def runtime_parameter_schema(self) -> dict[str, dict[str, Any]]:
        defaults = RUNTIME_TOOL_DEFAULTS.get(self.tool_id, {})
        constraints = RUNTIME_TOOL_CONSTRAINTS.get(self.tool_id, {})
        return {
            name: {
                "default": _thaw_json(defaults[name]),
                **_thaw_json(constraints.get(name, {})),
            }
            for name in sorted(defaults)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolCatalogEntry":
        runtime = data.get("runtime")
        if not isinstance(runtime, Mapping):
            raise CandidateConfigError(
                f"catalog entry {data.get('tool_id')!r} has no runtime object"
            )
        status = runtime.get("status")
        if status not in _RUNTIME_STATUSES:
            raise CandidateConfigError(
                f"catalog entry {data.get('tool_id')!r} has invalid runtime "
                f"status {status!r}"
            )
        hints = data.get("parameter_hints", [])
        if not isinstance(hints, list) or not all(isinstance(x, str) for x in hints):
            raise CandidateConfigError("parameter_hints must be a list of strings")
        return cls(
            tool_id=data["tool_id"],
            name=data["name"],
            category=data["category"],
            parameter_hints=tuple(hints),
            sheet=_freeze_json(data.get("sheet", {}), path="catalog.sheet"),
            runtime_status=status,
            runtime_reason=str(runtime.get("reason", "")),
            input_type=runtime.get("input_type"),
            output_type=runtime.get("output_type"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "category": self.category,
            "parameter_hints": list(self.parameter_hints),
            "sheet": _thaw_json(self.sheet),
            "runtime": {
                "status": self.runtime_status,
                "reason": self.runtime_reason,
                "input_type": self.input_type,
                "output_type": self.output_type,
            },
        }


@dataclass(frozen=True)
class ToolCatalog:
    schema_version: int
    source: Mapping[str, Any]
    tools: tuple[ToolCatalogEntry, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise CandidateConfigError(
                f"unsupported tool-catalog schema {self.schema_version!r}"
            )
        ids = [entry.tool_id for entry in self.tools]
        expected = {f"T{index:02d}" for index in range(1, 31)}
        if len(ids) != len(set(ids)):
            raise CandidateConfigError("tool catalog contains duplicate IDs")
        if not expected.issubset(set(ids)):
            raise CandidateConfigError(
                "tool catalog must contain the original T01-T30; "
                f"missing={sorted(expected - set(ids))}, "
                f"extra={sorted(set(ids) - expected)}"
            )
        if any(not _TOOL_ID.fullmatch(tool_id) for tool_id in ids):
            raise CandidateConfigError("tool catalog contains a malformed tool ID")
        object.__setattr__(self, "source", _freeze_json(self.source, path="catalog.source"))

    @property
    def by_id(self) -> Mapping[str, ToolCatalogEntry]:
        return MappingProxyType({entry.tool_id: entry for entry in self.tools})

    def require(self, tool_id: str) -> ToolCatalogEntry:
        try:
            return self.by_id[tool_id]
        except KeyError as exc:
            raise CandidateConfigError(
                f"tool {tool_id!r} is not in the Candidate Tools snapshot"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": _thaw_json(self.source),
            "tools": [entry.to_dict() for entry in self.tools],
        }


def load_tool_catalog(path: str | Path | None = None) -> ToolCatalog:
    """Load and validate the checked-in Candidate Tools snapshot."""
    catalog_path = Path(path) if path is not None else DEFAULT_TOOL_CATALOG_PATH
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateConfigError(
            f"cannot load tool catalog {catalog_path}: {exc}"
        ) from exc
    if not isinstance(data, Mapping):
        raise CandidateConfigError("tool catalog root must be an object")
    tools = data.get("tools")
    if not isinstance(tools, list):
        raise CandidateConfigError("tool catalog 'tools' must be a list")
    return ToolCatalog(
        schema_version=data.get("schema_version"),
        source=data.get("source", {}),
        tools=tuple(ToolCatalogEntry.from_dict(entry) for entry in tools),
    )


def _registry_interfaces() -> Mapping[str, tuple[str, str]]:
    """Return registry contracts without constructing or running any tool."""
    from .tools.registry import TOOL_REGISTRY

    return MappingProxyType(
        {
            tool_id: (
                getattr(tool_class, "input_type", ""),
                getattr(tool_class, "output_type", ""),
            )
            for tool_id, tool_class in TOOL_REGISTRY.items()
            if _TOOL_ID.fullmatch(tool_id)
        }
    )


def validate_evidence_stages(
    stages: Sequence[EvidenceStage],
    *,
    catalog: ToolCatalog | None = None,
    registry_interfaces: Mapping[str, tuple[str, str]] | None = None,
) -> None:
    """Fail fast on unavailable tools, bad topology, or incompatible types.

    Validation is intentionally independent of graph execution and does not
    instantiate a tool.  This makes it safe to call before starting a paid
    subprocess or making an API request.
    """
    active_catalog = catalog or load_tool_catalog()
    interfaces = (
        registry_interfaces
        if registry_interfaces is not None
        else _registry_interfaces()
    )
    seen_outputs: dict[str, str] = {}

    for stage in stages:
        if not isinstance(stage, EvidenceStage):
            raise CandidateConfigError("stages must contain EvidenceStage objects")
        if stage.stage_id in seen_outputs:
            raise CandidateConfigError(f"duplicate stage_id {stage.stage_id!r}")

        entry = active_catalog.require(stage.tool_id)
        if entry.runtime_status != "runnable":
            detail = f": {entry.runtime_reason}" if entry.runtime_reason else ""
            raise CandidateConfigError(
                f"tool {stage.tool_id} is {entry.runtime_status}, not runnable{detail}"
            )
        validate_tool_params(stage.tool_id, stage.params)

        registry_contract = interfaces.get(stage.tool_id)
        if registry_contract is None:
            raise CandidateConfigError(
                f"tool {stage.tool_id} is marked runnable but absent from TOOL_REGISTRY"
            )
        registry_input, registry_output = registry_contract
        if (registry_input, registry_output) != (entry.input_type, entry.output_type):
            raise CandidateConfigError(
                f"tool {stage.tool_id} registry contract "
                f"{registry_input!r}->{registry_output!r} does not match catalog "
                f"{entry.input_type!r}->{entry.output_type!r}"
            )

        if stage.input_from == "trajectory":
            source_type = "trajectory"
        else:
            try:
                source_type = seen_outputs[stage.input_from]
            except KeyError as exc:
                raise CandidateConfigError(
                    f"stage {stage.stage_id!r} input_from {stage.input_from!r} must "
                    "refer to 'trajectory' or an earlier stage_id"
                ) from exc

        if registry_input not in {"any", source_type} and source_type != "any":
            raise CandidateConfigError(
                f"stage {stage.stage_id!r} ({stage.tool_id}) expects "
                f"{registry_input!r} input but {stage.input_from!r} emits "
                f"{source_type!r}"
            )
        seen_outputs[stage.stage_id] = registry_output


@dataclass(frozen=True)
class CandidateConfig:
    """Immutable, fully materialised candidate lineage record."""

    schema_version: int
    candidate_id: str
    parent_candidate_id: str | None
    prompt: PromptSnapshot
    evidence_stages: tuple[EvidenceStage, ...]
    structure: str
    attribution_prompt: PromptSnapshot | None
    hypothesis: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != CANDIDATE_SCHEMA_VERSION
        ):
            raise CandidateConfigError(
                f"unsupported candidate schema {self.schema_version!r}; "
                f"expected {CANDIDATE_SCHEMA_VERSION}"
            )
        _require_safe_id(self.candidate_id, field_name="candidate_id")
        if self.parent_candidate_id is not None:
            _require_safe_id(
                self.parent_candidate_id, field_name="parent_candidate_id"
            )
            if self.parent_candidate_id == self.candidate_id:
                raise CandidateConfigError("candidate cannot be its own parent")
        if not isinstance(self.prompt, PromptSnapshot):
            raise CandidateConfigError("prompt must be a PromptSnapshot")
        if self.attribution_prompt is not None and not isinstance(
            self.attribution_prompt, PromptSnapshot
        ):
            raise CandidateConfigError(
                "attribution_prompt must be a PromptSnapshot or None"
            )
        stages = tuple(self.evidence_stages)
        for stage in stages:
            if not isinstance(stage, EvidenceStage):
                raise CandidateConfigError(
                    "evidence_stages must contain EvidenceStage objects"
                )
        stages = tuple(
            replace(
                stage,
                params=materialize_tool_params(stage.tool_id, stage.params),
            )
            for stage in stages
        )
        object.__setattr__(self, "evidence_stages", stages)
        if self.structure not in _STRUCTURES:
            raise CandidateConfigError(
                f"structure must be one of {sorted(_STRUCTURES)}; got {self.structure!r}"
            )
        if self.structure == "two_pass" and self.attribution_prompt is None:
            raise CandidateConfigError(
                "structure='two_pass' requires an attribution_prompt snapshot"
            )
        if self.structure == "single" and self.attribution_prompt is not None:
            raise CandidateConfigError(
                "structure='single' cannot include an unused attribution_prompt"
            )
        if not isinstance(self.hypothesis, str):
            raise CandidateConfigError("hypothesis must be a string")
        if len(self.hypothesis) > 10_000:
            raise CandidateConfigError("hypothesis exceeds 10,000 characters")
        validate_evidence_stages(self.evidence_stages)
        expected_candidate_id = f"candidate_{self.execution_hash()[:12]}"
        if self.candidate_id != expected_candidate_id:
            raise CandidateConfigError(
                "candidate_id must be derived from execution_hash: expected "
                f"{expected_candidate_id!r}, got {self.candidate_id!r}"
            )

    @classmethod
    def materialize(
        cls,
        *,
        prompt_name: str,
        prompt_content: str,
        evidence_stages: Sequence[EvidenceStage | Mapping[str, Any]] = (),
        structure: str = "single",
        attribution_prompt_name: str | None = None,
        attribution_prompt_content: str | None = None,
        hypothesis: str = "",
        parent_candidate_id: str | None = None,
        candidate_id: str | None = None,
    ) -> "CandidateConfig":
        """Create a config and derive a stable ID when none is supplied."""
        if (attribution_prompt_name is None) != (
            attribution_prompt_content is None
        ):
            raise CandidateConfigError(
                "attribution_prompt_name and attribution_prompt_content must be "
                "provided together"
            )
        stages = tuple(
            stage
            if isinstance(stage, EvidenceStage)
            else EvidenceStage.from_dict(stage)
            for stage in evidence_stages
        )
        materialized_stages = tuple(
            replace(
                stage,
                params=materialize_tool_params(stage.tool_id, stage.params),
            )
            for stage in stages
        )
        prompt = PromptSnapshot.from_content(
            name=prompt_name, content=prompt_content
        )
        attribution_prompt = (
            PromptSnapshot.from_content(
                name=attribution_prompt_name,
                content=attribution_prompt_content,
            )
            if attribution_prompt_name is not None
            and attribution_prompt_content is not None
            else None
        )
        execution_payload = cls._make_execution_payload(
            schema_version=CANDIDATE_SCHEMA_VERSION,
            prompt=prompt,
            evidence_stages=materialized_stages,
            structure=structure,
            attribution_prompt=attribution_prompt,
        )
        execution_hash = _canonical_hash(execution_payload)
        derived_candidate_id = f"candidate_{execution_hash[:12]}"
        if candidate_id is not None and candidate_id != derived_candidate_id:
            raise CandidateConfigError(
                "candidate_id must be derived from execution_hash: expected "
                f"{derived_candidate_id!r}, got {candidate_id!r}"
            )
        return cls(
            schema_version=CANDIDATE_SCHEMA_VERSION,
            candidate_id=derived_candidate_id,
            parent_candidate_id=parent_candidate_id,
            prompt=prompt,
            evidence_stages=materialized_stages,
            structure=structure,
            attribution_prompt=attribution_prompt,
            hypothesis=hypothesis,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateConfig":
        if not isinstance(data, Mapping):
            raise CandidateConfigError("candidate config must be an object")
        allowed = {
            "schema_version",
            "candidate_id",
            "parent_candidate_id",
            "prompt",
            "prompt_snapshot",  # compatibility alias
            "evidence_stages",
            "structure",
            "attribution_prompt",
            "hypothesis",
            "execution_hash",
            "config_hash",
        }
        unknown = set(data) - allowed
        if unknown:
            raise CandidateConfigError(f"unknown candidate fields: {sorted(unknown)}")
        prompt_data = data.get("prompt", data.get("prompt_snapshot"))
        if prompt_data is None:
            raise CandidateConfigError("candidate config is missing 'prompt'")
        raw_stages = data.get("evidence_stages", [])
        if not isinstance(raw_stages, list):
            raise CandidateConfigError("evidence_stages must be a list")
        attr_data = data.get("attribution_prompt")
        prompt_snapshot = PromptSnapshot.from_dict(prompt_data)
        attr_snapshot = (
            PromptSnapshot.from_dict(attr_data) if attr_data is not None else None
        )
        schema_version = data.get("schema_version", CANDIDATE_SCHEMA_VERSION)
        if (
            isinstance(schema_version, bool)
            or schema_version != CANDIDATE_SCHEMA_VERSION
        ):
            raise CandidateConfigError(
                f"unsupported candidate schema {schema_version!r}; "
                f"expected {CANDIDATE_SCHEMA_VERSION}"
            )
        candidate = cls.materialize(
            candidate_id=data.get("candidate_id"),
            parent_candidate_id=data.get("parent_candidate_id"),
            prompt_name=prompt_snapshot.name,
            prompt_content=prompt_snapshot.content,
            evidence_stages=tuple(
                EvidenceStage.from_dict(stage) for stage in raw_stages
            ),
            structure=data.get("structure", "single"),
            attribution_prompt_name=(attr_snapshot.name if attr_snapshot else None),
            attribution_prompt_content=(
                attr_snapshot.content if attr_snapshot else None
            ),
            hypothesis=data.get("hypothesis", ""),
        )
        supplied_execution_hash = data.get("execution_hash")
        if (
            supplied_execution_hash is not None
            and supplied_execution_hash != candidate.execution_hash()
        ):
            raise CandidateConfigError(
                "candidate execution_hash mismatch: executable content was modified"
            )
        supplied_config_hash = data.get("config_hash")
        if (
            supplied_config_hash is not None
            and supplied_config_hash != candidate.config_hash()
        ):
            raise CandidateConfigError(
                "candidate config_hash mismatch: snapshot content was modified"
            )
        return candidate

    @staticmethod
    def _make_execution_payload(
        *,
        schema_version: int,
        prompt: PromptSnapshot,
        evidence_stages: Sequence[EvidenceStage],
        structure: str,
        attribution_prompt: PromptSnapshot | None,
    ) -> dict[str, Any]:
        """Canonical content that can change what the evaluated model sees."""
        return {
            "schema_version": schema_version,
            "prompt": {"content": prompt.content, "sha256": prompt.sha256},
            "evidence_stages": [stage.to_dict() for stage in evidence_stages],
            "structure": structure,
            "attribution_prompt": (
                {
                    "content": attribution_prompt.content,
                    "sha256": attribution_prompt.sha256,
                }
                if attribution_prompt is not None
                else None
            ),
        }

    def _execution_payload(self) -> dict[str, Any]:
        return self._make_execution_payload(
            schema_version=self.schema_version,
            prompt=self.prompt,
            evidence_stages=self.evidence_stages,
            structure=self.structure,
            attribution_prompt=self.attribution_prompt,
        )

    def _record_payload(self) -> dict[str, Any]:
        """Executable identity plus research provenance and rationale."""
        return {
            "execution_hash": self.execution_hash(),
            "prompt_name": self.prompt.name,
            "attribution_prompt_name": (
                self.attribution_prompt.name
                if self.attribution_prompt is not None
                else None
            ),
            "parent_candidate_id": self.parent_candidate_id,
            "hypothesis": self.hypothesis,
        }

    def execution_hash(self) -> str:
        """SHA-256 over only the exact executable candidate configuration."""
        return _canonical_hash(self._execution_payload())

    def config_hash(self) -> str:
        """SHA-256 over executable content plus hypothesis and lineage."""
        return _canonical_hash(self._record_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "prompt": self.prompt.to_dict(),
            "evidence_stages": [stage.to_dict() for stage in self.evidence_stages],
            "structure": self.structure,
            "attribution_prompt": (
                self.attribution_prompt.to_dict()
                if self.attribution_prompt is not None
                else None
            ),
            "hypothesis": self.hypothesis,
        }

    def snapshot_dict(self) -> dict[str, Any]:
        """Serialisable run record including both independently checked hashes."""
        return {
            **self.to_dict(),
            "execution_hash": self.execution_hash(),
            "config_hash": self.config_hash(),
        }
