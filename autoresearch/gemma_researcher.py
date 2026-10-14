"""Opus-led, batched parallel search over frozen-Gemma constructor candidates.

This controller never loads a held-out manifest or calls a constructor itself.
It proposes immutable candidates, queues train-only work, and consults GPT
through the separate bounded advisor when Opus requests it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .candidate_config import CandidateConfig, CandidateConfigError, load_tool_catalog
from .gemma_advisor import ask_gpt_advisor
from .gemma_baseline import _write_json
from .gemma_candidate import PURE_COMPUTE_TOOL_IDS
from .gemma_search_queue import _load_config, enqueue, job_results, store_candidate, write_report


SYSTEM_PROMPT = Path(__file__).parent / "prompts/gemma_researcher_system.txt"

RESEARCH_TOOLS = [
    {
        "name": "submit_candidates",
        "description": "Submit several independent Gemma-constructor experiments for parallel screening.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hypothesis": {"type": "string"},
                            "prompt": {"type": "string"},
                            "evidence_stages": {"type": "array", "items": {"type": "object"}},
                            "parent_candidate_id": {"type": "string"},
                            "structure": {"type": "string", "enum": ["single", "two_pass"]},
                            "second_prompt": {"type": "string"},
                        },
                        "required": ["hypothesis", "prompt", "evidence_stages", "structure"],
                    },
                },
                "new_tool_ideas": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["candidates"],
        },
    },
    {
        "name": "ask_gpt_advisor",
        "description": "Ask a frontier GPT for fresh hypotheses using train-only aggregate history.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
]


def _append_event(root: Path, event: dict[str, Any]) -> None:
    path = root / "research_history.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(), **event,
        }, ensure_ascii=False) + "\n")


def _catalog_summary() -> list[dict[str, Any]]:
    catalog = load_tool_catalog()
    rows = []
    for tool in catalog.tools:
        row = tool.to_dict()
        row["runtime"]["gemma_audited"] = tool.tool_id in PURE_COMPUTE_TOOL_IDS
        row["runtime"]["parameter_schema"] = tool.runtime_parameter_schema
        rows.append(row)
    return rows


def _safe_history(root: Path) -> dict[str, Any]:
    history: dict[str, Any] = {}
    for stage in ("screen", "train"):
        rows = []
        for job in job_results(root, stage)[:30]:
            result = json.loads((Path(job["result_dir"]) / "reconstruction_metric.json").read_text())
            scored = result["scored"]
            rows.append({
                "candidate_id": job["candidate_id"],
                "lds_v2": job["score"],
                "balanced_accuracy": scored["discrimination"]["balanced_accuracy"],
                "macro_f1": scored["attribution"]["macro_f1"],
                "by_domain_lds_v2": {
                    domain: value["lds_v2"]
                    for domain, value in scored["by_domain"].items()
                },
                "benign_false_flag_rate": scored["ul_metrics"]["benign_false_flag_rate"],
                "construct_recall": scored["ul_metrics"]["construct_recall"],
                "constructor_errors": result["meta"]["constructor_errors"],
                "unusable_ledgers": result["meta"]["sanitizer_unusable"],
            })
        history[stage] = rows
    return history


def _best_prompt(root: Path) -> tuple[str, str]:
    train = job_results(root, "train")
    screen = job_results(root, "screen")
    seed = json.loads((root / "seed.json").read_text())
    seed_id = seed["candidate_id"]
    if train:
        candidate_id = (
            train[0]["candidate_id"]
            if train[0]["score"] > seed["full_train_lds_v2_from_completed_baseline"]
            else seed_id
        )
    else:
        candidate_id = (screen or [{"candidate_id": seed_id}])[0]["candidate_id"]
    candidate = CandidateConfig.from_dict(json.loads(
        (root / "candidates" / f"{candidate_id}.json").read_text()
    ))
    return candidate_id, candidate.prompt.content


def _proposal_context(root: Path, batch_size: int, stalled_rounds: int) -> str:
    best_id, best_prompt = _best_prompt(root)
    seed = json.loads((root / "seed.json").read_text())
    history_path = root / "research_history.jsonl"
    recent_rejections = []
    if history_path.exists():
        for line in history_path.read_text().splitlines()[-40:]:
            event = json.loads(line)
            if event.get("event") == "candidate_rejected":
                recent_rejections.append(event["reason"])
    return json.dumps({
        "instruction": (
            f"Propose {batch_size} genuinely different candidate configurations. "
            "Each prompt is a complete Gemma system prompt, not a diff. "
            "Use the full tool catalog as inspiration but select only runnable, "
            "Gemma-audited "
            "stages. A graph stage needs stage_id, tool_id, input_from, params, "
            "and max_output_chars. For two_pass, provide a factual second_prompt. "
            "Do not use labels, task-specific identifiers, Haiku, or GPT in "
            "case-time construction."
        ),
        "best_candidate_id": best_id,
        "best_candidate_prompt": best_prompt,
        "no_tools_seed_full_train_lds_v2": seed["full_train_lds_v2_from_completed_baseline"],
        "train_only_scores": _safe_history(root),
        "rounds_without_full_train_improvement": stalled_rounds,
        "recent_invalid_candidate_reasons": recent_rejections[-8:],
        "full_tool_catalog": _catalog_summary(),
    }, ensure_ascii=False)


def _call_opus(
    *, client: Any, model: str, root: Path, batch_size: int,
    stalled_rounds: int, gpt_model: str | None, advisor_cap: int,
    remaining_calls: int,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{
        "role": "user", "content": _proposal_context(root, batch_size, stalled_rounds),
    }]
    available_tools = RESEARCH_TOOLS if stalled_rounds >= 2 else RESEARCH_TOOLS[:1]
    for _ in range(5 if remaining_calls == 0 else min(5, remaining_calls)):
        context_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        _append_event(root, {"event": "opus_attempt", "model": model, "context_sha256": context_hash})
        try:
            response = client.messages.create(
                model=model,
                max_tokens=12000,
                temperature=0.4,
                system=SYSTEM_PROMPT.read_text(),
                tools=available_tools,
                messages=messages,
            )
        except Exception as exc:
            _append_event(root, {"event": "opus_error", "model": model, "error": f"{type(exc).__name__}: {exc}"})
            raise
        _append_event(root, {
            "event": "opus_call", "model": model,
            "usage": response.usage.model_dump() if hasattr(response.usage, "model_dump") else {},
        })
        tool_blocks = [block for block in response.content if block.type == "tool_use"]
        for block in tool_blocks:
            if block.name == "submit_candidates":
                return block.input
        if not tool_blocks:
            raise RuntimeError("Opus returned no candidate submission or advisor call")
        messages.append({
            "role": "assistant",
            "content": [block.model_dump() for block in response.content],
        })
        results = []
        for block in tool_blocks:
            if block.name != "ask_gpt_advisor":
                raise RuntimeError(f"unexpected Opus tool call: {block.name}")
            if gpt_model is None:
                answer = "GPT advisor is not configured; propose candidates directly."
            else:
                try:
                    answer = ask_gpt_advisor(
                        root=root, model=gpt_model, question=block.input["question"],
                        max_calls=advisor_cap,
                    )
                except Exception as exc:
                    answer = f"Advisor unavailable: {type(exc).__name__}: {exc}"
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": answer})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("Opus did not submit candidates within five tool turns")


def _materialize_proposals(
    root: Path, proposals: dict[str, Any], batch_size: int,
) -> list[str]:
    raw = proposals.get("candidates")
    if not isinstance(raw, list) or not 1 <= len(raw) <= batch_size:
        raise ValueError(f"Opus must submit 1–{batch_size} candidates")
    ids: list[str] = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {index} must be an object")
        try:
            structure = item.get("structure", "single")
            second_prompt = item.get("second_prompt") if structure == "two_pass" else None
            candidate = CandidateConfig.materialize(
                prompt_name=f"opus_candidate_{int(time.time())}_{index}",
                prompt_content=item["prompt"],
                evidence_stages=item.get("evidence_stages", []),
                structure=structure,
                attribution_prompt_name=("factual_second_pass" if second_prompt else None),
                attribution_prompt_content=second_prompt,
                hypothesis=item["hypothesis"],
                parent_candidate_id=item.get("parent_candidate_id") or None,
            )
            if len(candidate.prompt.content) > 16000:
                raise ValueError("constructor prompt exceeds 16,000 characters")
            if candidate.parent_candidate_id is not None and not (
                root / "candidates" / f"{candidate.parent_candidate_id}.json"
            ).exists():
                raise ValueError("parent_candidate_id is not a stored candidate")
            store_candidate(root, candidate)
            enqueue(root, candidate.candidate_id, "screen")
            ids.append(candidate.candidate_id)
        except (CandidateConfigError, KeyError, TypeError, ValueError) as exc:
            _append_event(root, {
                "event": "candidate_rejected", "index": index,
                "reason": f"{type(exc).__name__}: {exc}",
            })
    for idea in proposals.get("new_tool_ideas", []):
        if isinstance(idea, str) and idea.strip():
            _append_event(root, {"event": "new_tool_idea", "idea": idea[:2000]})
    return list(dict.fromkeys(ids))


def _wait_for_jobs(root: Path, ids: list[str], stage: str, deadline: float, poll_seconds: float) -> None:
    pending = {f"{candidate_id}_{stage}" for candidate_id in ids}
    while pending and time.time() < deadline:
        for job_id in list(pending):
            if (root / "jobs" / "done" / f"{job_id}.json").exists():
                pending.remove(job_id)
            elif (root / "jobs" / "failed" / f"{job_id}.json").exists():
                pending.remove(job_id)
        if pending:
            time.sleep(poll_seconds)


def run_research(
    *, root: Path, researcher_model: str, gpt_model: str | None,
    batch_size: int, promote_count: int, max_opus_calls: int,
    advisor_cap: int, poll_seconds: float,
) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for the Opus researcher")
    if batch_size < 1 or promote_count < 1 or promote_count > batch_size:
        raise ValueError("invalid batch_size/promote_count")
    import anthropic

    client = anthropic.Anthropic()
    config = _load_config(root)
    started_path = root / "started_at.json"
    waiting_started = time.monotonic()
    while not started_path.exists():
        if time.monotonic() - waiting_started > 600:
            raise RuntimeError("no Gemma worker started the search within ten minutes")
        time.sleep(poll_seconds)
    deadline = json.loads(started_path.read_text())["started_at_unix"] + config["wall_hours"] * 3600
    seed_id = json.loads((root / "seed.json").read_text())["candidate_id"]
    _wait_for_jobs(root, [seed_id], "screen", deadline, poll_seconds)
    baseline = json.loads((root / "seed.json").read_text())[
        "full_train_lds_v2_from_completed_baseline"
    ]
    best_score = max([baseline] + [row["score"] for row in job_results(root, "train")])
    stalled_rounds = 0
    history_path = root / "research_history.jsonl"

    def counted_calls() -> int:
        if not history_path.exists():
            return 0
        return sum(
            json.loads(line).get("event") == "opus_attempt"
            for line in history_path.read_text().splitlines()
        )

    if max_opus_calls < 0:
        raise ValueError("max_opus_calls must be nonnegative; zero means no call cap")
    calls = counted_calls()
    while time.time() < deadline and (max_opus_calls == 0 or calls < max_opus_calls):
        proposals = _call_opus(
            client=client, model=researcher_model, root=root,
            batch_size=batch_size, stalled_rounds=stalled_rounds,
            gpt_model=gpt_model, advisor_cap=advisor_cap,
            remaining_calls=0 if max_opus_calls == 0 else max_opus_calls - calls,
        )
        calls = counted_calls()
        ids = _materialize_proposals(root, proposals, batch_size)
        _append_event(root, {"event": "batch_submitted", "candidate_ids": ids})
        if not ids:
            stalled_rounds += 1
            continue
        _wait_for_jobs(root, ids, "screen", deadline, poll_seconds)
        successful = [row for row in job_results(root, "screen") if row["candidate_id"] in ids]
        promoted = [row["candidate_id"] for row in successful[:promote_count]]
        for candidate_id in promoted:
            enqueue(root, candidate_id, "train")
        _wait_for_jobs(root, promoted, "train", deadline, poll_seconds)
        train = job_results(root, "train")
        current_best = train[0]["score"] if train else best_score
        if current_best > best_score:
            best_score = current_best
            stalled_rounds = 0
        else:
            stalled_rounds += 1
        _append_event(root, {
            "event": "batch_finished", "promoted": promoted,
            "best_train_lds_v2": best_score,
            "rounds_without_improvement": stalled_rounds,
        })
        if best_score >= 1.0:
            break
    reason = (
        "target" if best_score >= 1.0 else
        "wall_clock" if time.time() >= deadline else
        "opus_call_cap"
    )
    _write_json(root / "research_done.json", {
        "reason": reason, "best_train_lds_v2": best_score,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    train_results = job_results(root, "train")
    best_candidate = train_results[0] if train_results else None
    seed_wins = best_candidate is None or best_candidate["score"] <= baseline
    _write_json(root / "winner.json", {
        "candidate_id": seed_id if seed_wins else best_candidate["candidate_id"],
        "train_lds_v2": baseline if seed_wins else best_candidate["score"],
        "is_seed": seed_wins,
    })
    write_report(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--researcher-model", required=True)
    parser.add_argument("--gpt-model")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--promote-count", type=int, default=2)
    parser.add_argument("--max-opus-calls", type=int, default=0,
                        help="optional API budget cap; 0 continues until wall-clock/score target")
    parser.add_argument("--advisor-cap", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    run_research(
        root=args.root, researcher_model=args.researcher_model,
        gpt_model=args.gpt_model, batch_size=args.batch_size,
        promote_count=args.promote_count, max_opus_calls=args.max_opus_calls,
        advisor_cap=args.advisor_cap, poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
