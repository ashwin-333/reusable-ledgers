"""Bounded GPT advice for the offline Gemma research agent.

Only aggregate screen/train results enter this request. The advisor is never
part of a per-trajectory constructor or reader call.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gemma_search_queue import job_results


def train_only_summary(root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for stage in ("screen", "train"):
        rows = []
        for job in job_results(root, stage)[:30]:
            result_path = Path(job["result_dir"]) / "reconstruction_metric.json"
            result = json.loads(result_path.read_text())
            scored = result["scored"]
            rows.append({
                "candidate_id": job["candidate_id"],
                "lds_v2": result["score"],
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
        summary[stage] = rows
    return summary


def _response_text(response: dict[str, Any]) -> str:
    blocks = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                blocks.append(part["text"])
    answer = "\n".join(blocks).strip()
    if not answer:
        raise RuntimeError("GPT advisor returned no text message")
    return answer


def ask_gpt_advisor(
    *, root: Path, model: str, question: str, max_calls: int = 3,
    timeout_seconds: float = 120.0,
) -> str:
    if not model.strip():
        raise ValueError("pin an explicit GPT model ID")
    if not question.strip() or len(question) > 2000:
        raise ValueError("advisor question must be 1–2000 characters")
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for GPT advice")
    log_path = root / "advisor_calls.jsonl"
    attempts_path = root / "advisor_attempts.jsonl"
    existing = attempts_path.read_text().splitlines() if attempts_path.exists() else []
    if len(existing) >= max_calls:
        raise RuntimeError(f"GPT advisor call cap reached ({max_calls})")
    summary = train_only_summary(root)
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    with attempts_path.open("a") as handle:
        handle.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(),
            "model": model, "question": question,
            "train_only_summary": summary,
        }, ensure_ascii=False) + "\n")
    payload = {
        "model": model,
        "store": False,
        "max_output_tokens": 1500,
        "input": [
            {
                "role": "system",
                "content": (
                    "You advise an offline research search for better factual ledgers "
                    "written by frozen Gemma. Propose task-general prompt or pure-compute "
                    "tool hypotheses. Do not generate case ledgers, labels, or test-set guesses. "
                    "Only train-set aggregate metrics are available."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"question": question, "train_only_results": summary}),
            },
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"GPT advisor HTTP {exc.code}: {detail}") from exc
    answer = _response_text(result)
    event = {
        "at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "question": question,
        "train_only_summary": summary,
        "response_id": result.get("id"),
        "usage": result.get("usage"),
        "answer": answer,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-calls", type=int, default=3)
    args = parser.parse_args()
    print(ask_gpt_advisor(
        root=args.root, model=args.model, question=args.question,
        max_calls=args.max_calls,
    ))


if __name__ == "__main__":
    main()
