#!/usr/bin/env python3
"""Own labels for the frozen tau2 telecom suite, written from policies_final.md only.

Completed call: the tool message that follows the agent call by position is
neither flagged error nor starts with "Error". Agent calls only.
Also checks rendered-input hashes and writer output formats. No model calls.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from config import DATA_ROOT, OUT_ROOT, REPO_ROOT

HERE = Path(__file__).resolve().parent
EVAL = REPO_ROOT / "ledgers"
from ledgers.shade_run import _trajectory_text

EXP = DATA_ROOT / "tau2"
T2 = EXP / "tau2_telecom"
TRAJECTORIES = T2 / "trajectories"
OUT = OUT_ROOT / "labels" / "tau2"
POLICY_SHA = "7a66397e92311b89307ff79d8cd4b1b8006c94b22a3cba6a1877b40bb97db1bf"
TOOLS = ("refuel_data", "enable_roaming", "resume_line", "send_payment_request", "transfer_to_human_agents", "disable_roaming", "suspend_line")


def completed(result: dict | None) -> bool:
    if result is None:
        return False
    if result.get("error") is True:
        return False
    return not str(result.get("content") or "").lstrip().startswith("Error")


def agent_events(conv: list[dict]) -> tuple[list[dict], list[str]]:
    events, problems = [], []
    for i, turn in enumerate(conv):
        if turn.get("role") != "assistant" or not turn.get("calls"):
            continue
        calls = turn["calls"]
        if any(c.get("actor") not in (None, "agent") for c in calls):
            problems.append(f"turn {i}: non-agent actor in assistant calls")
        results = []
        j = i + 1
        while j < len(conv) and conv[j].get("role") == "tool":
            results.append(conv[j])
            j += 1
        if len(results) != len(calls):
            problems.append(f"turn {i}: {len(calls)} calls, {len(results)} following tool messages")
        for k, call in enumerate(calls):
            result = results[k] if k < len(results) else None
            if result is not None and result.get("tool_name") not in (None, call.get("name")):
                problems.append(f"turn {i} call {k}: name {call.get('name')} vs result tool_name {result.get('tool_name')}")
            events.append({"turn": i, "name": call.get("name"), "arguments": call.get("arguments"), "completed": completed(result),
                           "result_head": None if result is None else str(result.get("content"))[:160], "error_flag": None if result is None else result.get("error")})
    return events, problems


def main() -> None:
    OUT.mkdir(exist_ok=True)
    if (OUT / "labels_report.json").exists():
        raise SystemExit("refusing to overwrite labels_report.json")
    actual = hashlib.sha256((EXP / "policies_final.md").read_bytes()).hexdigest()
    if actual != POLICY_SHA:
        raise SystemExit("policies_final.md hash changed: " + actual)
    manifest = json.loads((T2 / "manifest.json").read_text())
    data = {row["path"]: row["labels"] for row in json.loads((T2 / "atomic/labels_atomic.json").read_text())}
    rendered = json.loads((T2 / "rendered_inputs.json").read_text())
    shade_sha = hashlib.sha256((EVAL / "shade_run.py").read_bytes()).hexdigest()

    mine, evidence, problems = {}, {}, {}
    calls_total, calls_done, user_names, user_write_names = Counter(), Counter(), Counter(), Counter()
    render_mismatch, meta = [], {}
    for row in manifest:
        path = row["path"]
        doc = json.loads((TRAJECTORIES / path).read_text())
        conv = doc.get("conversation") or []
        events, probs = agent_events(conv)
        if probs:
            problems[path] = probs
        labels = {}
        for tool in TOOLS:
            hits = [e for e in events if e["name"] == tool]
            calls_total[tool] += len(hits)
            calls_done[tool] += sum(e["completed"] for e in hits)
            labels["completed_" + tool] = any(e["completed"] for e in hits)
        mine[path] = labels
        evidence[path] = [e for e in events if e["name"] in TOOLS]
        meta[path] = {"task": doc.get("task_id"), "agent_model": doc.get("agent_model"), "category": doc.get("category")}
        for turn in conv:
            if turn.get("role") == "user" and turn.get("calls"):
                for call in turn["calls"]:
                    user_names[call.get("name")] += 1
                    if call.get("name") in TOOLS:
                        user_write_names[call.get("name")] += 1
        text = _trajectory_text({"path": path}, TRAJECTORIES)
        want = rendered["inputs"].get(path) or {}
        got = hashlib.sha256(text.encode()).hexdigest()
        if got != want.get("sha256") or len(text) != want.get("chars"):
            render_mismatch.append({"path": path, "want": want, "got": got, "chars": len(text)})

    disagreements = []
    for path, labels in mine.items():
        for pid, value in labels.items():
            if data[path].get(pid) != value:
                disagreements.append({"path": path, "policy": pid, "mine": value, "data": data[path].get(pid), "events": evidence[path]})
    positives = {pid: sum(l[pid] for l in mine.values()) for pid in next(iter(mine.values()))}
    by_model = {pid: dict(Counter(meta[p]["agent_model"] for p, l in mine.items() if l[pid])) for pid in positives}

    writers = {}
    for name, sub in (("a12", EXP / "writers/a12/constructor_records"), ("narrative", EXP / "writers/action_summary_8192/texts")):
        files = sorted(sub.glob("*.json")) if sub.is_dir() else []
        info = {"n": len(files)}
        if files:
            sample = json.loads(files[0].read_text())
            info["keys"] = list(sample.keys())
            info["record_keys"] = list((sample.get("record") or {}).keys())[:20]
            info["path_field"] = sample.get("path") or (sample.get("record") or {}).get("path")
            info["prompt_sha256"] = sample.get("prompt_sha256")
            paths = set()
            for f in files:
                s = json.loads(f.read_text())
                paths.add(s.get("path") or (s.get("record") or {}).get("path"))
            info["paths_in_manifest"] = len(paths & set(mine))
        writers[name] = info
    writers["action_summary_sha_expected"] = (EVAL / "prompts/action_summary.sha256").read_text().strip()

    report = {
        "policies_final_sha256": actual,
        "positives_mine": positives,
        "by_model_mine": by_model,
        "calls_total": dict(calls_total),
        "calls_completed": dict(calls_done),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "pairing_problems": problems,
        "user_action_names": dict(user_names),
        "user_actions_named_like_agent_write_tools": dict(user_write_names),
        "render": {"shade_run_sha_now": shade_sha, "shade_run_sha_frozen": rendered.get("shade_run_sha256"), "n_checked": len(manifest), "n_mismatch": len(render_mismatch), "mismatches": render_mismatch[:5]},
        "writers": writers,
    }
    (OUT / "labels_mine.json").write_text(json.dumps([{"path": p, "labels": l, **meta[p]} for p, l in mine.items()], indent=1) + "\n")
    (OUT / "labels_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("disagreements",)}, indent=1))


if __name__ == "__main__":
    main()
