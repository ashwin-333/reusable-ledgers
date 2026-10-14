#!/usr/bin/env python3
"""Build the tau2 telecom atomic and paraphrased benches (new dirs only).

Policy texts are parsed from the hashed policies_final.md; labels are the
agreed own labels (0 disagreements with DATA).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from config import DATA_ROOT, OUT_ROOT, RUNS_DIR

HERE = Path(__file__).resolve().parent
EXP = DATA_ROOT / "tau2"
TRAJECTORIES = EXP / "tau2_telecom/trajectories"
OCT = RUNS_DIR / "mp_controls/oct3"
POLICY_SHA = "7a66397e92311b89307ff79d8cd4b1b8006c94b22a3cba6a1877b40bb97db1bf"
FINAL = ("completed_refuel_data", "completed_enable_roaming", "completed_resume_line", "completed_send_payment_request", "completed_transfer_to_human_agents")


def parse_texts(md: str) -> dict[str, dict[str, str]]:
    out = {}
    for pid in FINAL:
        block = md.split("**`%s`**" % pid, 1)[1].split("**`", 1)[0]
        atomic = re.search(r'- Atomic: "(.*)"\s*$', block, re.M).group(1)
        para = re.search(r'- Paraphrase: "(.*)"\s*$', block, re.M).group(1)
        out[pid] = {"atomic": atomic, "nl": para}
    return out


def write_bench(name: str, policies: list[dict], rows: list[dict]) -> dict:
    dest = OCT / "benches" / name
    if dest.exists():
        raise SystemExit("refusing existing bench " + str(dest))
    dest.mkdir(parents=True)
    (dest / "policies.json").write_text(json.dumps({"policies": policies}, indent=2) + "\n")
    sha = hashlib.sha256((dest / "policies.json").read_bytes()).hexdigest()
    (dest / "multi_policy_benchmark.json").write_text(json.dumps({"policy_sha256": sha, "data_root": str(TRAJECTORIES), "trajectories": rows}) + "\n")
    (dest / "policy.sha256").write_text(sha + "\n")
    return {"bench": str(dest), "hash": sha, "n": len(rows), "policies": policies}


def main() -> None:
    md_bytes = (EXP / "policies_final.md").read_bytes()
    if hashlib.sha256(md_bytes).hexdigest() != POLICY_SHA:
        raise SystemExit("policies_final.md hash changed")
    texts = parse_texts(md_bytes.decode())
    report = json.loads((OUT_ROOT / "labels/tau2/labels_report.json").read_text())
    if report["n_disagreements"] != 0 or report["render"]["n_mismatch"] != 0:
        raise SystemExit("labels or render not clean")
    mine = json.loads((OUT_ROOT / "labels/tau2/labels_mine.json").read_text())
    manifest = [row["path"] for row in json.loads((EXP / "tau2_telecom/manifest.json").read_text())]
    by_path = {row["path"]: row for row in mine}
    rows = [{"path": p, "task": str(by_path[p]["task"]), "domain": "telecom", "labels": {pid: by_path[p]["labels"][pid] for pid in FINAL}} for p in manifest]
    out = {}
    for suite, key in (("tau2_atomic", "atomic"), ("tau2_nl", "nl")):
        out[suite] = write_bench(suite, [{"id": pid, "text": texts[pid][key]} for pid in FINAL], rows)
    out["positives"] = {pid: sum(r["labels"][pid] for r in rows) for pid in FINAL}
    (OUT_ROOT / "labels/tau2/benches.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
