"""Build the SFT train/val files from dataset B.

Clean subset (no inline label mentions, not truncated), the four f205f205
candidates only. System = gemma_factual_ledger_baseline.txt (replaces the Haiku
instructions and drops the tool-output block); user = the trajectory exactly as
Haiku saw it; assistant = neutral ledger target. Split is unchanged.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
from config import DATA_ROOT, REPO_ROOT

HERE = Path(__file__).resolve().parent
DATA = DATA_ROOT / "sft_data"
PROMPT = REPO_ROOT / "ledgers/prompts/gemma_factual_ledger_baseline.txt"
F205_CANDIDATES = {"1d3f30ee8dc1", "dc72787dc8d3", "4d44c1247322", "ba68d47fcfdf"}


def main() -> None:
    prompt = PROMPT.read_text()
    out_dir = DATA_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"prompt_file": str(PROMPT.relative_to(REPO_ROOT)),
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    for split in ("train", "val"):
        rows = []
        for line in open(DATA / f"dataset_B_{split}.jsonl"):
            ex = json.loads(line)
            meta = ex["meta"]
            if ex["candidate"] not in F205_CANDIDATES:
                continue
            if meta["inline_label_mentions"] != 0 or meta["constructor_truncated"]:
                continue
            assert meta["split"] == split
            assert "heldout" not in ex["trajectory_id"] and "calendar" not in ex["trajectory_id"]
            user = ex["input"]["user"]
            assert user.startswith("### INPUT TRAJECTORY") and ex["evidence"] not in user
            rows.append({
                "id": ex["id"],
                "trajectory_id": ex["trajectory_id"],
                "candidate": ex["candidate"],
                "label": ex["label"],
                "construct": ex["construct"],
                "domain": ex["domain"],
                "bucket": meta["bucket"],
                "system": prompt,
                "user": user,
                "target": ex["target"],
            })
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[split] = {
            "n": len(rows),
            "trajectories": len({r["trajectory_id"] for r in rows}),
            "by_candidate": dict(Counter(r["candidate"] for r in rows)),
            "by_bucket": dict(sorted(Counter(r["bucket"] for r in rows).items())),
            "by_label": dict(Counter(r["label"] for r in rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
