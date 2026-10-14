"""Reader training rows: sanitized C and D teacher ledgers, verdict JSON targets.

Training uses dataset C and dataset D rows only. The ledger val split is the
selection set and is not written into the training file. Held-out eval
manifests are not read.
"""
import json
from pathlib import Path

from autoresearch.reconstruction_metric import (
    GEMMA_DECODER_SYSTEM_PROMPT,
    ReconstructionMetricError,
    sanitize_ledger,
)
from config import DATA_ROOT

S = DATA_ROOT
CONSTRUCTS = ("C1", "C2", "C3", "C4")
OUT = S / "data_reader"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def verdict_target(row):
    label = row.get("label")
    construct = row.get("construct")
    if label == "benign":
        body = {"verdict": "PASS", "violations_found": []}
    elif label == "malicious" and construct in CONSTRUCTS:
        body = {"verdict": "FAIL", "violations_found": [construct]}
    else:
        return None
    return json.dumps(body, separators=(",", ":"))


def convert(row):
    target = verdict_target(row)
    ledger = row.get("target") or ""
    if target is None or not str(ledger).strip():
        return None
    try:
        clean = sanitize_ledger(ledger)
    except ReconstructionMetricError:
        return None
    return {
        "id": row["id"],
        "bucket": row.get("bucket") or "reader",
        "label": row["label"],
        "construct": row.get("construct"),
        "domain": row.get("domain"),
        "system": GEMMA_DECODER_SYSTEM_PROMPT,
        "user": "[NEUTRAL_LEDGER]\n" + clean,
        "target": target,
    }


def main() -> None:
    keep = json.loads((S / "ledger_keep_ids.json").read_text())
    base_ids = set(keep["train"]["ids"])
    val_ids = set(keep["val"]["ids"])
    val_traj = {i.split("::", 1)[1] for i in val_ids}

    c_rows = load_jsonl(S / "data_c/train.jsonl")
    d_rows = load_jsonl(S / "data_d/train.jsonl")
    c_ids = {r["id"] for r in c_rows}
    # C training rows are the data_c train file minus the original ledger train ids.
    # D training rows are the data_d train file minus every data_c id.
    sources = [("C", [r for r in c_rows if r["id"] not in base_ids]),
               ("D", [r for r in d_rows if r["id"] not in c_ids])]
    train = []
    dropped = {"bad_label": 0, "sanitize": 0, "val_id": 0, "val_trajectory": 0}
    for name, rows in sources:
        for row in rows:
            tid = row.get("trajectory_id") or row["id"].split("::", 1)[-1]
            if row["id"] in val_ids:
                dropped["val_id"] += 1
                continue
            if tid in val_traj:
                dropped["val_trajectory"] += 1
                continue
            made = convert(row)
            if made is None:
                if verdict_target(row) is None:
                    dropped["bad_label"] += 1
                else:
                    dropped["sanitize"] += 1
                continue
            made["source"] = name
            train.append(made)

    val_src = load_jsonl(S / "data_c/val.jsonl")
    val = []
    for row in val_src:
        made = convert(row)
        if made is not None:
            val.append(made)
    if len(val) != 20:
        raise SystemExit(f"reader val has {len(val)} rows, expected 20")
    train_ids = {r["id"] for r in train}
    if train_ids & {r["id"] for r in val}:
        raise SystemExit("a val id is in the reader training set")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = OUT / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_source = {s: sum(r["source"] == s for r in train) for s in ("C", "D")}
    summary = {
        "n_train": len(train),
        "n_val": len(val),
        "by_source": by_source,
        "dropped": dropped,
        "target": '{"verdict":"PASS"|"FAIL","violations_found":["C1","C2","C3","C4"]}',
        "user": "[NEUTRAL_LEDGER]\\n + sanitized teacher ledger",
        "val_use": "selection only",
    }
    (S / "data_reader" / "12b_reader_data.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
