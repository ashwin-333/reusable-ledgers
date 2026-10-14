"""LLMLingua-2 compression of raw trajectories. Policies are not read.

The target is the median token length of the non-empty trained ledgers,
measured with this compressor's tokenizer, and four times that median.
A trajectory already at or under the target is copied unchanged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from config import DATA_ROOT, RUNS_DIR, STATUS_DIR

HERE = Path(__file__).resolve().parent

from ledgers.e5_real import _shade_ledgers
from ledgers.shade_run import _trajectory_text

SFT = DATA_ROOT
BENCH = SFT / "multi_policy_real" / "multi_policy_benchmark.json"
OUT = RUNS_DIR / "mp_controls/lingua"
STATUS = STATUS_DIR / "mp_controls"
MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


def _count(tokenizer, text: str) -> int:
    return len(tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"])


def _median(values: list[int]) -> int:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return int(ordered[mid])
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))


def main() -> None:
    from llmlingua import PromptCompressor

    payload = json.loads(BENCH.read_text())
    rows = payload["trajectories"]
    data_root = Path(payload.get("data_root") or (SFT / "shade_real"))
    compressor = PromptCompressor(model_name=MODEL, use_llmlingua2=True, device_map="cpu")
    tokenizer = compressor.tokenizer
    ledgers = _shade_ledgers("ledger_C")
    lengths = [_count(tokenizer, text) for text in ledgers.values() if text.strip()]
    if not lengths:
        raise SystemExit("no non-empty trained ledgers to set the token budget")
    median = _median(lengths)
    targets = {"median": median, "x4": 4 * median}
    STATUS.mkdir(parents=True, exist_ok=True)
    budget = {
        "tokenizer": MODEL,
        "n_ledgers": len(lengths),
        "median_ledger_tokens": median,
        "x4_ledger_tokens": 4 * median,
        "source": "sanitized ledger_C text, empty ledgers excluded",
        "policies_seen": False,
    }
    (STATUS / "ledger_token_budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    print(json.dumps(budget), flush=True)
    for name, target in targets.items():
        directory = OUT / name
        directory.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows, 1):
            dest = directory / f"{index:03d}.json"
            if dest.is_file():
                saved = json.loads(dest.read_text())
                if saved.get("path") == row["path"] and saved.get("target_tokens") == target and saved.get("text"):
                    continue
            started = time.monotonic()
            source = _trajectory_text(row, data_root)
            origin = _count(tokenizer, source)
            if origin <= target:
                text = source
                compressed = origin
                rate = 1.0
            else:
                rate = target / origin
                result = compressor.compress_prompt(source, rate=rate)
                text = result["compressed_prompt"]
                compressed = int(result.get("compressed_tokens") or _count(tokenizer, text))
            dest.write_text(json.dumps({
                "path": row["path"],
                "text": text,
                "seconds": time.monotonic() - started,
                "origin_tokens": origin,
                "compressed_tokens": compressed,
                "target_tokens": target,
                "rate": rate,
                "policies_seen": False,
            }) + "\n")
            if index % 25 == 0:
                print(name, index, flush=True)
    print("compress done", flush=True)


if __name__ == "__main__":
    main()
