"""Write ordinary factual summaries with the untrained 12B model.

The prompt file is hashed and checked before any generation. The token cap
is the median Gemma-token length of the non-empty fine-tuned ledgers, written
to the cap file before the first summary call. A later chunk refuses to
generate if that file is missing or the prompt hash does not match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from config import DATA_ROOT, OPENAI_BASE_URL, RUNS_DIR, STATUS_DIR, SUBSET_FILE, WRITER_MODEL, snapshot_dir

HERE = Path(__file__).resolve().parent

from ledgers.e5_real import _shade_ledgers
from ledgers.shade_run import _trajectory_text

PROMPT_PATH = HERE / "prompts" / "summary_factual.txt"
PROMPT_SHA_PATH = HERE / "prompts" / "summary_factual.sha256"
CAP_PATH = STATUS_DIR / "mp_controls/summary_cap.json"
BENCH = DATA_ROOT / "multi_policy_real/multi_policy_benchmark.json"
SUBSET = SUBSET_FILE
OUT = RUNS_DIR / "mp_controls/summary/texts"
MODEL = WRITER_MODEL
LINGUA_MEDIAN = 1342


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_prompt() -> tuple[str, str]:
    digest = _sha(PROMPT_PATH)
    frozen = PROMPT_SHA_PATH.read_text().strip()
    if digest != frozen:
        raise SystemExit("summary prompt does not match the frozen hash; refusing to run")
    return PROMPT_PATH.read_text(), digest


def _median(values: list[int]) -> int:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return int(ordered[mid])
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))


def _snapshot() -> Path:
    return snapshot_dir(MODEL)


def prepare() -> None:
    prompt, digest = _frozen_prompt()
    if not prompt.strip():
        raise SystemExit("summary prompt is empty")
    snap = _snapshot()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    ledgers = _shade_ledgers("ledger_C")
    lengths = []
    for text in ledgers.values():
        if not text.strip():
            continue
        lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
    if not lengths:
        raise SystemExit("no non-empty fine-tuned ledgers")
    cap = _median(lengths)
    if cap < 1:
        raise SystemExit("median ledger length is not positive")
    payload = {
        "prompt_sha256": digest,
        "prompt_path": str(PROMPT_PATH),
        "tokenizer": MODEL,
        "tokenizer_snapshot": snap.name,
        "add_special_tokens": False,
        "n_ledgers": len(lengths),
        "median_ledger_tokens": cap,
        "source": "non-empty sanitized shade ledger_C text",
        "even_median": "mean of the two central lengths, then round",
        "lingua_median_tokens": LINGUA_MEDIAN,
        "lingua_note": "LLMLingua tokenizer on the same ledgers; not the generation cap",
        "policies_seen": False,
    }
    CAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAP_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


def _cap(digest: str) -> int:
    if not CAP_PATH.is_file():
        raise SystemExit("summary cap was not frozen before generation")
    payload = json.loads(CAP_PATH.read_text())
    if payload.get("prompt_sha256") != digest:
        raise SystemExit("summary cap was frozen for a different prompt")
    cap = int(payload["median_ledger_tokens"])
    if cap < 1:
        raise SystemExit("summary cap is not positive")
    return cap


def _post(base_url: str, model: str, system: str, user: str, max_tokens: int, timeout: float) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 0,
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    if body.get("model") != model:
        raise RuntimeError("summary writer answered as " + repr(body.get("model")))
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    return {
        "text": choice["message"]["content"] or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def write(args: argparse.Namespace) -> None:
    prompt, digest = _frozen_prompt()
    cap = _cap(digest)
    payload = json.loads(BENCH.read_text())
    rows = payload["trajectories"]
    data_root = Path(payload.get("data_root") or str(DATA_ROOT / "shade_real"))
    keep = set(json.loads(SUBSET.read_text())["paths"])
    selected = [(index, row) for index, row in enumerate(rows, 1) if row["path"] in keep]
    if len(selected) != len(keep):
        raise SystemExit(f"subset has {len(keep)} paths but {len(selected)} match the benchmark")
    mine = [item for n, item in enumerate(selected) if n % args.num_chunks == args.chunk]
    OUT.mkdir(parents=True, exist_ok=True)
    for done, (index, row) in enumerate(mine, 1):
        dest = OUT / f"{index:03d}.json"
        if dest.is_file():
            saved = json.loads(dest.read_text())
            if (
                saved.get("path") == row["path"]
                and saved.get("prompt_sha256") == digest
                and int(saved.get("max_tokens") or 0) == cap
                and (saved.get("text") or "").strip()
            ):
                print(f"[{done}/{len(mine)}] #{index:03d} resume", flush=True)
                continue
        started = time.monotonic()
        user = _trajectory_text(row, data_root)
        result = None
        last = None
        for attempt in range(3):
            try:
                result = _post(args.base_url, args.served_model, prompt, user, cap, args.timeout_seconds)
                last = None
                break
            except (OSError, urllib.error.URLError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                last = type(exc).__name__ + ": " + str(exc)
                time.sleep(2 ** attempt)
        record = {
            "path": row["path"],
            "prompt_sha256": digest,
            "max_tokens": cap,
            "model": MODEL,
            "seconds": round(time.monotonic() - started, 3),
        }
        if result is None:
            record["text"] = ""
            record["error"] = last
        else:
            record.update(result)
        dest.write_text(json.dumps(record) + "\n")
        print(f"[{done}/{len(mine)}] #{index:03d} ok", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("prepare", "write"), required=True)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL)
    parser.add_argument("--served-model", default=MODEL)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.role == "prepare":
        prepare()
        return
    if not args.base_url:
        raise SystemExit("missing --base-url")
    write(args)


if __name__ == "__main__":
    main()
