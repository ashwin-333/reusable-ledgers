"""Action summaries with the ledger writer's 8192-token generation limit.

Same untrained 12B writer, trajectory input, and frozen action-summary
prompt. Only the generation ceiling changes, from 1600 to 8192. Texts go
to a new directory. The prompt hash is checked before any generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from config import DATA_ROOT, OPENAI_BASE_URL, RUNS_DIR, STATUS_DIR, SUBSET_FILE, WRITER_MODEL

HERE = Path(__file__).resolve().parent

PROMPT_PATH = HERE / "prompts" / "action_summary.txt"
PROMPT_SHA_PATH = HERE / "prompts" / "action_summary.sha256"
CAP_PATH = STATUS_DIR / "mp_controls/action_summary_8192_cap.json"
BENCH = DATA_ROOT / "multi_policy_real/multi_policy_benchmark.json"
SUBSET = SUBSET_FILE
OUT = RUNS_DIR / "mp_controls/action_summary_8192/texts"
MODEL = WRITER_MODEL
TARGET_TOKENS = 1100
MAX_TOKENS = 8192


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_prompt() -> tuple[str, str]:
    digest = _sha(PROMPT_PATH)
    frozen = PROMPT_SHA_PATH.read_text().strip()
    if digest != frozen:
        raise SystemExit("action-summary prompt does not match the frozen hash; refusing to run")
    return PROMPT_PATH.read_text(), digest


def prepare() -> None:
    prompt, digest = _frozen_prompt()
    if not prompt.strip():
        raise SystemExit("action-summary prompt is empty")
    payload = {
        "prompt_sha256": digest,
        "prompt_path": str(PROMPT_PATH),
        "model": MODEL,
        "target_tokens": TARGET_TOKENS,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 0,
        "policies_seen": False,
        "generation_limit": "ledger writer constructor_max_tokens",
    }
    CAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAP_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


def _cap(digest: str) -> int:
    if not CAP_PATH.is_file():
        raise SystemExit("action-summary cap was not frozen before generation")
    payload = json.loads(CAP_PATH.read_text())
    if payload.get("prompt_sha256") != digest:
        raise SystemExit("action-summary cap was frozen for a different prompt")
    cap = int(payload["max_tokens"])
    if cap != MAX_TOKENS:
        raise SystemExit("action-summary cap does not match the frozen ceiling")
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
        raise RuntimeError("action-summary writer answered as " + repr(body.get("model")))
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    return {
        "text": choice["message"]["content"] or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def _selected(rows: list, scope: str) -> list:
    """Same benchmark index the subset writer uses. `rest` is the other 200."""
    keep = set(json.loads(SUBSET.read_text())["paths"])
    if scope == "subset":
        selected = [(index, row) for index, row in enumerate(rows, 1) if row["path"] in keep]
        if len(selected) != len(keep):
            raise SystemExit(f"subset has {len(keep)} paths but {len(selected)} match the benchmark")
        return selected
    if scope == "rest":
        selected = [(index, row) for index, row in enumerate(rows, 1) if row["path"] not in keep]
        if len(selected) != 200:
            raise SystemExit(f"expected 200 remaining trajectories, found {len(selected)}")
        return selected
    raise SystemExit("scope must be subset or rest")


def write(args: argparse.Namespace) -> None:
    prompt, digest = _frozen_prompt()
    cap = _cap(digest)
    payload = json.loads(BENCH.read_text())
    rows = payload["trajectories"]
    data_root = Path(payload.get("data_root") or str(DATA_ROOT / "shade_real"))
    selected = _selected(rows, args.scope)
    mine = [item for n, item in enumerate(selected) if n % args.num_chunks == args.chunk]
    OUT.mkdir(parents=True, exist_ok=True)
    for done, (index, row) in enumerate(mine, 1):
        dest = OUT / f"{index:03d}.json"
        if dest.is_file():
            saved = json.loads(dest.read_text())
            if saved.get("path") not in (None, row["path"]):
                raise SystemExit(f"refusing to overwrite {dest}")
            if (
                saved.get("path") == row["path"]
                and saved.get("prompt_sha256") == digest
                and int(saved.get("max_tokens") or 0) == cap
                and (saved.get("text") or "").strip()
            ):
                print(f"[{done}/{len(mine)}] #{index:03d} resume", flush=True)
                continue
        started = time.monotonic()
        from ledgers.shade_run import _trajectory_text
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
            "target_tokens": TARGET_TOKENS,
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
    parser.add_argument("--scope", choices=("subset", "rest"), default="subset")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    if args.role == "prepare":
        prepare()
        return
    if not args.base_url:
        raise SystemExit("missing --base-url")
    write(args)


if __name__ == "__main__":
    main()
