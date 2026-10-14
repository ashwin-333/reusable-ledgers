"""LoRA ledger writer on Qwen3.5-9B.

Separate from train_sft.py so a Gemma preemption never loads this file.
Same data, targets, and training loop as 12b_ledger. The chat template is
Qwen's, with thinking closed, because the Gemma template is not this tokenizer.
"""
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from training import train_sft
from config import DATA_ROOT, OUT_ROOT, QWEN_READER_MODEL

MODEL_ID = QWEN_READER_MODEL
# Native context from the model card config. YaRN is not applied: 64k is
# inside this window, and a static YaRN factor would change shorter positions.
NATIVE_CTX = 262144
CE_CHUNK = 1024
LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))"
)
log = train_sft.log


def _finite(t):
    """Replace NaN and ±inf with 0. Finite values are unchanged."""
    if torch.is_tensor(t) and t.is_floating_point():
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def _guard_kernel(orig):
    def guarded(query, key, value, g, beta, *args, **kwargs):
        query, key, value = _finite(query), _finite(key), _finite(value)
        g, beta = _finite(g), _finite(beta)
        out = orig(query, key, value, g, beta, *args, **kwargs)
        if isinstance(out, tuple):
            return (_finite(out[0]),) + tuple(out[1:])
        return _finite(out)

    guarded._nan_guard = True
    return guarded


def install_nan_guard():
    """Nonfinite replacement on both of Qwen's attention paths.

    Full attention (8 of 32 layers) goes through SDPA. The previous guard
    cleaned q/k/v only, so a NaN produced inside the kernel still reached the
    residual. Linear attention (24 of 32 layers) is GatedDeltaNet. flash-linear-
    attention is not installed, so that path is the torch chunk kernel: l2norm
    of a nonfinite state is NaN, and the SDPA guard never sees those tensors.
    Finite values are unchanged.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_model

    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def guarded(module, query, key, value, attention_mask, *args, **kwargs):
        query, key, value = _finite(query), _finite(key), _finite(value)
        out = orig(module, query, key, value, attention_mask, *args, **kwargs)
        if isinstance(out, tuple):
            return (_finite(out[0]),) + tuple(out[1:])
        return _finite(out)

    # Local override. sdpa lives on the class mapping, which pop() cannot delete.
    ALL_ATTENTION_FUNCTIONS["sdpa"] = guarded
    if not getattr(qwen_model.torch_chunk_gated_delta_rule, "_nan_guard", False):
        qwen_model.torch_chunk_gated_delta_rule = _guard_kernel(qwen_model.torch_chunk_gated_delta_rule)
        qwen_model.torch_recurrent_gated_delta_rule = _guard_kernel(qwen_model.torch_recurrent_gated_delta_rule)
    log("NAN GUARD sdpa q/k/v+output, and DeltaNet chunk kernel q/k/v/g/beta+output")


def bind_deltanet_guard(model):
    """Point already-built DeltaNet modules at the guarded kernel.

    The module copies the kernel function at init. Patching the module global
    before load is what new instances see; this covers a module that captured
    the unguarded function.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_model

    n = 0
    for mod in model.modules():
        fn = getattr(mod, "chunk_gated_delta_rule", None)
        if fn is None:
            continue
        if not getattr(fn, "_nan_guard", False):
            mod.chunk_gated_delta_rule = qwen_model.torch_chunk_gated_delta_rule
        rec = getattr(mod, "recurrent_gated_delta_rule", None)
        if rec is not None and not getattr(rec, "_nan_guard", False):
            mod.recurrent_gated_delta_rule = qwen_model.torch_recurrent_gated_delta_rule
        n += 1
        # A NaN attention output must not replace the residual. Zero only the
        # submodule output; the decoder adds it back onto the finite residual.
        if not getattr(mod.forward, "_nan_guard", False):
            orig = mod.forward

            def forward(*args, _orig=orig, **kwargs):
                out = _orig(*args, **kwargs)
                if torch.is_tensor(out):
                    return _finite(out)
                if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
                    return (_finite(out[0]),) + tuple(out[1:])
                return out

            forward._nan_guard = True
            mod.forward = forward
    log(f"NAN GUARD deltanet bound on {n} modules")


def require_fast_path():
    """Refuse the torch chunk fallback. That loop is what overflowed at 64k."""
    import causal_conv1d
    import fla
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        is_fast_path_available,
        torch_chunk_gated_delta_rule,
    )
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )

    report = {
        "fla": getattr(fla, "__version__", None),
        "causal_conv1d": getattr(causal_conv1d, "__version__", None),
        "causal_conv1d_available": bool(is_causal_conv1d_available()),
        "fla_available": bool(is_flash_linear_attention_available()),
        "is_fast_path_available": bool(is_fast_path_available),
        "torch_chunk_is_not_the_kernel": torch_chunk_gated_delta_rule is not None,
    }
    log("FAST PATH", json.dumps(report))
    if not is_fast_path_available:
        raise SystemExit(
            "Qwen fast path is not available (need flash-linear-attention and causal-conv1d). "
            "Refusing the torch chunk fallback."
        )


def _reject_fast_path_warning(records):
    hits = [r.getMessage() for r in records if "fast path is not available" in r.getMessage()]
    if hits:
        raise SystemExit("fast path warning during load: " + hits[0])


def ensure_snapshot():
    """Download into the standard HF cache if the snapshot is not already there."""
    log("DOWNLOAD", MODEL_ID)
    # huggingface_hub reads HF_HUB_OFFLINE once, at import, and the attention
    # guard imports transformers before this download.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    import huggingface_hub.constants as hub_constants
    from huggingface_hub import snapshot_download
    hub_constants.HF_HUB_OFFLINE = False
    path = Path(snapshot_download(MODEL_ID))
    hub_constants.HF_HUB_OFFLINE = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
        raise SystemExit(f"download returned no weights in {path}")
    log("DOWNLOADED", path)
    return str(path)


def load_model(path, n_gpus, gpu0_layers, quant):
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(path)
    cfg = AutoConfig.from_pretrained(path)
    text = cfg.text_config
    rope = dict(getattr(text, "rope_parameters", None) or {})
    report = {
        "model_id": MODEL_ID,
        "model_path": path,
        "architectures": list(cfg.architectures or []),
        "num_hidden_layers": text.num_hidden_layers,
        "hidden_size": text.hidden_size,
        "num_attention_heads": text.num_attention_heads,
        "num_key_value_heads": text.num_key_value_heads,
        "head_dim": text.head_dim,
        "max_position_embeddings": text.max_position_embeddings,
        "rope_parameters": rope,
        "yarn": False,
        "native_context": int(text.max_position_embeddings),
        "training_max_len": None,
    }
    if int(text.max_position_embeddings) < 65536:
        raise SystemExit(f"{MODEL_ID} native context is {text.max_position_embeddings}, under 64k")
    if rope.get("rope_type") not in (None, "default"):
        raise SystemExit(f"refusing to train with rope_type={rope.get('rope_type')}; YaRN stays off")
    log("CONTEXT", json.dumps(report))

    n_layers = text.num_hidden_layers
    rest = n_layers - gpu0_layers
    per = [gpu0_layers] + [rest // (n_gpus - 1) + (1 if i < rest % (n_gpus - 1) else 0) for i in range(n_gpus - 1)]
    if n_gpus == 1:
        per = [n_layers]
    dm = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.rotary_emb": 0,
        "lm_head": 0,
        "model.language_model.norm": n_gpus - 1,
    }
    i = 0
    for g, k in enumerate(per):
        for _ in range(k):
            dm[f"model.language_model.layers.{i}"] = g
            i += 1
    if quant != "bf16":
        raise SystemExit(f"qwen_ledger is bf16 LoRA only; refusing quant={quant}")
    import logging
    caught = []

    class _Hit(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.WARNING)

        def emit(self, record):
            caught.append(record)

    handler = _Hit()
    logging.getLogger("transformers").addHandler(handler)
    t = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, device_map=dm, attn_implementation="sdpa",
        low_cpu_mem_usage=True)
    logging.getLogger("transformers").removeHandler(handler)
    _reject_fast_path_warning(caught)
    types = list(getattr(text, "layer_types", []) or [])
    split = []
    start = 0
    for g, k in enumerate(per):
        chunk = types[start:start + k]
        split.append({
            "gpu": g, "n": k,
            "linear_attention": chunk.count("linear_attention"),
            "full_attention": chunk.count("full_attention"),
        })
        start += k
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_model
    kernel = None
    for mod in model.modules():
        fn = getattr(mod, "chunk_gated_delta_rule", None)
        if fn is not None:
            kernel = fn
            break
    kernel_name = None if kernel is None else f"{getattr(kernel, '__module__', '?')}.{getattr(kernel, '__name__', '?')}"
    log("DELTANET KERNEL", kernel_name)
    if kernel is None or kernel is qwen_model.torch_chunk_gated_delta_rule:
        raise SystemExit(f"DeltaNet is not the flash-linear-attention kernel ({kernel_name})")
    log(f"loaded model in {time.time() - t:.0f}s; layers per GPU {per}")
    log("LAYER SPLIT", json.dumps(split))
    if getattr(model.model, "visual", None) is not None:
        model.model.visual = None
        torch.cuda.empty_cache()
        log("dropped vision tower")
    model.config.use_cache = False
    model.model.language_model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    report["layers_per_gpu"] = per
    return model, tok, report


def tokenize_split(path, tok, cache_path):
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    eos = tok.eos_token
    out = []
    for line in open(path):
        r = json.loads(line)
        enc = tok.apply_chat_template(
            [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        prompt = list(enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc)
        target = tok(r["target"] + eos, add_special_tokens=False)["input_ids"]
        out.append({"id": r["id"], "bucket": r["bucket"], "label": r["label"],
                    "prompt_len": len(prompt), "target_len": len(target),
                    "ids": torch.tensor(prompt + target, dtype=torch.int32)})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(out, tmp)
    os.replace(tmp, cache_path)
    return out


def add_lora(model, rank, alpha, dropout, quant):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if quant == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS,
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def qwen_loss(model, ids, prompt_len):
    """Sum of CE on the target span. No Gemma softcap; this model has none."""
    m = train_sft.inner(model)
    lm = m.model.language_model
    dev0 = m.lm_head.weight.device
    x = ids[None].to(dev0, dtype=torch.long)
    out = lm(input_ids=x, use_cache=False)
    h = out.last_hidden_state[0, prompt_len - 1:-1].to(dev0)
    y = x[0, prompt_len:]
    weight = m.lm_head.weight
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[0], CE_CHUNK):
        hc, yc = h[s:s + CE_CHUNK], y[s:s + CE_CHUNK]

        def ce(hc, yc):
            return F.cross_entropy(F.linear(hc, weight).float(), yc, reduction="sum")

        total = total + (checkpoint(ce, hc, yc, use_reentrant=False) if torch.is_grad_enabled() else ce(hc, yc))
    return total, int(y.numel())


def _log_first_backwards():
    """Memory after each of the first 8 backwards (two optimizer steps).

    train_sft.train only prints MEM during the first step. This wrapper does
    not change that file, which ledger-D can still requeue onto.
    """
    orig = torch.Tensor.backward
    seen = {"n": 0}

    def wrapped(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if seen["n"] < 8:
            n = torch.cuda.device_count()
            alloc = [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(n)]
            log("MEM", f"after backward {seen['n']} allocated_gib={alloc} peak_gib={train_sft.gpu_mem()}")
            seen["n"] += 1

    torch.Tensor.backward = wrapped


def step0_few(model, val_ex, path, k=4):
    """Untrained val loss on the shortest examples, before any optimizer step."""
    model.eval()
    few = sorted(val_ex, key=lambda e: len(e["ids"]))[:k]
    rows = []
    with torch.no_grad():
        for ex in few:
            t = time.time()
            loss, n = qwen_loss(model, ex["ids"], ex["prompt_len"])
            lv = loss.item()
            rec = {"id": ex["id"], "seq": len(ex["ids"]), "target_len": int(ex["target_len"]),
                   "loss": lv / n if n else None, "loss_sum": lv, "n": n,
                   "finite": bool(n > 0 and lv == lv and abs(lv) != float("inf")),
                   "sec": round(time.time() - t, 1)}
            rows.append(rec)
            log("STEP0", json.dumps(rec))
    finite = [r["loss"] for r in rows if r["finite"]]
    report = {"n": len(rows), "mean_loss": (sum(finite) / len(finite)) if finite else None,
              "all_finite": len(finite) == len(rows), "examples": rows,
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    # A finite loss near ln(248320)=12.42 is a uniform head,
    # which is what a zeroed hidden state produces. That is not a usable step 0.
    collapsed = [r for r in rows if r["finite"] and r["loss"] is not None and r["loss"] > 5.0]
    report["collapsed"] = [r["id"] for r in collapsed]
    train_sft.write_json(path, report)
    if not report["all_finite"] or collapsed:
        log("STEP0 REJECT", "nonfinite" if not report["all_finite"] else "collapsed hidden (loss>5)")
        raise SystemExit(2)
    return report


def main():
    ap = __import__("argparse").ArgumentParser(fromfile_prefix_chars="@")
    ap.convert_arg_line_to_args = lambda line: [] if not line.strip() or line.lstrip().startswith("#") else [line.strip()]
    ap.add_argument("--mode", choices=["train"], default="train")
    ap.add_argument("--data-dir", default="data", help="relative paths resolve under DATA_ROOT")
    ap.add_argument("--keep-ids", default="ledger_keep_ids.json")
    ap.add_argument("--run-dir", required=True, help="relative paths resolve under OUT_ROOT")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--quant", choices=["bf16"], default="bf16")
    ap.add_argument("--max-len", type=int, default=65536)
    ap.add_argument("--gpu0-layers", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eff-batch", type=int, default=4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every-sec", type=int, default=300)
    args = ap.parse_args()
    args.data_dir = str(DATA_ROOT / args.data_dir)
    if args.keep_ids:
        args.keep_ids = str(DATA_ROOT / args.keep_ids)
    args.run_dir = str(OUT_ROOT / args.run_dir)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    n_gpus = torch.cuda.device_count()
    log("GPUs:", [torch.cuda.get_device_name(i) for i in range(n_gpus)], "args:", vars(args))
    # No NaN replacement on DeltaNet or SDPA. A nonfinite step 0 still aborts.
    log("NAN GUARD off")
    require_fast_path()
    path = args.model_path or ensure_snapshot()
    model, tok, report = load_model(path, n_gpus, args.gpu0_layers, args.quant)
    report["training_max_len"] = args.max_len
    report["yarn"] = False
    train_sft.write_json(run_dir / "context.json", report)
    train_sft.write_json(DATA_ROOT / "runs/qwen_ledger_context.json", report)

    cache = Path(args.data_dir) / "_tok_qwen35"
    train_ex = tokenize_split(Path(args.data_dir) / "train.jsonl", tok, cache / "train.pt")
    val_ex = tokenize_split(Path(args.data_dir) / "val.jsonl", tok, cache / "val.pt")
    ex0 = train_ex[0]
    prompt_text = tok.decode(ex0["ids"][:ex0["prompt_len"]].tolist())
    if "</think>" not in prompt_text:
        raise SystemExit("Qwen chat template did not close thinking; refusing to train")
    log("TAIL", repr(tok.decode(ex0["ids"][ex0["prompt_len"] - 16:ex0["prompt_len"] + 12].tolist())))
    log("END", repr(tok.decode(ex0["ids"][-6:].tolist())))

    model = add_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, args.quant)
    m = train_sft.inner(model)
    lm = m.model.language_model
    for p in (lm.embed_tokens.weight, m.lm_head.weight):
        if p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    # kbit prep casts every non-4bit parameter to fp32. DeltaNet's conv1d then
    # rejects the bf16 activation. Norms stay fp32; their forward casts back.
    n_conv = 0
    for mod in lm.modules():
        if isinstance(mod, torch.nn.Conv1d) and mod.weight.dtype != torch.bfloat16:
            mod.weight.data = mod.weight.data.to(torch.bfloat16)
            n_conv += 1
        for attr in ("dt_bias", "A_log"):
            p = getattr(mod, attr, None)
            if isinstance(p, torch.nn.Parameter) and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_dtypes = sorted({str(p.dtype).replace("torch.", "") for n, p in model.named_parameters() if p.requires_grad})
    base_dtypes = sorted({str(p.dtype).replace("torch.", "") for n, p in model.named_parameters() if not p.requires_grad})
    log(f"trainable params {n_tr / 1e6:.1f}M; lora {lora_dtypes}; base {base_dtypes}; conv1d restored {n_conv}")
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.train()

    def has_target(e):
        n, p = len(e["ids"]), int(e["prompt_len"])
        return 0 < p < n and int(e["target_len"]) > 0

    def drop_empty(rows, split):
        kept = [e for e in rows if has_target(e)]
        gone = [e["id"] for e in rows if not has_target(e)]
        if gone:
            log(f"DROP EMPTY TARGET {split}", len(gone))
        return kept

    keep_tr = drop_empty([e for e in train_ex if len(e["ids"]) <= args.max_len], "train")
    keep_va = drop_empty([e for e in val_ex if len(e["ids"]) <= args.max_len], "val")
    spec = json.loads(Path(args.keep_ids).read_text())

    def take(rows, wanted, split):
        by_id = {e["id"]: e for e in rows}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            over = []
            src = train_ex if split == "train" else val_ex
            by_all = {e["id"]: e for e in src}
            for i in missing:
                e = by_all.get(i)
                over.append({"id": i, "seq": None if e is None else len(e["ids"])})
            train_sft.write_json(run_dir / "length_mismatch.json", {"split": split, "missing": over})
            raise SystemExit(
                f"keep-ids: {len(missing)} {split} ids are not in this data under max_len {args.max_len}")
        return [by_id[i] for i in wanted]

    keep_tr = take(keep_tr, spec["train"]["ids"], "train")
    keep_va = take(keep_va, spec["val"]["ids"], "val")
    log(f"KEEP IDS train {len(keep_tr)} val {len(keep_va)}")
    lens = [len(e["ids"]) for e in keep_tr]
    log(f"TRAIN LEN min {min(lens)} max {max(lens)} n {len(lens)}")
    train_sft.write_json(run_dir / "examples_used.json", {
        "max_len": args.max_len,
        "model_id": MODEL_ID,
        "train": {"used": len(keep_tr), "ids": [e["id"] for e in keep_tr]},
        "val": {"used": len(keep_va), "ids": [e["id"] for e in keep_va]},
    })

    probe_path = DATA_ROOT / "runs/qwen_ledger_step0.json"
    if not (run_dir / "state.pt").exists():
        step0_few(model, keep_va, probe_path, k=4)
    else:
        log("RESUME skip step0 probe; state.pt exists")

    os.environ["SFT_SKIP_SPOTCHECK"] = "1"
    train_sft.target_loss_sum = qwen_loss
    _log_first_backwards()
    train_sft.train(args, model, keep_tr, keep_va, run_dir)


if __name__ == "__main__":
    main()
