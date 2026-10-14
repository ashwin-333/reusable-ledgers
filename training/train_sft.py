"""LoRA SFT of Gemma-4-31B-it on long trajectories, preemption-safe.

One process drives all visible GPUs (layer-wise model parallelism). The base
model is frozen; LoRA adapters are trained on every attention and MLP
projection of the language model. Loss is computed on target tokens only.

Memory at long context:
- masks are never materialized: sliding layers run blockwise SDPA with a
  (W x 2W) band mask, global layers run causal SDPA (memory-efficient kernel);
- decoder layers are gradient-checkpointed, and inside a layer the MLP is
  evaluated in sequence chunks under nested checkpointing;
- logits are only formed for target positions, in chunks.

Preemption: the full training state (adapter, optimizer, scheduler, partial
gradient accumulation, data position, partial eval results) is saved
atomically between micro-batches, so a restart loses at most one micro-batch.
"""
import argparse
import glob
import json
import math
import os
import random
import subprocess
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from config import DATA_ROOT, OUT_ROOT, WRITER_MODEL, snapshot_dir

END_OF_TURN = "<turn|>"
MLP_CHUNK = 4096
CE_CHUNK = 1024
GLOBAL_Q_CHUNK = 4096
# 12B global layers have 1 KV head and 16 query heads at dim 512. One MATH
# SDPA over all query heads materializes [16, q_chunk, seq] fp32 scores
# (~16 GiB at 64k) plus a dense causal mask of the same size. One query head
# at a time keeps that tile near 1 GiB.
GLOBAL_HEAD_CHUNK = 1
LORA_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"

ATTN_CALLS = {"sliding": 0, "global": 0}
# When on, global layers run one KV group at a time under checkpointing, so the
# 8x-expanded K/V and their grads exist for one group only (less memory, one
# extra attention forward in backward).
GLOBAL_GROUP_CKPT = {"on": False}


def _math_sdpa(q, k, v, *, is_causal=False, attn_mask=None, scale=None):
    """Per-head attention with the softmax in fp32.

    bf16 MATH SDPA softmax overflows once a score is large (global head_dim
    512 with scaling 1), and the NaNs then sit in the residual. Eager Gemma
    attention upcasts the same way. The causal mask is bottom-right, matching
    SDPA's is_causal. A bool attn_mask uses True for positions that attend.
    """
    # A nonfinite query or key makes every score inf, and softmax(inf, -inf) is NaN.
    # Finite values are left unchanged. This only replaces overflow.
    qf = torch.nan_to_num(q.float(), nan=0.0, posinf=0.0, neginf=0.0)
    kf = torch.nan_to_num(k.float(), nan=0.0, posinf=0.0, neginf=0.0)
    vf = torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0)
    scores = torch.matmul(qf, kf.transpose(-2, -1))
    if scale is None:
        scale = q.shape[-1] ** -0.5
    scores = scores * scale
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if is_causal:
        qlen, klen = scores.shape[-2:]
        # triu diagonal klen - qlen + 1 is the strict future under bottom-right alignment.
        future = torch.ones(qlen, klen, device=scores.device, dtype=torch.bool).triu(klen - qlen + 1)
        scores = scores.masked_fill(future, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask.float()
    probs = torch.softmax(scores, dim=-1)
    # A fully masked row is all -inf and softmax returns NaN. Drop that row.
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, vf).to(dtype=q.dtype)


def _causal_heads(q, k, v, scaling):
    """Causal SDPA for a few query heads. k/v may be one KV head, broadcast by repeat."""
    n = q.shape[1]
    if k.shape[1] != n:
        k = k.repeat(1, n, 1, 1)
        v = v.repeat(1, n, 1, 1)
    # Chunks with q_len < kv_len rely on PyTorch's bottom-right causal alignment:
    # queries at s:e against keys 0:e is exactly "key <= query".
    return _math_sdpa(q, k, v, is_causal=True, scale=scaling)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- attention
def blockwise_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                   sliding_window=None, **kwargs):
    """query [b, hq, L, d]; key/value [b, hkv, L, d]. Causal (optionally windowed)."""
    L = query.shape[2]
    rep = query.shape[1] // key.shape[1]
    if sliding_window is None:
        ATTN_CALLS["global"] += 1
        outs = []
        grad = torch.is_grad_enabled()
        for s in range(0, L, GLOBAL_Q_CHUNK):
            e = min(s + GLOBAL_Q_CHUNK, L)
            q = query[:, :, s:e]
            k = key[:, :, :e]
            v = value[:, :, :e]
            chunks = []
            for g in range(key.shape[1]):
                qg, kg, vg = q[:, g * rep:(g + 1) * rep], k[:, g:g + 1], v[:, g:g + 1]
                for h0 in range(0, qg.shape[1], GLOBAL_HEAD_CHUNK):
                    qh = qg[:, h0:h0 + GLOBAL_HEAD_CHUNK]
                    if grad:
                        chunks.append(checkpoint(_causal_heads, qh, kg, vg, scaling, use_reentrant=False))
                    else:
                        chunks.append(_causal_heads(qh, kg, vg, scaling))
            outs.append(chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1))
        return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None
    ATTN_CALLS["sliding"] += 1
    if rep > 1:
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
    W = sliding_window
    outs = []
    grad = torch.is_grad_enabled()
    for s in range(0, L, W):
        e = min(s + W, L)
        ks = max(0, s - W + 1)
        qpos = torch.arange(s, e, device=query.device)[:, None]
        kpos = torch.arange(ks, e, device=query.device)[None, :]
        band = (kpos <= qpos) & (kpos > qpos - W)
        qb, kb, vb = query[:, :, s:e], key[:, :, ks:e], value[:, :, ks:e]
        if grad:
            outs.append(checkpoint(_sliding_block, qb, kb, vb, band, scaling, use_reentrant=False))
        else:
            outs.append(_sliding_block(qb, kb, vb, band, scaling))
    return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None


def _sliding_block(q, k, v, band, scaling):
    return _math_sdpa(q, k, v, attn_mask=band, scale=scaling)


def patch_model_code():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    from transformers.models.gemma4 import modeling_gemma4 as mg

    AttentionInterface.register("blockwise_sdpa", blockwise_sdpa)
    AttentionMaskInterface.register("blockwise_sdpa", sdpa_mask)

    orig = mg.Gemma4TextMLP.forward

    def chunked_mlp(self, x):
        L = x.shape[1]
        if L <= MLP_CHUNK:
            return orig(self, x)
        grad = torch.is_grad_enabled()
        outs = []
        for s in range(0, L, MLP_CHUNK):
            xs = x[:, s:s + MLP_CHUNK]
            outs.append(checkpoint(orig, self, xs, use_reentrant=False) if grad else orig(self, xs))
        return torch.cat(outs, dim=1)

    mg.Gemma4TextMLP.forward = chunked_mlp


# ---------------------------------------------------------------- data
def tokenize_split(path, tok, cache_path):
    if cache_path.exists():
        return torch.load(cache_path)
    out = []
    for line in open(path):
        r = json.loads(line)
        enc = tok.apply_chat_template(
            [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
            tokenize=True, add_generation_prompt=True)
        prompt = list(enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc)
        target = tok(r["target"] + END_OF_TURN, add_special_tokens=False)["input_ids"]
        out.append({"id": r["id"], "bucket": r["bucket"], "label": r["label"],
                    "prompt_len": len(prompt), "target_len": len(target),
                    "ids": torch.tensor(prompt + target, dtype=torch.int32)})
    tmp = cache_path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(out, tmp)
    os.replace(tmp, cache_path)
    return out


def length_table(examples, caps):
    buckets = sorted({e["bucket"] for e in examples}, key=lambda b: (b != "0-10k", len(b), b))
    rows = {}
    for b in buckets:
        xs = [len(e["ids"]) for e in examples if e["bucket"] == b]
        rows[b] = {"n": len(xs), "max_tokens": max(xs), **{f"fit_{c}": sum(x <= c for x in xs) for c in caps}}
    return rows


# ---------------------------------------------------------------- model
def build_model(n_gpus, gpu0_layers, quant, model_path):
    from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(model_path)
    cfg = AutoConfig.from_pretrained(model_path)
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
    n_layers = text.num_hidden_layers
    rest = n_layers - gpu0_layers
    per = [gpu0_layers] + [rest // (n_gpus - 1) + (1 if i < rest % (n_gpus - 1) else 0) for i in range(n_gpus - 1)]
    per = per if n_gpus > 1 else [n_layers]
    dm = {"model.vision_tower": 0, "model.embed_vision": 0,
          "model.audio_tower": 0, "model.embed_audio": 0,
          "model.language_model.embed_tokens": 0,
          "model.language_model.rotary_emb": 0, "lm_head": 0,
          "model.language_model.norm": n_gpus - 1}
    i = 0
    for g, k in enumerate(per):
        for _ in range(k):
            dm[f"model.language_model.layers.{i}"] = g
            i += 1
    kwargs = {}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["vision_tower", "embed_vision", "audio_tower", "embed_audio", "lm_head"])
    t = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=dm, attn_implementation="sdpa",
        low_cpu_mem_usage=True, **kwargs)
    log(f"loaded model in {time.time()-t:.0f}s; layers per GPU {per}")
    dropped = False
    for name in ("vision_tower", "embed_vision", "audio_tower", "embed_audio"):
        if getattr(model.model, name, None) is not None:
            setattr(model.model, name, None)
            dropped = True
    if dropped:
        torch.cuda.empty_cache()
        log("dropped unused vision/audio towers")
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, per


def set_attn(model, impl):
    lm = model.model.language_model
    lm.config._attn_implementation = impl
    assert all(l.self_attn.config._attn_implementation == impl for l in lm.layers)


def add_lora(model, rank, alpha, dropout, quant):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if quant == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS,
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    return model


def inner(model):
    """Unwrap PEFT wrappers until Gemma4ForConditionalGeneration."""
    m = model
    for _ in range(8):
        if hasattr(m, "lm_head") and hasattr(m, "model") and hasattr(m.model, "language_model"):
            return m
        m = m.base_model.model if hasattr(m, "base_model") else m.model
    raise RuntimeError(f"could not unwrap {type(model)}")


def enable_gc(model):
    m = inner(model)
    # Reentrant checkpoint runs the forward under no_grad, so 4-bit and LoRA
    # activations are not kept for every layer. Non-reentrant still held about
    # 2 GiB per layer at 64k and filled the second GPU before the step finished.
    m.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    assert m.model.language_model.gradient_checkpointing or all(
        getattr(l, "gradient_checkpointing", False) for l in m.model.language_model.layers)


def _softcap(logits, cap):
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return logits


def _chunked_ce_sum(h_tgt, y, weight, cap):
    """lm_head on the target span only, in chunks. h_tgt predicts y."""
    total = h_tgt.new_zeros((), dtype=torch.float32)
    for s in range(0, h_tgt.shape[0], CE_CHUNK):
        hc, yc = h_tgt[s:s + CE_CHUNK], y[s:s + CE_CHUNK]

        def ce(hc, yc):
            return F.cross_entropy(_softcap(F.linear(hc, weight).float(), cap), yc, reduction="sum")

        total = total + (checkpoint(ce, hc, yc, use_reentrant=False) if torch.is_grad_enabled() else ce(hc, yc))
    return total


def target_loss_sum(model, ids, prompt_len):
    """Sum of CE over target tokens (ids[prompt_len:]) and their count.

    The vocab is ~262k, so lm_head is applied only to the target positions.
    A full-sequence logit tensor at 48k-64k would be tens of GB on the head GPU.
    """
    m = inner(model)
    lm = m.model.language_model
    dev0 = m.lm_head.weight.device
    x = ids[None].to(dev0, dtype=torch.long)
    out = lm(input_ids=x, attention_mask={"full_attention": None, "sliding_attention": None}, use_cache=False)
    h = out.last_hidden_state[0, prompt_len - 1:-1].to(dev0)
    y = x[0, prompt_len:]
    total = _chunked_ce_sum(h, y, m.lm_head.weight, m.config.get_text_config().final_logit_softcapping)
    return total, y.numel()


def verify_sliced_loss(model, ids, prompt_len):
    """Sliced, chunked lm_head must match CE on the full logit tensor for a short sequence."""
    was_training = model.training
    model.eval()
    m = inner(model)
    weight = m.lm_head.weight
    dev0 = weight.device
    x = ids[None].to(dev0, dtype=torch.long)
    with torch.no_grad():
        h = m.model.language_model(
            input_ids=x, attention_mask={"full_attention": None, "sliding_attention": None},
            use_cache=False).last_hidden_state[0].to(weight.device)
        cap = m.config.get_text_config().final_logit_softcapping
        y = x[0, prompt_len:]
        full = _softcap(F.linear(h, weight).float(), cap)
        standard = F.cross_entropy(full[prompt_len - 1:-1], y, reduction="sum")
        del full
        sliced = _chunked_ce_sum(h[prompt_len - 1:-1], y, weight, cap)
        model.eval()
        trained, n = target_loss_sum(model, ids, prompt_len)
    if was_training:
        model.train()
    standard_v, sliced_v, trained_v = (float(standard), float(sliced), float(trained))
    rel_sliced = abs(sliced_v - standard_v) / max(abs(standard_v), 1e-8)
    rel_second = abs(trained_v - standard_v) / max(abs(standard_v), 1e-8)
    rec = {"seq_len": int(ids.numel()), "prompt_len": int(prompt_len), "n_target": int(n),
           "loss_standard": standard_v, "loss_sliced": sliced_v, "loss_target_loss_sum": trained_v,
           "rel_sliced": rel_sliced, "rel_second_forward": rel_second,
           "model_training": bool(model.training), "passed": rel_sliced < 1e-4}
    log("LOSS CHECK", json.dumps(rec))
    if not rec["passed"]:
        raise SystemExit("sliced lm_head loss does not match the full-logit loss")
    return rec


def gpu_mem():
    return [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]


def reset_mem():
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


# ---------------------------------------------------------------- selftest
def _rel(u, v):
    return (u - v).abs().norm().item() / u.norm().clamp_min(1e-8).item()


def probe_standard_kernels():
    """Memory-efficient SDPA / Flash without expanding K to the query head count."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def one(backend, hq, hkv, length, dim):
        q = torch.randn(1, hq, length, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, hkv, length, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, hkv, length, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        try:
            with sdpa_kernel([backend]):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0, enable_gqa=hq != hkv)
                out.sum().backward()
            return "ok"
        except Exception as e:
            return f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        finally:
            del q, k, v
            torch.cuda.empty_cache()

    report = {}
    try:
        import flash_attn
        report["flash_attn_package"] = getattr(flash_attn, "__version__", "installed")
    except Exception as e:
        report["flash_attn_package"] = f"not installed ({type(e).__name__})"
    for backend in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        report[f"global_d512_{backend.name}"] = one(backend, 32, 4, 2048, 512)
        report[f"sliding_d256_{backend.name}"] = one(backend, 32, 16, 1024, 256)
    usable = [k for k, v in report.items() if v == "ok" and k.startswith(("global", "sliding"))]
    report["usable_without_kv_expansion"] = usable
    log("STD ATTN", json.dumps(report))
    return report


def _flat_grads(model):
    parts = [p.grad.detach().float().reshape(-1) for p in model.parameters() if p.requires_grad]
    return torch.cat(parts).cpu()


def grad_check(model, examples, seq_len=2048):
    """Loss and LoRA gradients of blockwise attention vs HF sdpa, on one short example."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    was = model.training
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    enable_gc(model)
    ex = min(examples, key=lambda e: len(e["ids"]))
    ids = ex["ids"][:seq_len]
    prompt_len = seq_len - min(128, ex["target_len"])

    def run(impl):
        set_attn(inner(model), impl)
        model.zero_grad(set_to_none=True)
        # sdpa must build its own masks. Passing {None, None} makes every layer
        # fully causal, which is not the Gemma 4 sliding-window pattern.
        m = inner(model)
        dev0 = m.lm_head.weight.device
        x = ids[None].to(dev0, dtype=torch.long)
        lm = m.model.language_model
        if impl == "sdpa":
            hidden = lm(input_ids=x, use_cache=False).last_hidden_state
        else:
            hidden = lm(input_ids=x, attention_mask={"full_attention": None, "sliding_attention": None},
                        use_cache=False).last_hidden_state
        h = hidden[0, prompt_len - 1:-1].to(dev0)
        y = x[0, prompt_len:]
        weight = m.lm_head.weight
        cap = m.config.get_text_config().final_logit_softcapping

        def ce(hc, yc):
            logits = F.linear(hc, weight).float()
            if cap:
                logits = torch.tanh(logits / cap) * cap
            return F.cross_entropy(logits, yc, reduction="sum")

        loss = h.new_zeros((), dtype=torch.float32)
        for start in range(0, h.shape[0], CE_CHUNK):
            loss = loss + checkpoint(ce, h[start:start + CE_CHUNK], y[start:start + CE_CHUNK], use_reentrant=False)
        (loss / y.numel()).backward()
        return loss.item() / y.numel(), _flat_grads(model)

    with sdpa_kernel([SDPBackend.MATH]):
        loss_std, grad_std = run("sdpa")
        loss_ours, grad_ours = run("blockwise_sdpa")
    rel_loss = abs(loss_std - loss_ours) / max(abs(loss_std), 1e-8)
    rel_grad = (grad_std - grad_ours).norm().item() / max(grad_std.norm().item(), 1e-8)
    info = {
        "seq_len": int(ids.shape[0]),
        "backend": "MATH",
        "loss_sdpa": loss_std,
        "loss_blockwise": loss_ours,
        "rel_loss": rel_loss,
        "rel_grad": rel_grad,
        "max_abs_grad": (grad_std - grad_ours).abs().max().item(),
    }
    log("GRADCHECK", json.dumps(info))
    if was:
        model.train()
    if rel_loss > 1e-2 or rel_grad > 2e-2:
        raise AssertionError(f"custom attention disagrees with sdpa on {seq_len} tokens: {info}")
    return info


def _rel_to(ref, other):
    denom = ref.detach().float().norm().item()
    return (ref.detach().float() - other.detach().float()).norm().item() / max(denom, 1e-12)


def _fp32_attn_pair(name, hq, hkv, dim, seq, window):
    """One attention call: eager matmul vs KV-group blockwise, both fp32."""
    from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward

    torch.manual_seed(0)
    q = torch.randn(1, hq, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(1, hkv, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(1, hkv, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
    module = type("M", (), {"num_key_value_groups": hq // hkv, "head_dim": dim, "training": False})()
    qpos = torch.arange(seq, device="cuda")[:, None]
    kpos = torch.arange(seq, device="cuda")[None, :]
    allow = (kpos <= qpos) if window is None else ((kpos <= qpos) & (kpos > qpos - window))
    additive = torch.zeros(1, 1, seq, seq, device="cuda", dtype=torch.float32)
    additive.masked_fill_(~allow, torch.finfo(torch.float32).min)

    def score(out):
        return out.float().square().mean()

    loss_e = score(eager_attention_forward(module, q, k, v, additive, dropout=0.0, scaling=1.0)[0])
    loss_e.backward()
    grad_e = torch.cat([t.grad.detach().reshape(-1).float() for t in (q, k, v)])
    for t in (q, k, v):
        t.grad = None
    loss_o = score(blockwise_sdpa(module, q, k, v, None, scaling=1.0, sliding_window=window)[0])
    loss_o.backward()
    grad_o = torch.cat([t.grad.detach().reshape(-1).float() for t in (q, k, v)])
    info = {
        "name": name,
        "seq_len": seq,
        "sliding_window": window,
        "loss_eager": loss_e.item(),
        "loss_blockwise": loss_o.item(),
        "rel_loss": abs(loss_e.item() - loss_o.item()) / max(abs(loss_e.item()), 1e-12),
        "rel_grad": _rel_to(grad_e, grad_o),
        "max_abs_grad": (grad_e - grad_o).abs().max().item(),
    }
    log("FP32 ATTN", json.dumps(info))
    del q, k, v, grad_e, grad_o, additive
    torch.cuda.empty_cache()
    return info


def fp32_eager_check(seq_len=2048, rel_tol=1e-4):
    """fp32 loss and gradients vs eager attention, with real sliding-window masks.

    Direct cases use Gemma-4 head shapes (global d=512 GQA 32/4, sliding d=256
    GQA 32/16, window 1024). The model case is a 2-layer random Gemma4TextModel
    so eager builds the HF causal and sliding masks, while blockwise builds its
    own. Global queries are chunked at 512 so the s>0 causal mask is exercised
    on this short sequence.
    """
    from transformers import Gemma4TextConfig, Gemma4TextModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    global GLOBAL_Q_CHUNK
    prev_chunk = GLOBAL_Q_CHUNK
    GLOBAL_Q_CHUNK = 512
    try:
        direct = [
            _fp32_attn_pair("global_d512", 32, 4, 512, seq_len, None),
            _fp32_attn_pair("sliding_d256_w1024", 32, 16, 256, seq_len, 1024),
        ]
        cfg = Gemma4TextConfig(
            vocab_size=128,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=32,
            num_key_value_heads=16,
            num_global_key_value_heads=4,
            head_dim=256,
            global_head_dim=512,
            sliding_window=1024,
            attention_k_eq_v=True,
            attention_dropout=0.0,
            attention_bias=False,
            num_kv_shared_layers=0,
            max_position_embeddings=seq_len,
            final_logit_softcapping=None,
            layer_types=["sliding_attention", "full_attention"],
            use_cache=False,
        )
        torch.manual_seed(1)
        lm = Gemma4TextModel(cfg).to(device="cuda", dtype=torch.float32).train()
        assert [layer.self_attn.layer_type for layer in lm.layers] == ["sliding_attention", "full_attention"]
        assert lm.layers[0].self_attn.sliding_window == 1024
        assert lm.layers[1].self_attn.sliding_window is None
        assert lm.layers[1].self_attn.head_dim == 512 and lm.layers[0].self_attn.head_dim == 256
        assert lm.layers[1].self_attn.v_proj is None
        head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).to(device="cuda", dtype=torch.float32)
        for module in lm.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device="cuda")
        n_tgt = 128
        prompt_len = seq_len - n_tgt
        ATTN_CALLS["sliding"] = 0
        ATTN_CALLS["global"] = 0
        params = [p for p in list(lm.parameters()) + list(head.parameters()) if p.requires_grad]

        def run(impl, mask):
            lm.config._attn_implementation = impl
            for layer in lm.layers:
                layer.self_attn.config._attn_implementation = impl
            lm.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            hidden = lm(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            logits = head(hidden[0, prompt_len - 1:-1])
            loss = F.cross_entropy(logits.float(), ids[0, prompt_len:])
            loss.backward()
            flat = torch.cat([p.grad.detach().reshape(-1).float() for p in params])
            return loss.detach(), flat

        loss_e, grad_e = run("eager", None)
        loss_o, grad_o = run("blockwise_sdpa", {"full_attention": None, "sliding_attention": None})
        model_info = {
            "name": "gemma4_2layer_ce",
            "seq_len": seq_len,
            "sliding_window": 1024,
            "global_q_chunk": GLOBAL_Q_CHUNK,
            "loss_eager": loss_e.item(),
            "loss_blockwise": loss_o.item(),
            "rel_loss": abs(loss_e.item() - loss_o.item()) / max(abs(loss_e.item()), 1e-12),
            "rel_grad": _rel_to(grad_e, grad_o),
            "max_abs_grad": (grad_e - grad_o).abs().max().item(),
            "attn_calls": dict(ATTN_CALLS),
        }
        log("FP32 MODEL", json.dumps(model_info))
        cases = direct + [model_info]
        passed = all(c["rel_loss"] <= rel_tol and c["rel_grad"] <= rel_tol for c in cases)
        return {
            "passed": passed,
            "rel_tol": rel_tol,
            "dtype": "float32",
            "reference": "eager_attention",
            "seq_len": seq_len,
            "cases": cases,
            "rel_loss": model_info["rel_loss"],
            "rel_grad": model_info["rel_grad"],
        }
    finally:
        GLOBAL_Q_CHUNK = prev_chunk


def selftest_tiling():
    """Blockwise SDPA must match one-shot SDPA with the same mask formula."""
    dev = "cuda:0"
    cases = []
    for name, L, W, d, hq, hkv in (
        ("sliding_4k_d256", 4096, 1024, 256, 32, 16),
        ("global_4k_d512", 4096, None, 512, 32, 4),
        ("global_4k_d512_gqa1", 4096, None, 512, 16, 1),
    ):
        q = torch.randn(1, hq, L, d, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, hkv, L, d, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, hkv, L, d, device=dev, dtype=torch.bfloat16)
        krep, vrep = k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1)
        if W is None:
            ref = F.scaled_dot_product_attention(q, krep, vrep, is_causal=True, scale=1.0)
        else:
            qpos = torch.arange(L, device=dev)[:, None]
            kpos = torch.arange(L, device=dev)[None, :]
            band = (kpos <= qpos) & (kpos > qpos - W)
            ref = F.scaled_dot_product_attention(q, krep, vrep, attn_mask=band, scale=1.0)
        ours, _ = blockwise_sdpa(None, q, k, v, None, scaling=1.0, sliding_window=W)
        ours = ours.transpose(1, 2)
        rec = {"name": name, "rel": _rel(ref.float(), ours.float()),
               "max_abs": (ref.float() - ours.float()).abs().max().item()}
        cases.append(rec)
        log("TILETEST", rec)
        assert rec["rel"] < 5e-3, f"blockwise tiling mismatch: {rec}"
    return cases


def selftest(model, examples):
    """Check blockwise tiling, then log (not fail) a full-model compare to HF sdpa."""
    tile = selftest_tiling()
    was_training = model.training
    model.eval()
    ex = min(examples, key=lambda e: len(e["ids"]))
    L = 4096
    ids = ex["ids"][-L:] if len(ex["ids"]) > L else ex["ids"]
    P = len(ids) - min(ex["target_len"], 800)
    m = inner(model)
    lm = m.model.language_model
    x = ids[None].to(next(lm.parameters()).device).long()

    def hidden(impl):
        set_attn(m, impl)
        if impl == "sdpa":
            return lm(input_ids=x, use_cache=False).last_hidden_state[0, P - 1:-1].float().cpu()
        ATTN_CALLS.update({"sliding": 0, "global": 0})
        return lm(input_ids=x, attention_mask={"full_attention": None, "sliding_attention": None},
                  use_cache=False).last_hidden_state[0, P - 1:-1].float().cpu()

    with torch.no_grad():
        a = hidden("sdpa")
        c = hidden("blockwise_sdpa")
        set_attn(m, "blockwise_sdpa")
        lb, n = target_loss_sum(model, ids, P)
    info = {"seq_len": len(ids), "rel_err_hidden": _rel(a, c), "max_abs_err": (a - c).abs().max().item(),
            "loss_blockwise": lb.item() / n, "attn_calls": dict(ATTN_CALLS), "tile": tile}
    log("SELFTEST", json.dumps(info))
    if was_training:
        model.train()
    return info


# ---------------------------------------------------------------- memtest
def memtest(model, examples, lengths, out_path, steps=2):
    # Attention changed (one global query head, no dense mask). Re-check tiling
    # even when the full-model selftest is skipped.
    selftest_tiling()
    lm = inner(model).model.language_model

    def _layer_mem(i, _mod, args):
        if i % 8 == 0:
            alloc = [round(torch.cuda.memory_allocated(d) / 2**30, 2) for d in range(torch.cuda.device_count())]
            hs = args[0] if args else None
            kind = None if hs is None else str(hs.dtype).replace("torch.", "")
            log("LAYER", i, "alloc_gib", alloc, "hidden", kind)

    for i, layer in enumerate(lm.layers):
        layer.register_forward_pre_hook(lambda mod, args, i=i: _layer_mem(i, mod, args))
    donor = max(examples, key=lambda e: len(e["ids"]))
    short_n = 1024
    short = donor["ids"][:short_n].long()
    checked = verify_sliced_loss(model, short, short_n - 64)
    out_path.with_name("loss_check.json").write_text(json.dumps(checked, indent=2) + "\n")
    model.train()
    loss, n = target_loss_sum(model, short, short_n - 64)
    (loss / n).backward()
    model.zero_grad(set_to_none=True)
    del loss
    torch.cuda.empty_cache()
    log("SHORT BACKWARD ok")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-7)
    results = []
    # Lengths are shortest-first. A longer context cannot fit after a shorter one failed.
    # After a success, keep going so the longest success is the one we train.
    for L in lengths:
        tgt = min(donor["target_len"], 4000)
        ids = torch.cat([donor["ids"][:L - tgt], donor["ids"][-tgt:]])
        assert len(ids) == L
        kept = [e for e in examples if len(e["ids"]) <= L]
        rec = {"max_len": L, "global_group_ckpt": True, "ok": False, "n_train_kept": len(kept)}
        try:
            reset_mem()
            for s in range(steps):
                torch.cuda.synchronize()
                t = time.time()
                loss, n = target_loss_sum(model, ids, L - tgt)
                (loss / n).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                rec[f"step{s}_sec"] = round(time.time() - t, 1)
            sec = rec.get("step1_sec", rec.get("step0_sec"))
            tokens = sum(len(e["ids"]) for e in kept)
            rec.update(ok=True, loss=round(loss.item() / n, 4),
                       step_sec=sec,
                       epoch_sec_upper=round(len(kept) * sec, 1),
                       epoch_sec_token_linear=round(sec * tokens / L, 1),
                       train_tokens_kept=tokens)
        except torch.OutOfMemoryError as e:
            rec["error"] = str(e).split("\n")[0][:300]
            rec["trace"] = traceback.format_exc().strip().splitlines()[-6:]
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise
            rec["error"] = str(e).split("\n")[0][:300]
            rec["trace"] = traceback.format_exc().strip().splitlines()[-6:]
        rec["peak_gib_per_gpu"] = gpu_mem()
        rec["total_gib_per_gpu"] = [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)
                                    for i in range(torch.cuda.device_count())]
        log("MEMTEST", json.dumps(rec))
        results.append(rec)
        opt.zero_grad(set_to_none=True)
        loss = None
        torch.cuda.empty_cache()
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        if not rec["ok"]:
            later = lengths[lengths.index(L) + 1:]
            if later and later[0] < L:
                log("MEMTEST", L, "did not fit; trying shorter", later[0])
            else:
                log("MEMTEST stop: longer contexts skipped after", L)
                break
    return results


# ---------------------------------------------------------------- training
def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def lr_lambda_factory(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return f


def save_adapter(model, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))


def train(args, model, train_ex, val_ex, run_dir):
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    n_train = len(train_ex)
    spe = math.ceil(n_train / args.eff_batch)
    total_steps = spe * args.epochs
    warmup = max(1, math.ceil(args.warmup_ratio * total_steps))
    eval_steps = sorted({min(total_steps, round(k * spe / 2)) for k in range(1, 2 * args.epochs + 1)})
    eval_points = [0] + eval_steps
    plan = {"n_train": n_train, "n_val": len(val_ex), "steps_per_epoch": spe, "total_steps": total_steps,
            "warmup_steps": warmup, "eval_after_steps": eval_points, "lr": args.lr}
    log("PLAN", json.dumps(plan))

    opt = torch.optim.AdamW(list(trainable.values()), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total_steps, warmup))

    def order(epoch):
        idx = list(range(n_train))
        random.Random(args.seed + epoch).shuffle(idx)
        return idx

    state_path = run_dir / "state.pt"
    hist_path = run_dir / "history.json"
    st = {"step": 0, "micro": 0, "evals": {}, "eval_partial": None, "train_log": [], "done": False,
          "interval_loss": [0.0, 0], "diverged": None}
    if state_path.exists():
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        for n, p in trainable.items():
            p.data.copy_(ck["params"][n].to(p.device))
            g = ck["grads"].get(n)
            p.grad = g.to(p.device) if g is not None else None
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        st = ck["st"]
        log(f"RESUMED step={st['step']} micro={st['micro']} evals={sorted(st['evals'])}")
    last_save = time.time()

    def save_state(force=False):
        nonlocal last_save
        if not force and time.time() - last_save < args.save_every_sec:
            return
        t = time.time()
        atomic_save({"params": {n: p.detach().cpu() for n, p in trainable.items()},
                     "grads": {n: p.grad.detach().cpu() for n, p in trainable.items() if p.grad is not None},
                     "opt": opt.state_dict(), "sched": sched.state_dict(), "st": st, "plan": plan}, state_path)
        write_json(hist_path, {"plan": plan, "evals": st["evals"], "train_log": st["train_log"],
                               "diverged": st["diverged"], "done": st["done"]})
        last_save = time.time()
        log(f"saved state ({time.time()-t:.0f}s) step={st['step']} micro={st['micro']}")

    def run_eval(step):
        key = str(step)
        if key in st["evals"]:
            return
        model.eval()
        part = st["eval_partial"] if st["eval_partial"] and st["eval_partial"]["step"] == step else \
            {"step": step, "done": {}}
        st["eval_partial"] = part
        t0 = time.time()
        with torch.no_grad():
            for i, ex in enumerate(val_ex):
                if ex["id"] in part["done"]:
                    continue
                l, n = target_loss_sum(model, ex["ids"], ex["prompt_len"])
                lv = l.item()
                if n <= 0 or not math.isfinite(lv):
                    log("EVAL NONFINITE", ex["id"], "loss", lv, "n", n,
                        "seq", len(ex["ids"]), "prompt", int(ex["prompt_len"]))
                part["done"][ex["id"]] = [lv, n]
                save_state()
        tot = sum(v[0] for v in part["done"].values())
        ntok = sum(v[1] for v in part["done"].values())
        per_ex = [v[0] / v[1] for v in part["done"].values()]
        il = st["interval_loss"]
        rec = {"step": step, "epoch": round(step / spe, 3), "val_loss": tot / ntok,
               "val_loss_mean_per_example": sum(per_ex) / len(per_ex),
               "train_loss_interval": (il[0] / il[1]) if il[1] else None,
               "eval_sec": round(time.time() - t0), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        st["interval_loss"] = [0.0, 0]
        st["evals"][key] = rec
        st["eval_partial"] = None
        log("EVAL", json.dumps(rec))
        if step > 0:
            ckpt = run_dir / f"checkpoint-{step:04d}"
            save_adapter(model, ckpt)
        # divergence: NaN, or training loss rising at two consecutive evals
        tl = [st["evals"][str(s)]["train_loss_interval"] for s in eval_points if str(s) in st["evals"] and s > 0]
        if not math.isfinite(rec["val_loss"]) or (tl and tl[-1] is not None and not math.isfinite(tl[-1])):
            st["diverged"] = f"non-finite loss at step {step}"
        elif len(tl) >= 3 and tl[-1] > tl[-2] > tl[-3]:
            st["diverged"] = f"training loss rose at two consecutive evals (steps up to {step}): {tl[-3:]}"
        model.train()
        save_state(force=True)

    run_eval(0)
    model.train()
    while st["step"] < total_steps and not st["diverged"]:
        step = st["step"]
        epoch, in_epoch = divmod(step, spe)
        idx = order(epoch)[in_epoch * args.eff_batch:(in_epoch + 1) * args.eff_batch]
        batch_tokens = sum(train_ex[i]["target_len"] for i in idx)
        while st["micro"] < len(idx):
            ex = train_ex[idx[st["micro"]]]
            t = time.time()
            loss, n = target_loss_sum(model, ex["ids"], ex["prompt_len"])
            lv = loss.item()
            if not math.isfinite(lv):
                st["diverged"] = f"non-finite training loss at step {step} micro {st['micro']} ({ex['id']})"
                break
            (loss / batch_tokens).backward()
            st["interval_loss"][0] += lv
            st["interval_loss"][1] += n
            st["micro"] += 1
            st["train_log"].append({"step": step, "micro": st["micro"], "seq_len": len(ex["ids"]),
                                    "loss": lv / n, "sec": round(time.time() - t, 1)})
            if st["micro"] <= 2 and step == 0:
                log("MEM", gpu_mem())
            log(f"step {step+1}/{total_steps} micro {st['micro']}/{len(idx)} len={len(ex['ids'])} "
                f"loss={lv/n:.4f} {time.time()-t:.0f}s")
            if st["micro"] < len(idx):
                save_state()
        if st["diverged"]:
            break
        gn = torch.nn.utils.clip_grad_norm_(list(trainable.values()), args.max_grad_norm).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        st["step"] += 1
        st["micro"] = 0
        step_losses = [r["loss"] for r in st["train_log"] if r["step"] == step]
        log(f"OPTSTEP {st['step']}/{total_steps} lr={sched.get_last_lr()[0]:.3e} grad_norm={gn:.3f} "
            f"mean_loss={sum(step_losses)/len(step_losses):.4f}")
        st["train_log"].append({"step": step, "optstep_done": True, "grad_norm": gn,
                                "lr": sched.get_last_lr()[0]})
        if not math.isfinite(gn):
            st["diverged"] = f"non-finite grad norm at step {st['step']}"
            break
        save_state(force=True)
        if st["step"] in eval_steps:
            run_eval(st["step"])
    if st["diverged"]:
        log("DIVERGED", st["diverged"])
        save_state(force=True)
        (run_dir / "DIVERGED").write_text(st["diverged"] + "\n")
        return
    st["done"] = True
    save_state(force=True)
    (run_dir / "DONE").write_text("done\n")
    log("TRAINING DONE")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(fromfile_prefix_chars="@")
    ap.convert_arg_line_to_args = lambda line: [] if not line.strip() or line.lstrip().startswith("#") else [line.strip()]
    ap.add_argument("--mode", choices=["memtest", "train", "verify", "fp32check"], required=True)
    ap.add_argument("--data-dir", default="data", help="relative paths resolve under DATA_ROOT")
    ap.add_argument("--keep-ids", default="",
                    help="examples_used.json; train and val are restricted to those ids")
    ap.add_argument("--run-dir", required=True, help="relative paths resolve under OUT_ROOT")
    ap.add_argument("--model-path", default=os.environ.get("SFT_MODEL_PATH", ""))
    ap.add_argument("--quant", choices=["bf16", "4bit"], default="bf16")
    ap.add_argument("--lengths", default="98304,65536,49152,32768")
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--gpu0-layers", type=int, default=12)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eff-batch", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every-sec", type=int, default=300)
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--global-group-ckpt", action="store_true")
    args = ap.parse_args()
    args.data_dir = str(DATA_ROOT / args.data_dir)
    if args.keep_ids:
        args.keep_ids = str(DATA_ROOT / args.keep_ids)
    args.run_dir = str(OUT_ROOT / args.run_dir)
    GLOBAL_GROUP_CKPT["on"] = args.global_group_ckpt

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    patch_model_code()
    if args.mode == "fp32check":
        checked = fp32_eager_check()
        write_json(run_dir / "fp32_eager_check.json", checked)
        shared = DATA_ROOT / "runs/fp32_eager_check.json"
        write_json(shared, checked)
        log("FP32 CHECK", "PASS" if checked["passed"] else "FAIL", json.dumps(checked))
        if not checked["passed"]:
            raise SystemExit(1)
        return
    # The 4-bit loss/grad check (exact loss, ~0.5% grad gap) is the gate for
    # 4-bit training. The fp32 check is reported separately and does not block.
    n_gpus = torch.cuda.device_count()
    log("GPUs:", [torch.cuda.get_device_name(i) for i in range(n_gpus)], "args:", vars(args))
    kernels = probe_standard_kernels()
    torch.cuda.empty_cache()
    model, tok, per = build_model(n_gpus, args.gpu0_layers, args.quant,
                                   args.model_path or str(snapshot_dir(WRITER_MODEL, weights=True)))

    cache = Path(args.data_dir) / "_tok"
    cache.mkdir(exist_ok=True)
    train_ex = tokenize_split(Path(args.data_dir) / "train.jsonl", tok, cache / "train.pt")
    val_ex = tokenize_split(Path(args.data_dir) / "val.jsonl", tok, cache / "val.pt")
    caps = [16384, 24576, 32768, 49152, 65536, 98304]
    write_json(run_dir / "length_table.json", {"train": length_table(train_ex, caps), "val": length_table(val_ex, caps)})
    ex0 = train_ex[0]
    log("TAIL", repr(tok.decode(ex0["ids"][ex0["prompt_len"] - 12:ex0["prompt_len"] + 20].tolist())))
    log("END", repr(tok.decode(ex0["ids"][-8:].tolist())))

    set_attn(inner(model), "sdpa")
    model = add_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, args.quant)
    # prepare_model_for_kbit_training casts every non-4bit parameter to fp32.
    # layer_scalar and the token embedding then promote the residual to fp32:
    # one fp32 hidden state per layer is 1 GiB at 64k. Norms stay fp32; their
    # output is cast back to the activation dtype.
    m = inner(model)
    lm = m.model.language_model
    for p in (lm.embed_tokens.weight, m.lm_head.weight):
        if p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    for layer in lm.layers:
        layer.layer_scalar.data = layer.layer_scalar.data.to(torch.bfloat16)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable params {n_tr/1e6:.1f}M; embed {lm.embed_tokens.weight.dtype}, "
        f"layer_scalar {lm.layers[0].layer_scalar.dtype}")
    flash_note = "skipped: no memory-efficient kernel for global head_dim 512"
    if kernels.get("global_d512_FLASH_ATTENTION") == "ok":
        try:
            set_attn(inner(model), "flash_attention_2")
            with torch.no_grad():
                target_loss_sum(model, train_ex[0]["ids"][:512], 400)
            flash_note = "ok"
        except Exception as e:
            flash_note = f"{type(e).__name__}: {str(e).splitlines()[0][:240]}"
            set_attn(inner(model), "sdpa")
    kernels["flash_attention_2_forward"] = flash_note
    log("FLASH", flash_note)
    standard_ok = flash_note == "ok" or (
        kernels.get("global_d512_EFFICIENT_ATTENTION") == "ok"
        and kernels.get("sliding_d256_EFFICIENT_ATTENTION") == "ok")
    write_json(run_dir / "attn_probe.json", {**kernels, "use_standard": standard_ok})
    prior = DATA_ROOT / "runs/grad_check.json"
    if standard_ok:
        pass
    elif args.mode != "verify" and prior.exists():
        log("GRADCHECK reused", prior.read_text().strip())
    else:
        torch.cuda.empty_cache()
        checked = grad_check(model, train_ex, seq_len=2048)
        write_json(run_dir / "grad_check.json", checked)
        write_json(prior, checked)
    if args.mode == "verify":
        log("VERIFY DONE")
        return
    if not args.skip_selftest and not (run_dir / "selftest.json").exists():
        write_json(run_dir / "selftest.json", selftest(model, train_ex))
    set_attn(inner(model), "flash_attention_2" if flash_note == "ok" else "blockwise_sdpa")
    enable_gc(model)
    model.train()

    if args.mode == "memtest":
        lengths = [int(x) for x in args.lengths.split(",")]
        memtest(model, train_ex, lengths, run_dir / "memtest.json")
        return

    def has_target(e):
        n, p = len(e["ids"]), int(e["prompt_len"])
        return 0 < p < n and int(e["target_len"]) > 0

    def drop_empty(rows, split):
        kept = [e for e in rows if has_target(e)]
        gone = [e["id"] for e in rows if not has_target(e)]
        if gone:
            log(f"DROP EMPTY TARGET {split}", len(gone), gone[:8])
        return kept

    keep_tr = drop_empty([e for e in train_ex if len(e["ids"]) <= args.max_len], "train")
    keep_va = drop_empty([e for e in val_ex if len(e["ids"]) <= args.max_len], "val")
    if args.keep_ids:
        spec = json.loads(Path(args.keep_ids).read_text())
        def take(rows, wanted, split):
            by_id = {e["id"]: e for e in rows}
            missing = [i for i in wanted if i not in by_id]
            if missing:
                raise SystemExit(f"keep-ids: {len(missing)} {split} ids are not in this data under max_len")
            return [by_id[i] for i in wanted]
        keep_tr = take(keep_tr, spec["train"]["ids"], "train")
        keep_va = take(keep_va, spec["val"]["ids"], "val")
        log(f"KEEP IDS train {len(keep_tr)} val {len(keep_va)} from {args.keep_ids}")
    write_json(run_dir / "examples_used.json", {
        "max_len": args.max_len,
        "train": {"used": len(keep_tr), "dropped": len(train_ex) - len(keep_tr),
                  "used_by_bucket": {b: sum(e["bucket"] == b for e in keep_tr) for b in sorted({e["bucket"] for e in train_ex})},
                  "total_by_bucket": {b: sum(e["bucket"] == b for e in train_ex) for b in sorted({e["bucket"] for e in train_ex})},
                  "ids": [e["id"] for e in keep_tr]},
        "val": {"used": len(keep_va), "dropped": len(val_ex) - len(keep_va),
                "used_by_bucket": {b: sum(e["bucket"] == b for e in keep_va) for b in sorted({e["bucket"] for e in val_ex})},
                "total_by_bucket": {b: sum(e["bucket"] == b for e in val_ex) for b in sorted({e["bucket"] for e in val_ex})},
                "ids": [e["id"] for e in keep_va]},
    })
    train(args, model, keep_tr, keep_va, run_dir)


if __name__ == "__main__":
    main()
