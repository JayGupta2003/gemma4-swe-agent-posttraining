"""G3: LoRA SFT of Gemma 4 31B (bf16 QAT-unquantized checkpoint, 4 x L4) on token-exact sequences captured in G2.

Input jsonl rows: {"input_ids": [...], "loss_spans": [[start, end), ...], "n_loss_tokens": int, "task": str, ...} (scripts/g2/build_sft_dataset.py).
The loss is the mean negative log-likelihood over the tokens inside the spans (token k is predicted from position k-1). Gradient accumulation groups
are normalised by their total number of loss tokens, so every loss token has equal weight.
Memory: sequence-chunked decoder layers (scripts/g1/gemma4_memfix.py, exact, verified at the real 31B dimensions) + chunked lm_head loss: trained and probed
on the L4x4 up to 18,432 tokens in G1d. The environment fixes found in G1 are kept: torchao hidden from peft, checkpoint staged to local disk by the notebook.

Environment variables: G3_MODEL, G3_DATA, G3_OUT (required); G3_TINY=1 (tiny random model for local tests); G3_ACC (default 2); G3_LR (2e-4);
G3_EPOCHS (1); G3_HF_GC (1 = HF gradient checkpointing, for E4B/E2B); G3_TINY_PLE (1 = tiny test model with per-layer inputs); G3_WARMUP (3 optimizer steps); G3_SEED (0); G3_MAXMEM (JSON per-GPU limits); G3_MAX_SEQS (use only the first N sequences, tests); G3_CKPT_EVERY (10).
"""

import contextlib
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.modules["torchao"] = None   # Kaggle's torchao 0.10.0 is rejected by peft 0.19.1 and we do not use it
import torch
import torch.nn.functional as F
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "g1"))
import gemma4_memfix as MF  # noqa: E402

MODEL, DATA, OUT = os.environ["G3_MODEL"], os.environ["G3_DATA"], Path(os.environ["G3_OUT"])
TINY = os.environ.get("G3_TINY") == "1"
ACC = int(os.environ.get("G3_ACC", "2"))
LR = float(os.environ.get("G3_LR", "2e-4"))
EPOCHS = int(os.environ.get("G3_EPOCHS", "1"))
WARMUP = int(os.environ.get("G3_WARMUP", "3"))
SEED = int(os.environ.get("G3_SEED", "0"))
MAXMEM = json.loads(os.environ.get("G3_MAXMEM", "null"))
MAX_SEQS = int(os.environ.get("G3_MAX_SEQS", "0"))
CKPT_EVERY = int(os.environ.get("G3_CKPT_EVERY", "10"))
TEST_MEMFRAC = os.environ.get("G3_TEST_MEMFRAC")
HF_GC = os.environ.get("G3_HF_GC") == "1"       # HF gradient checkpointing: needed for models whose layers fall back to the stock forward (E4B/E2B: per-layer inputs, shared KV)
TINY_PLE = os.environ.get("G3_TINY_PLE") == "1"  # tiny test model WITH per-layer inputs and shared-KV layers (E4B-like)
TARGET_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
OUT.mkdir(parents=True, exist_ok=True)
res: dict = {"stage": "start", "tiny": TINY, "acc": ACC, "lr": LR, "epochs": EPOCHS, "warmup": WARMUP, "seed": SEED,
             "torch": torch.__version__, "transformers": transformers.__version__, "n_gpus": torch.cuda.device_count()}


def save() -> None:
    (OUT / "train_metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def log(*a) -> None:
    print("[g3]", *a, flush=True)


def gpu_peaks() -> list[float]:
    return [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]


def load_model():
    cls = transformers.Gemma4ForConditionalGeneration
    MF.install()
    MF.install_layers(chunk_sliding=1024, chunk_global=128)
    if TINY:
        from transformers import Gemma4Config, Gemma4TextConfig
        t = Gemma4TextConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=16, sliding_window=32, layer_types=(["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"] if TINY_PLE
                                     else ["sliding_attention"] * 5 + ["full_attention"]), final_logit_softcapping=30.0,
                             hidden_size_per_layer_input=(16 if TINY_PLE else 0), vocab_size_per_layer_input=1000, num_kv_shared_layers=(2 if TINY_PLE else 0),
                             use_double_wide_mlp=False, max_position_embeddings=65536)
        cfg = Gemma4Config(text_config=t.to_dict(), vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                           num_key_value_heads=2, head_dim=16, global_head_dim=16), audio_config=None)
        return cls._from_config(cfg, attn_implementation="sdpa").to(torch.bfloat16).to("cuda:0" if torch.cuda.is_available() else "cpu")
    kw = dict(dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    if MAXMEM:
        kw["max_memory"] = {int(k): v for k, v in MAXMEM.items()}
    return cls.from_pretrained(MODEL, **kw)


def main() -> None:
    seqs = [json.loads(line) for line in open(DATA, encoding="utf-8")]
    if MAX_SEQS:
        seqs = seqs[:MAX_SEQS]
    res["data"] = {"sequences": len(seqs), "tokens": sum(len(s["input_ids"]) for s in seqs), "loss_tokens": sum(s["n_loss_tokens"] for s in seqs),
                   "max_len": max(len(s["input_ids"]) for s in seqs), "tasks": len({s["task"] for s in seqs})}
    log("data", res["data"])
    save()

    t0 = time.perf_counter()
    model = load_model()
    res["load_seconds"] = round(time.perf_counter() - t0, 1)
    res["mem_after_load_gib"] = [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]
    log("loaded in", res["load_seconds"], "s; mem after load (GiB):", res["mem_after_load_gib"])
    save()

    if TEST_MEMFRAC and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(TEST_MEMFRAC))
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    if HF_GC:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log("HF gradient checkpointing enabled")
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=TARGET_RE, task_type="CAUSAL_LM"))
    params = [p for p in pm.parameters() if p.requires_grad]
    res["trainable_params_m"] = round(sum(p.numel() for p in params) / 1e6, 1)
    assert any(n.endswith(".lora_A.default") for n, _ in pm.named_modules()), "TARGET_RE matched no module"
    pm.train()
    dev = model.get_input_embeddings().weight.device
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    inner, lm_head = model.model, model.lm_head
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    from torch.utils.checkpoint import checkpoint
    CH = 512
    base_alloc = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]

    def chunk_loss(h, tgt):
        logits = lm_head(h).float()
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        return F.cross_entropy(logits, tgt, reduction="sum")

    def fwd_bwd(seq: dict, denom: float) -> tuple[float, int]:
        ids = seq["input_ids"]
        pos = sorted({j - 1 for s, e in seq["loss_spans"] for j in range(max(s, 1), min(e, len(ids)))})
        x = torch.tensor([ids], device=dev)
        hs = inner(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).last_hidden_state[0]
        hd = lm_head.weight.device
        h_sel = hs[torch.tensor(pos, device=hs.device)].to(hd)
        tgt = torch.tensor([ids[p + 1] for p in pos], device=hd)
        total = None
        for a in range(0, len(pos), CH):
            part = checkpoint(chunk_loss, h_sel[a:a + CH], tgt[a:a + CH], use_reentrant=False)
            total = part if total is None else total + part
        (total / denom).backward()
        return float(total.detach()), len(pos)

    def cleanup() -> None:
        opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def lr_at(step: int, total: int) -> float:       # linear warmup, then cosine decay to 10% of the peak
        if step < WARMUP:
            return LR * (step + 1) / WARMUP
        prog = (step - WARMUP) / max(1, total - WARMUP)
        return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    order = list(range(len(seqs)))
    groups = []
    for ep in range(EPOCHS):
        random.Random(SEED + ep).shuffle(order)
        groups += [order[i:i + ACC] for i in range(0, len(order), ACC)]
    total_steps = len(groups)
    res.update({"stage": "training", "optimizer_steps_planned": total_steps, "steps": [], "oom_events": []})
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    t_train0 = time.perf_counter()
    tok_done = 0
    for step, grp in enumerate(groups):
        lr = lr_at(step, total_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        denom = float(sum(seqs[i]["n_loss_tokens"] for i in grp))
        loss_sum, n_loss, n_tok, t1 = 0.0, 0, 0, time.perf_counter()
        failed = False
        for i in grp:
            oom = False
            try:
                ls, nl = fwd_bwd(seqs[i], denom)
                loss_sum += ls
                n_loss += nl
                n_tok += len(seqs[i]["input_ids"])
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:     # handled outside the except block so the failed step's tensors can be freed
                res["oom_events"].append({"step": step, "task": seqs[i]["task"], "tokens": len(seqs[i]["input_ids"])})
                log("OOM on", seqs[i]["task"], len(seqs[i]["input_ids"]), "tokens; skipping this group")
                failed = True
                break
        if failed:
            cleanup()
            save()
            continue
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        tok_done += n_tok
        rec = {"step": step, "lr": round(lr, 7), "loss": round(loss_sum / max(1, n_loss), 4), "grad_norm": round(gn, 4), "tokens": n_tok, "loss_tokens": n_loss,
               "seconds": round(dt, 1), "tasks": [seqs[i]["task"] for i in grp]}
        res["steps"].append(rec)
        res["tokens_per_second"] = round(tok_done / (time.perf_counter() - t_train0), 1)
        res["peak_gib_per_gpu"] = gpu_peaks()
        save()
        log(f"step {step + 1}/{total_steps} lr {lr:.2e} loss {rec['loss']} gnorm {rec['grad_norm']} tokens {n_tok} {dt:.0f}s | {res['tokens_per_second']} tok/s | peak {gpu_peaks()}")
        if (step + 1) % CKPT_EVERY == 0:
            pm.save_pretrained(str(OUT / "adapter_ckpt"))
            log("checkpoint saved")
    leak = max((torch.cuda.memory_allocated(i) - b) for i, b in enumerate(base_alloc)) / 2**30 if base_alloc else 0.0
    res["leak_gib_at_end"] = round(leak, 2)
    res["stage"] = "saving"
    pm.save_pretrained(str(OUT / "adapter_full"))
    res["stage"] = "done"
    res["train_seconds"] = round(time.perf_counter() - t_train0, 1)
    save()
    log("training finished: steps", len(res["steps"]), "of", total_steps, "| tok/s", res.get("tokens_per_second"), "| adapter saved")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # noqa: BLE001
        res["stage"] = "FAILED"
        res["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        res["traceback"] = traceback.format_exc()[-3000:]
        save()
        traceback.print_exc()
        sys.exit(1)
