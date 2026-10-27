"""RPO (DPO + NLL on the chosen trajectory) LoRA training of Gemma 4 on trajectory preference pairs (scripts/g15/build_pairs.py). Same memory machinery as scripts/g3/train_g3.py.

Objective per pair (y_w chosen, y_l rejected; log p(y) = sum of the token log-probabilities over the loss spans = the model-generated tokens):
    m    = beta * [(log p_theta(y_w) - log p_ref(y_w)) - (log p_theta(y_l) - log p_ref(y_l))]       (RPO_NORM=1 divides every log-ratio by the number of loss tokens of its sequence)
    loss = -log sigmoid(m)  +  alpha * NLL(y_w) / n_loss_tokens(y_w)
Reference = the same bf16 model with the LoRA adapter disabled (log-probs cached once per pair; they never change).
Two-pass trick (only one autograd graph is alive, peak memory stays at the SFT level): the gradient is  -w [grad log p(y_w) - grad log p(y_l)] - alpha/n_w grad log p(y_w)  with
w = beta * sigmoid(-m) (times 1/n per side with RPO_NORM=1). So per pair: no-grad forwards of the policy give m and w; then one forward+backward per trajectory with the scalar coefficient
c * log p(y) (c detached). tests/test_train_rpo.py checks that this equals the plain autograd gradient of the full loss.

Input rows (pairs.jsonl): {"task", "kind", "chosen": {"input_ids", "loss_spans", "n_loss_tokens"}, "rejected": {...}}.
Environment: RPO_MODEL, RPO_DATA, RPO_OUT (required); RPO_BETA (0.1), RPO_ALPHA (1.0), RPO_NORM (0), RPO_ACC (2 pairs per optimizer step), RPO_LR (2e-5), RPO_EPOCHS (2), RPO_WARMUP (2),
RPO_SEED (0), RPO_HF_GC (1 = HF gradient checkpointing), RPO_TINY (1 = tiny random model for tests), RPO_TINY_PLE (1 = tiny model with per-layer inputs), RPO_MAXMEM (JSON per-GPU limits),
RPO_CKPT_EVERY (10 optimizer steps), RPO_KINDS (comma list, default all), RPO_MAX_PAIRS (0 = all).
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

TARGET_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
_INSTALLED = False


def log(*a) -> None:
    print("[rpo]", *a, flush=True)


def load_model(path=None, tiny=False, ple=False, maxmem=None):
    global _INSTALLED
    cls = transformers.Gemma4ForConditionalGeneration
    if not _INSTALLED:
        MF.install()
        MF.install_layers(chunk_sliding=1024, chunk_global=128)
        _INSTALLED = True
    if tiny:
        from transformers import Gemma4Config, Gemma4TextConfig
        types = (["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"] if ple else ["sliding_attention"] * 5 + ["full_attention"])
        t = Gemma4TextConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=32,
                             layer_types=types, final_logit_softcapping=30.0, hidden_size_per_layer_input=(16 if ple else 0), vocab_size_per_layer_input=1000,
                             num_kv_shared_layers=(2 if ple else 0), use_double_wide_mlp=False, max_position_embeddings=65536)
        cfg = Gemma4Config(text_config=t.to_dict(), vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                                                       head_dim=16, global_head_dim=16), audio_config=None)
        return cls._from_config(cfg, attn_implementation="sdpa").to(torch.bfloat16).to("cuda:0" if torch.cuda.is_available() else "cpu")
    kw = dict(dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    if maxmem:
        kw["max_memory"] = {int(k): v for k, v in maxmem.items()}
    return cls.from_pretrained(path, **kw)


def lora_wrap(model, hf_gc=False, rank=16, alpha=32):
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    if hf_gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=TARGET_RE, task_type="CAUSAL_LM"))
    assert any(n.endswith(".lora_A.default") for n, _ in pm.named_modules()), "TARGET_RE matched no module"
    return pm


class RPO:
    def __init__(self, pm, beta=0.1, alpha=1.0, norm=False, chunk=512):
        self.pm, self.beta, self.alpha, self.norm, self.CH = pm, beta, alpha, norm, chunk
        model = pm.base_model.model
        self.inner, self.lm_head = model.model, model.lm_head
        self.dev = model.get_input_embeddings().weight.device
        self.softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

    def _chunk_logp(self, h, tgt):
        logits = self.lm_head(h).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return -F.cross_entropy(logits, tgt, reduction="sum")

    def logp(self, seq, grad):
        """Sum of the token log-probabilities over the loss spans of `seq` (a tensor with grad when grad=True, otherwise computed under no_grad)."""
        from torch.utils.checkpoint import checkpoint
        ids = seq["input_ids"]
        pos = sorted({j - 1 for s, e in seq["loss_spans"] for j in range(max(s, 1), min(e, len(ids)))})
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx:
            x = torch.tensor([ids], device=self.dev)
            hs = self.inner(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).last_hidden_state[0]
            hd = self.lm_head.weight.device
            h_sel = hs[torch.tensor(pos, device=hs.device)].to(hd)
            tgt = torch.tensor([ids[p + 1] for p in pos], device=hd)
            total = None
            for a in range(0, len(pos), self.CH):
                part = (checkpoint(self._chunk_logp, h_sel[a:a + self.CH], tgt[a:a + self.CH], use_reentrant=False) if grad else self._chunk_logp(h_sel[a:a + self.CH], tgt[a:a + self.CH]))
                total = part if total is None else total + part
        return total

    def ref_logp(self, seq):
        with self.pm.disable_adapter():
            return float(self.logp(seq, False))

    def _scales(self, pair):
        nw, nl = pair["chosen"]["n_loss_tokens"], pair["rejected"]["n_loss_tokens"]
        return (1.0 / nw, 1.0 / nl, nw) if self.norm else (1.0, 1.0, nw)

    def margin(self, lw, ll, rw, rl, sw, sl):
        return self.beta * ((lw - rw) * sw - (ll - rl) * sl)

    def pair_backward(self, pair, ref, group_size):
        """Two-pass gradient of the RPO loss of one pair, accumulated into the LoRA .grad (divided by group_size). ref = (log p_ref(y_w), log p_ref(y_l))."""
        sw, sl, nw = self._scales(pair)
        lw = float(self.logp(pair["chosen"], False))
        ll = float(self.logp(pair["rejected"], False))
        m = self.margin(lw, ll, ref[0], ref[1], sw, sl)
        sig = 1.0 / (1.0 + math.exp(m)) if m > -700 else 1.0              # sigmoid(-m)
        w = self.beta * sig
        c_w = -(w * sw + self.alpha / nw) / group_size                   # coefficient of log p(y_w) in the loss
        c_l = (w * sl) / group_size                                       # coefficient of log p(y_l)
        for seq, c in ((pair["chosen"], c_w), (pair["rejected"], c_l)):
            (self.logp(seq, True) * c).backward()
        return {"dpo_loss": math.log1p(math.exp(-m)) if m > -700 else -m, "margin": m, "acc": float(m > 0), "nll_chosen": -lw / nw, "logp_chosen": lw, "logp_rejected": ll,
                "d_chosen": lw - ref[0], "d_rejected": ll - ref[1]}

    def direct_loss(self, pair, ref):
        """The same loss through plain autograd (a single graph over both trajectories): for tests only (needs the memory of two graphs)."""
        sw, sl, nw = self._scales(pair)
        lw, ll = self.logp(pair["chosen"], True), self.logp(pair["rejected"], True)
        m = self.beta * ((lw - ref[0]) * sw - (ll - ref[1]) * sl)
        return F.softplus(-m) + self.alpha * (-lw / nw)


def main() -> None:
    E = os.environ.get
    MODEL, DATA, OUT = E("RPO_MODEL", "x"), E("RPO_DATA"), Path(E("RPO_OUT"))
    BETA, ALPHA, NORM = float(E("RPO_BETA", "0.1")), float(E("RPO_ALPHA", "1.0")), E("RPO_NORM") == "1"
    ACC, LR, EPOCHS, WARMUP, SEED = int(E("RPO_ACC", "2")), float(E("RPO_LR", "2e-5")), int(E("RPO_EPOCHS", "2")), int(E("RPO_WARMUP", "2")), int(E("RPO_SEED", "0"))
    TINY, TINY_PLE, HF_GC = E("RPO_TINY") == "1", E("RPO_TINY_PLE") == "1", E("RPO_HF_GC") == "1"
    MAXMEM, CKPT_EVERY, MAX_PAIRS = json.loads(E("RPO_MAXMEM", "null")), int(E("RPO_CKPT_EVERY", "10")), int(E("RPO_MAX_PAIRS", "0"))
    KINDS = set(E("RPO_KINDS", "").split(",")) - {""}
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"stage": "start", "beta": BETA, "alpha": ALPHA, "norm": NORM, "acc": ACC, "lr": LR, "epochs": EPOCHS, "warmup": WARMUP, "seed": SEED, "tiny": TINY,
           "torch": torch.__version__, "transformers": transformers.__version__, "n_gpus": torch.cuda.device_count()}

    def save():
        (OUT / "train_metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    pairs = [json.loads(line) for line in open(DATA, encoding="utf-8")]
    if KINDS:
        pairs = [p for p in pairs if p["kind"] in KINDS]
    if MAX_PAIRS:
        pairs = pairs[:MAX_PAIRS]
    res["data"] = {"pairs": len(pairs), "kinds": {k: sum(1 for p in pairs if p["kind"] == k) for k in sorted({p["kind"] for p in pairs})}, "tasks": len({p["task"] for p in pairs}),
                   "tokens": sum(len(p[s]["input_ids"]) for p in pairs for s in ("chosen", "rejected")), "max_len": max(len(p[s]["input_ids"]) for p in pairs for s in ("chosen", "rejected"))}
    log("data", res["data"])
    save()
    t0 = time.perf_counter()
    model = load_model(MODEL, tiny=TINY, ple=TINY_PLE, maxmem=MAXMEM)
    res["load_seconds"] = round(time.perf_counter() - t0, 1)
    log("loaded in", res["load_seconds"], "s")
    pm = lora_wrap(model, hf_gc=HF_GC)
    params = [p for p in pm.parameters() if p.requires_grad]
    res["trainable_params_m"] = round(sum(p.numel() for p in params) / 1e6, 1)
    pm.train()
    rpo = RPO(pm, BETA, ALPHA, NORM)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    ndev = torch.cuda.device_count()
    base_alloc = [torch.cuda.memory_allocated(i) for i in range(ndev)]

    # reference log-probs, once per pair side (adapter disabled); cached on disk so a restart does not repeat them
    ref_path = OUT / "ref_logps.json"
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    t_ref = time.perf_counter()
    for i, p in enumerate(pairs):
        for side in ("chosen", "rejected"):
            key = f"{i}:{p['task']}:{side}"
            if key not in ref:
                ref[key] = rpo.ref_logp(p[side])
        if i % 10 == 9:
            ref_path.write_text(json.dumps(ref))
            log(f"reference log-probs {i + 1}/{len(pairs)} ({time.perf_counter() - t_ref:.0f}s)")
    ref_path.write_text(json.dumps(ref))
    res["ref_seconds"] = round(time.perf_counter() - t_ref, 1)
    save()

    def lr_at(step, total):
        if step < WARMUP:
            return LR * (step + 1) / WARMUP
        prog = (step - WARMUP) / max(1, total - WARMUP)
        return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    order = list(range(len(pairs)))
    groups = []
    for ep in range(EPOCHS):
        random.Random(SEED + ep).shuffle(order)
        groups += [order[i:i + ACC] for i in range(0, len(order), ACC)]
    total_steps = len(groups)
    res.update({"stage": "training", "optimizer_steps_planned": total_steps, "steps": [], "oom_events": []})
    for i in range(ndev):
        torch.cuda.reset_peak_memory_stats(i)
    t_train0, tok_done = time.perf_counter(), 0
    for step, grp in enumerate(groups):
        lr = lr_at(step, total_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        t1, ms, n_tok, failed = time.perf_counter(), [], 0, False
        for i in grp:
            p = pairs[i]
            try:
                ms.append(rpo.pair_backward(p, (ref[f"{i}:{p['task']}:chosen"], ref[f"{i}:{p['task']}:rejected"]), len(grp)))
                n_tok += len(p["chosen"]["input_ids"]) + len(p["rejected"]["input_ids"])
            except torch.cuda.OutOfMemoryError:
                res["oom_events"].append({"step": step, "task": p["task"], "tokens": len(p["chosen"]["input_ids"]) + len(p["rejected"]["input_ids"])})
                failed = True
            if failed:
                break
        if failed:
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            log("OOM at step", step, "- group skipped")
            save()
            continue
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        tok_done += n_tok
        mean = lambda k: round(sum(m[k] for m in ms) / len(ms), 4)  # noqa: E731
        rec = {"step": step, "lr": round(lr, 8), "grad_norm": round(gn, 4), "seconds": round(dt, 1), "tokens": n_tok, "tasks": [pairs[i]["task"] for i in grp],
               **{k: mean(k) for k in ("dpo_loss", "margin", "acc", "nll_chosen", "d_chosen", "d_rejected")}}
        res["steps"].append(rec)
        res["tokens_per_second"] = round(tok_done / (time.perf_counter() - t_train0), 1)
        res["peak_gib_per_gpu"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(ndev)]
        save()
        log(f"step {step + 1}/{total_steps} lr {lr:.2e} dpo {rec['dpo_loss']} margin {rec['margin']} acc {rec['acc']} nll {rec['nll_chosen']} dW {rec['d_chosen']} dL {rec['d_rejected']} "
            f"gnorm {rec['grad_norm']} {dt:.0f}s | {res['tokens_per_second']} tok/s")
        if (step + 1) % CKPT_EVERY == 0:
            pm.save_pretrained(str(OUT / "adapter_ckpt"))
    res["leak_gib_at_end"] = round(max(((torch.cuda.memory_allocated(i) - b) for i, b in enumerate(base_alloc)), default=0) / 2**30, 2)
    res["stage"] = "saving"
    pm.save_pretrained(str(OUT / "adapter_full"))
    res["stage"] = "done"
    res["train_seconds"] = round(time.perf_counter() - t_train0, 1)
    save()
    log("training finished: steps", len(res["steps"]), "of", total_steps, "| adapter saved")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # noqa: BLE001
        outp = Path(os.environ.get("RPO_OUT", "."))
        try:
            m = json.loads((outp / "train_metrics.json").read_text())
        except Exception:
            m = {}
        m.update({"stage": "FAILED", "error": f"{type(e).__name__}: {str(e)[:500]}", "traceback": traceback.format_exc()[-3000:]})
        (outp / "train_metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
        traceback.print_exc()
        sys.exit(1)
