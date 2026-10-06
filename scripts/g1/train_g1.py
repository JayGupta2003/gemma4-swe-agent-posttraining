"""G1 training spike: LoRA SFT of Gemma 4 31B (bf16 QAT-unquantized checkpoint) across the GPUs of one machine.

Measures what decides feasibility: tokens/s, peak memory per GPU, long-sequence headroom; saves a PEFT adapter in the exact key
format the scorer expects. The SAME file is tested locally on a tiny random Gemma 4 (G1_TINY=1) and run on Kaggle on the real model.

Environment variables (all strings):
  G1_MODEL   checkpoint dir (real run) or a dir holding tokenizer + chat template (tiny run)
  G1_DATA    jsonl of {"messages": [...]} (OpenAI-style, dict tool arguments)
  G1_OUT     output dir (metrics json, adapter_full/)
  G1_TINY    "1" = build a tiny random Gemma 4 instead of loading G1_MODEL weights
  G1_MAXLEN  max tokens per training sequence (default 16384; median trajectory is ~15.5k tokens, the fixed system+task prompt alone ~4.5k; steps down x0.75 on OOM)
  G1_RENDER  "flat" (default; plain-text tool protocol) or "native" (Gemma 4 chat-template tool format; NOT prefix-stable, see STRATEGY.md G1 notes)
  G1_MICRO   number of micro-steps (sequences) to run (default 16); G1_ACC grad-accumulation (default 4)
  G1_MAXMEM  JSON dict gpu->limit for device_map auto, e.g. {"0":"15GiB","1":"19GiB",...}
  G1_TRAIN_SECONDS  stop training after this many seconds (default 1500)
  G1_ATTN    "chunked" (default) or "stock"; G1_DIAG lengths for the stock-vs-chunked memory diagnostic (default 2048,3072,4096)
  G1_LADDER  ascending lengths for the memory/throughput ladder (default 4096..32768); training then runs at the largest OK length <= G1_MAXLEN
"""

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Kaggle's image ships torchao 0.10.0, which peft 0.19.1 rejects with an ImportError while attaching LoRA layers (run 3 failure).
# We do not use torchao, so hide it: importlib.util.find_spec() returns None for a sys.modules entry of None.
sys.modules["torchao"] = None
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma4_memfix as MF

MODEL = os.environ["G1_MODEL"]
DATA = os.environ["G1_DATA"]
OUT = Path(os.environ["G1_OUT"])
TINY = os.environ.get("G1_TINY") == "1"
MAXLEN = int(os.environ.get("G1_MAXLEN", "16384"))
RENDER = os.environ.get("G1_RENDER", "flat")
MICRO = int(os.environ.get("G1_MICRO", "16"))
ACC = int(os.environ.get("G1_ACC", "4"))
MAXMEM = json.loads(os.environ.get("G1_MAXMEM", "null"))
TRAIN_SECONDS = float(os.environ.get("G1_TRAIN_SECONDS", "1500"))
ATTN = os.environ.get("G1_ATTN", "chunked")  # chunked (memory-bounded, scripts/g1/gemma4_memfix.py) | stock (sdpa)
LADDER = [int(x) for x in os.environ.get("G1_LADDER", "4096,8192,12288,16384,20480,24576,32768").split(",") if x]
DIAG = [int(x) for x in os.environ.get("G1_DIAG", "2048,3072,4096").split(",") if x]  # stock-vs-chunked memory comparison lengths
TARGET_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"

OUT.mkdir(parents=True, exist_ok=True)
res: dict = {"stage": "start", "tiny": TINY, "maxlen": MAXLEN, "torch": torch.__version__, "transformers": transformers.__version__,
             "n_gpus": torch.cuda.device_count()}


def save() -> None:
    (OUT / "train_metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def log(*a) -> None:
    print("[g1]", *a, flush=True)


def gpu_peaks() -> list[float]:
    return [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]


# ---------------------------------------------------------------- data: chat-template rendering with assistant-only labels
def flatten_for_template(msgs: list[dict]) -> list[dict]:
    """Fallback if the chat template rejects tool_calls/tool roles: plain-text tool protocol."""
    out = []
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = "\n".join(json.dumps({"name": c["function"]["name"], "arguments": c["function"]["arguments"]}) for c in m["tool_calls"])
            out.append({"role": "assistant", "content": (m.get("content") or "") + "\n<tool_call>\n" + calls + "\n</tool_call>"})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": "<tool_response>\n" + (m.get("content") or "") + "\n</tool_response>"})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def render_example(tok, msgs: list[dict]):
    def tmpl(m, gen=False):
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=gen)

    full = tmpl(msgs)
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        # NOT add_generation_prompt: the 31B template's generation prompt appends an empty thought channel that is never rendered
        # in history, so it can never be a prefix of the rendered turn. Labels therefore also cover the `<|turn>model` header (a few tokens).
        p, q = tmpl(msgs[:i]), tmpl(msgs[: i + 1])
        if not (q.startswith(p) and full.startswith(q)):
            return None, "template not prefix-stable"
        spans.append((len(p), len(q)))
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc["input_ids"], enc["offset_mapping"]
    labels = [-100] * len(ids)
    for k, (a, _b) in enumerate(off):
        if any(s <= a < e for s, e in spans):
            labels[k] = ids[k]
    return (ids, labels), "ok"


def build_examples(tok) -> list[tuple[list[int], list[int]]]:
    rows = [json.loads(line) for line in open(DATA, encoding="utf-8")]
    exs, why = [], {}
    mode_counts = {"native": 0, "flattened": 0}
    for r in rows:
        msgs = r["messages"]
        ex, w, mode = None, "", None
        if RENDER == "native":
            try:
                ex, w = render_example(tok, msgs)
                mode = "native"
            except Exception as e:  # noqa: BLE001
                ex, w = None, f"native render failed: {type(e).__name__}"
        if ex is None:
            try:
                ex, w = render_example(tok, flatten_for_template(msgs))
                mode = "flattened"
            except Exception as e:  # noqa: BLE001
                ex, w = None, f"flattened render failed: {type(e).__name__}: {str(e)[:80]}"
        if ex is not None and sum(1 for x in ex[1][:MAXLEN] if x != -100) < 200:  # must keep >= 200 label tokens inside the window
            ex, w = None, "too few label tokens inside MAXLEN"
        if ex is None:
            why[w] = why.get(w, 0) + 1
            continue
        mode_counts[mode] += 1
        ids, labels = ex
        exs.append((ids[:32768], labels[:32768]))
    res["data"] = {"rows": len(rows), "examples": len(exs), "render_modes": mode_counts, "dropped": why,
                   "tokens_median": sorted(len(i) for i, _ in exs)[len(exs) // 2] if exs else 0,
                   "label_fraction": round(sum(sum(1 for x in l if x != -100) for _, l in exs) / max(1, sum(len(i) for i, _ in exs)), 3)}
    log("data", res["data"])
    return exs


# ---------------------------------------------------------------- model
def load_model():
    cls = getattr(transformers, "Gemma4ForConditionalGeneration", None) or transformers.AutoModelForImageTextToText
    if ATTN == "chunked":
        MF.install()
    if TINY:
        from transformers import Gemma4Config, Gemma4TextConfig
        t = Gemma4TextConfig(vocab_size=262144, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, sliding_window=32,
                             layer_types=["sliding_attention"] * 5 + ["full_attention"], final_logit_softcapping=30.0,
                             hidden_size_per_layer_input=0, num_kv_shared_layers=0, use_double_wide_mlp=False, max_position_embeddings=65536)
        cfg = Gemma4Config(text_config=t.to_dict(), vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=2, head_dim=16, global_head_dim=16), audio_config=None)
        return cls._from_config(cfg, attn_implementation=MF.ATTN_NAME if ATTN == "chunked" else "sdpa").to(torch.bfloat16).to("cuda:0" if torch.cuda.is_available() else "cpu")
    kw = dict(dtype=torch.bfloat16, device_map="auto", attn_implementation=MF.ATTN_NAME if ATTN == "chunked" else "sdpa")
    if MAXMEM:
        kw["max_memory"] = {int(k): v for k, v in MAXMEM.items()}
    return cls.from_pretrained(MODEL, **kw)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    exs = build_examples(tok)
    save()
    assert exs, "no usable training examples"

    t0 = time.perf_counter()
    model = load_model()
    res["load_seconds"] = round(time.perf_counter() - t0, 1)
    res["device_map_gpus"] = sorted({str(v) for v in getattr(model, "hf_device_map", {}).values()}) if hasattr(model, "hf_device_map") else ["single"]
    log("loaded in", res["load_seconds"], "s; mem after load (GiB):", [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())])
    res["mem_after_load_gib"] = [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]
    save()

    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=TARGET_RE, task_type="CAUSAL_LM"))
    n_tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    res["trainable_params_m"] = round(n_tr / 1e6, 1)
    names = [n for n, _ in pm.named_modules() if n.endswith(".lora_A.default")]
    res["lora_modules"] = len(names)
    log("trainable params (M):", res["trainable_params_m"], "| lora modules:", len(names), "| e.g.", names[:2])
    assert names, "TARGET_RE matched no module"
    pm.train()
    dev = model.get_input_embeddings().weight.device
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    extra_kw: dict = {}

    from torch.utils.checkpoint import checkpoint
    inner, lm_head = model.model, model.lm_head
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    res["final_logit_softcapping"] = softcap
    CH = 512  # positions per lm_head chunk: [512, 262144] fp32 logits = 0.5 GiB, recomputed in backward (checkpoint)

    def chunk_loss(h, tgt):
        logits = lm_head(h).float()
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        return F.cross_entropy(logits, tgt, reduction="sum")

    def peak_max() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return round(max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30, 2)

    def fwd_bwd(ids: list[int], labels: list[int]):
        """One micro-step. Returns (loss, n_tokens, peak GiB right after the forward pass)."""
        pos = [t for t in range(len(ids) - 1) if labels[t + 1] != -100]
        if not pos:
            return None, 0, 0.0
        x = torch.tensor([ids], device=dev)
        for attempt in range(2):
            try:
                hs = inner(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, **extra_kw).last_hidden_state[0]
                break
            except (ValueError, TypeError) as e:
                if attempt == 0 and "token_type" in str(e):
                    extra_kw["token_type_ids"] = torch.zeros_like(x)
                    if "mm_token_type_ids" in str(e):
                        extra_kw["mm_token_type_ids"] = torch.zeros_like(x)
                    log("retrying with token_type_ids")
                    continue
                raise
        fwd_peak = peak_max()
        hd = lm_head.weight.device
        h_sel = hs[torch.tensor(pos, device=hs.device)].to(hd)
        tgt = torch.tensor([ids[t + 1] for t in pos], device=hd)
        loss = None
        for a in range(0, len(pos), CH):
            part = checkpoint(chunk_loss, h_sel[a:a + CH], tgt[a:a + CH], use_reentrant=False)
            loss = part if loss is None else loss + part
        loss = loss / len(pos)
        (loss / ACC).backward()
        return float(loss.detach()), len(ids), fwd_peak

    def set_impl(name: str) -> None:
        try:
            model.set_attn_implementation(name)
        except Exception:  # noqa: BLE001
            for c in (model.config, model.config.get_text_config()):
                c._attn_implementation = name

    def seq_of_len(L: int):
        ids, labels = [], []
        for e_ids, e_lab in sorted(exs, key=lambda e: -len(e[0])):  # longest examples first; concatenated if one is too short
            ids += e_ids
            labels += e_lab
            if len(ids) >= L:
                break
        return ids[:L], labels[:L]

    def measure(L: int, tag: str) -> dict:
        ids, labels = seq_of_len(L)
        if not any(x != -100 for x in labels[1:]):  # short windows lie inside the prompt (no labels): put labels on the last quarter
            q = L - L // 4
            labels = [-100] * q + ids[q:]
        MF.calls["attn"] = MF.calls["mlp"] = 0
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        base = [torch.cuda.memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())]
        t1 = time.perf_counter()
        try:
            _loss, n, fwd_peak = fwd_bwd(ids, labels)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t1
            r = {"tag": tag, "len": L, "ok": True, "seconds": round(dt, 1), "tokens_per_second": round(n / dt, 1),
                 "chunked_attn_calls": MF.calls["attn"], "chunked_mlp_calls": MF.calls["mlp"],
                 "fwd_peak_gib_max_gpu": fwd_peak, "peak_gib_per_gpu": gpu_peaks(),
                 "above_weights_gib_per_gpu": [round(p - b, 2) for p, b in zip(gpu_peaks(), base)]}
        except torch.cuda.OutOfMemoryError:
            r = {"tag": tag, "len": L, "ok": False, "error": "OOM", "peak_gib_per_gpu": gpu_peaks(),
                 "chunked_attn_calls": MF.calls["attn"], "chunked_mlp_calls": MF.calls["mlp"]}
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        log("measure", r)
        return r

    # ---- diagnostic: stock sdpa vs chunked attention at cheap lengths (which one explains the run-4 OOM at ~5k tokens?)
    res["stage"] = "diagnostic"
    res["diag"] = []
    if DIAG:
        if ATTN == "chunked":
            for impl, name in (("stock", "sdpa"), ("chunked", MF.ATTN_NAME)):
                set_impl(name)
                if impl == "stock":
                    MF.uninstall_mlp_patch()
                else:
                    MF.install()
                for L in DIAG:
                    res["diag"].append(measure(L, impl))
                    save()
                    if not res["diag"][-1]["ok"]:
                        break
            MF.install()
            set_impl(MF.ATTN_NAME)
        else:
            for L in DIAG:
                res["diag"].append(measure(L, "stock"))
                save()

    # ---- memory/throughput ladder with the configured attention
    res["stage"] = "ladder"
    res["probes"] = []
    best = 0
    for L in LADDER:
        r = measure(L, ATTN)
        res["probes"].append(r)
        save()
        if not r["ok"]:
            break
        best = L
    train_len = min(best, MAXLEN)
    res["largest_ok_len"] = best
    res["train_len"] = train_len
    log("largest OK length", best, "-> training at", train_len)

    # ---- training steps at the largest feasible length (skipped if the prompt alone does not fit: no label tokens)
    res["stage"] = "training"
    steps, tok_total, t_train0 = [], 0, None
    t_wall0 = time.perf_counter()
    usable = [(i[:train_len], l[:train_len]) for i, l in exs if sum(1 for x in l[:train_len] if x != -100) >= 200]
    res["train_examples_usable_at_len"] = len(usable)
    if train_len < 6000 or not usable:
        res["training_skipped"] = f"largest feasible length {train_len} leaves no usable examples (prompt alone is ~4.5k tokens)"
        log(res["training_skipped"])
    for k in range(MICRO if usable and train_len >= 6000 else 0):
        ids, labels = usable[k % len(usable)]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        try:
            loss, n_tok, _fp = fwd_bwd(ids, labels)
        except torch.cuda.OutOfMemoryError:
            res.setdefault("oom_events", []).append({"micro": k, "tokens": len(ids)})
            log("OOM at micro", k, "tokens", len(ids))
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            save()
            continue
        if (k + 1) % ACC == 0:
            torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        if k == 0:
            t_train0 = time.perf_counter()  # exclude warm-up from throughput
        else:
            tok_total += n_tok
        steps.append({"micro": k, "tokens": n_tok, "seconds": round(dt, 2), "loss": None if loss is None else round(loss, 4)})
        el = (time.perf_counter() - t_train0) if t_train0 else 0
        res.update({"steps": steps, "tokens_per_second": round(tok_total / el, 1) if el > 0 else None, "peak_gib_per_gpu": gpu_peaks()})
        save()
        log(f"micro {k} tokens {n_tok} {dt:.1f}s loss {loss} | tok/s so far {res['tokens_per_second']} | peak GiB {gpu_peaks()}")
        if time.perf_counter() - t_wall0 > TRAIN_SECONDS:
            res["stopped_by_time_budget"] = True
            break

    res["stage"] = "saving"
    pm.save_pretrained(str(OUT / "adapter_full"))
    res["stage"] = "done"
    save()
    log("done; adapter saved to", OUT / "adapter_full")


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
