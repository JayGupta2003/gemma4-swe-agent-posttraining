"""Local tests for gemma4_memfix on tiny Gemma 4 models, run in the venv that mirrors Kaggle (transformers 5.13.1, peft 0.19.1).

1. exactness: chunked attention + chunked MLP == stock eager attention (outputs and gradients), incl. sliding window across chunk edges
2. memory: reproduces the G1-run-4 failure mode (global head_dim 512 -> O(T^2) memory with stock sdpa) and shows the fix bounds it
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import gemma4_memfix as MF  # noqa: E402
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"


def make(hidden, layers, heads, kv, hd, ghd, gkv, inter, window, vocab, attn_impl):
    t = Gemma4TextConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers, num_attention_heads=heads,
                         num_key_value_heads=kv, head_dim=hd, global_head_dim=ghd, num_global_key_value_heads=gkv, attention_k_eq_v=True,
                         sliding_window=window, layer_types=["sliding_attention"] * (layers - 1) + ["full_attention"], final_logit_softcapping=30.0,
                         hidden_size_per_layer_input=0, num_kv_shared_layers=0, use_double_wide_mlp=False, max_position_embeddings=65536)
    cfg = Gemma4Config(text_config=t.to_dict(), vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                       num_key_value_heads=2, head_dim=16, global_head_dim=16), audio_config=None)
    return Gemma4ForConditionalGeneration._from_config(cfg, attn_implementation=attn_impl)


# ---------------------------------------------------------------- 1. exactness
MF.install(chunk_global=40, chunk_sliding=48, mlp_chunk=64)
torch.manual_seed(0)
ref = make(128, 6, 8, 4, 32, 64, 2, 256, 32, 1000, "eager").to(dev).float().train()
fix = make(128, 6, 8, 4, 32, 64, 2, 256, 32, 1000, MF.ATTN_NAME).to(dev).float().train()
fix.load_state_dict(ref.state_dict())
ids = torch.randint(0, 1000, (1, 200), device=dev)
MF.uninstall_mlp_patch()
o_ref = ref.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
MF.install(chunk_global=40, chunk_sliding=48, mlp_chunk=64)
o_fix = fix.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
d = (o_ref - o_fix).abs().max().item()
print(f"forward max abs diff (fp32): {d:.2e}")
assert d < 1e-4, d
# NOTE: not mean(h^2): the final output is RMS-normalised so that loss is ~constant and its gradients are ~1e-9 noise
w = torch.randn_like(o_ref)
(o_ref * w).sum().backward()
(o_fix * w).sum().backward()
worst = 0.0
diffs = []
for (n, p1), (_, p2) in zip(ref.named_parameters(), fix.named_parameters()):
    if p1.grad is None:
        assert p2.grad is None, n
        continue
    rel = ((p1.grad - p2.grad).abs().max() / (p1.grad.abs().max() + 1e-12)).item()
    diffs.append((rel, n, p1.grad.abs().max().item()))
    worst = max(worst, rel)
print(f"gradient worst relative diff: {worst:.2e}")
for r_, n_, g_ in sorted(diffs, reverse=True)[:6]: print(f"   {r_:.2e}  grad_absmax={g_:.2e}  {n_}")
assert worst < 1e-3, worst
print("EXACTNESS OK (outputs + gradients, window crossing chunk edges)")

# ---------------------------------------------------------------- 2. memory (needs a GPU)
if dev == "cuda":
    T = 8192
    MF.install(chunk_global=256, chunk_sliding=1024, mlp_chunk=2048)

    def peak(attn_impl, mlp_chunked):
        if mlp_chunked:
            MF.install(256, 1024, 2048)
        else:
            MF.uninstall_mlp_patch()
        m = make(512, 6, 32, 16, 64, 512, 4, 2048, 1024, 2000, attn_impl).to(dev).bfloat16().train()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        x = torch.randint(0, 2000, (1, T), device=dev)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.time()
        try:
            h = m.model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
            h.float().pow(2).mean().backward()
            torch.cuda.synchronize()
            r = (torch.cuda.max_memory_allocated() - base) / 2**30
            res = f"{r:.2f} GiB above weights, {time.time() - t0:.1f}s"
        except torch.cuda.OutOfMemoryError:
            res = "OOM"
        del m
        torch.cuda.empty_cache()
        return res

    print(f"T={T}, global head_dim 512, 32 heads (G1-run-4 geometry, small hidden):")
    print("  stock sdpa + stock MLP     :", peak("sdpa", False))
    print("  chunked attn + chunked MLP :", peak(MF.ATTN_NAME, True))
print("MEMFIX TESTS DONE")
