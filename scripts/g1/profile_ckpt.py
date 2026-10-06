"""Is gradient checkpointing actually effective on Gemma 4 (nested language_model)? Compare peak memory with/without it."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import gemma4_memfix as MF
from test_memfix import make  # noqa

def run(T, ckpt, how):
    MF.uninstall_mlp_patch()
    m = make(512, 12, 32, 16, 64, 512, 4, 2048, 1024, 2000, "sdpa").to("cuda").bfloat16().train()
    if ckpt:
        if how == "hf": m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    gc_flags = {n: getattr(mod, "gradient_checkpointing", None) for n, mod in m.named_modules() if hasattr(mod, "gradient_checkpointing")}
    layer = m.model.language_model.layers[0]
    x = torch.randint(0, 2000, (1, T), device="cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    h = m.model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
    h.float().pow(2).mean().backward(); torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    out = (peak, type(layer).__mro__[1].__name__, {k: v for k, v in gc_flags.items() if k.count('.') <= 2})
    del m, h; torch.cuda.empty_cache(); return out
print("no ckpt :", run(4096, False, "hf"))
print("hf ckpt :", run(4096, True, "hf"))
