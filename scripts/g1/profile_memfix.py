"""Where does the remaining memory go? Scaling of peak memory with T for chunked vs stock attention (local GPU, tiny hidden size)."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import gemma4_memfix as MF
from test_memfix import make  # noqa  (importing runs its checks; cheap)

dev = "cuda"
def run(T, impl, mlp_chunk, ckpt=True):
    (MF.install(256, 1024, 2048) if mlp_chunk else MF.uninstall_mlp_patch())
    m = make(512, 6, 32, 16, 64, 512, 4, 2048, 1024, 2000, impl).to(dev).bfloat16().train()
    if ckpt: m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    x = torch.randint(0, 2000, (1, T), device=dev)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    h = m.model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
    fwd = (torch.cuda.max_memory_allocated() - base) / 2**30
    h.float().pow(2).mean().backward(); torch.cuda.synchronize()
    tot = (torch.cuda.max_memory_allocated() - base) / 2**30
    del m, h; torch.cuda.empty_cache()
    return f"fwd {fwd:.2f} GiB, fwd+bwd {tot:.2f} GiB"
for T in (2048, 4096, 8192):
    print(T, "chunked:", run(T, MF.ATTN_NAME, True), "| stock sdpa:", run(T, "sdpa", False))
