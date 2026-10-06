"""Exactness of the sequence-chunked decoder layer vs stock (tiny Gemma 4, fp32, window crossing chunk edges, k_eq_v global layers)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import gemma4_memfix as MF  # noqa: E402
from test_memfix import make  # noqa: E402  (runs its own exactness tests on import)

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
ref = make(128, 6, 8, 4, 32, 64, 2, 256, 32, 1000, "sdpa").to(dev).float().train()
fix = make(128, 6, 8, 4, 32, 64, 2, 256, 32, 1000, "sdpa").to(dev).float().train()
fix.load_state_dict(ref.state_dict())
ids = torch.randint(0, 1000, (1, 300), device=dev)
w = torch.randn(1, 300, 128, device=dev, generator=torch.Generator(device=dev).manual_seed(3))

MF.uninstall_layers()
o_ref = ref.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
(o_ref * w).sum().backward()

MF.install_layers(chunk_sliding=48, chunk_global=40, kv_chunk=64, min_len=64)
MF.calls["layer"] = 0
o_fix = fix.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
(o_fix * w).sum().backward()
MF.uninstall_layers()
print("chunked layer calls:", MF.calls["layer"], "(expect 6)")
assert MF.calls["layer"] == 6
d = (o_ref - o_fix).abs().max().item()
worst = 0.0
for (n, p1), (_, p2) in zip(ref.named_parameters(), fix.named_parameters()):
    if p1.grad is None:
        assert p2.grad is None, n
        continue
    worst = max(worst, ((p1.grad - p2.grad).abs().max() / (p1.grad.abs().max() + 1e-12)).item())
print(f"output max abs diff {d:.2e} | worst relative grad diff {worst:.2e}")
assert d < 1e-4 and worst < 1e-3
print("LAYER-CHUNK EXACTNESS OK")
