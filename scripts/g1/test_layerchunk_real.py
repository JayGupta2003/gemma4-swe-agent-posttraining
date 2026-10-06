"""Exactness of the chunked layer at the REAL 31B layer dimensions + real rope config (2 layers: sliding + global), fp32, same weights,
stock layer vs chunked layer (switched on the same model), outputs and LoRA gradients."""
import sys
from pathlib import Path
import torch
sys.modules["torchao"] = None
sys.path.insert(0, str(Path(__file__).parent))
import gemma4_memfix as MF
from peft import LoraConfig, get_peft_model
from profile_real_dims import RE, build

m = build(["sliding_attention", "full_attention"]).float()
m.enable_input_require_grads()
pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=RE, task_type="CAUSAL_LM")); pm.train()
for n, p in pm.named_parameters():
    if "lora_B" in n:
        torch.nn.init.normal_(p, std=0.02)
T = 2600
ids = torch.randint(0, 1000, (1, T), device="cuda")
w = torch.randn(1, T, 5376, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))

def run():
    pm.zero_grad(set_to_none=True)
    h = m.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    (h * w).sum().backward()
    return h.detach().clone(), {n: p.grad.detach().clone() for n, p in pm.named_parameters() if p.grad is not None}

MF.uninstall_layers(); MF.uninstall_mlp_patch()
h0, g0 = run()
MF.calls["layer"] = 0
MF.install_layers(chunk_sliding=512, chunk_global=128, kv_chunk=1024, min_len=2048)
h1, g1 = run()
print("chunked layer calls:", MF.calls["layer"], "(expect 2)")
assert MF.calls["layer"] == 2
d = (h0 - h1).abs().max().item()
worst = max(((g0[k] - g1[k]).abs().max() / (g0[k].abs().max() + 1e-12)).item() for k in g0)
print(f"real dims, real rope: output max abs diff {d:.2e} (output scale {h0.abs().max().item():.1f}) | worst relative LoRA-grad diff {worst:.2e} over {len(g0)} tensors")
assert worst < 1e-3, worst
print("REAL-DIM LAYER-CHUNK EXACTNESS OK")
