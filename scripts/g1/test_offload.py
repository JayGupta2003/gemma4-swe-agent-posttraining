"""Local test: reentrant checkpointing + save_on_cpu (activation offload) gives the same LoRA gradients as plain non-reentrant checkpointing."""
import contextlib
import sys as _s
_s.modules["torchao"] = None  # same workaround as train_g1.py (torchao 0.10.0 vs peft 0.19.1)
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import gemma4_memfix as MF  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from test_memfix import make  # noqa: E402  (runs its own exactness checks on import)

dev = "cuda" if torch.cuda.is_available() else "cpu"
RE = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
MF.install(256, 1024, 64)


def grads(offload: bool, seed=0):
    torch.manual_seed(seed)
    m = make(128, 6, 8, 4, 32, 64, 2, 256, 32, 1000, "sdpa").to(dev).float()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": offload})
    m.enable_input_require_grads()
    pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=RE, task_type="CAUSAL_LM"))
    for n, p in pm.named_parameters():       # make LoRA-B non-zero so gradients w.r.t. A are non-trivial too
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    pm.train()
    ids = torch.randint(0, 1000, (1, 300), device=dev)
    w = torch.randn(1, 300, 128, device=dev, generator=torch.Generator(device=dev).manual_seed(1))
    with (torch.autograd.graph.save_on_cpu(pin_memory=torch.cuda.is_available()) if offload else contextlib.nullcontext()):
        h = m.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    (h * w).sum().backward()
    return {n: p.grad.clone() for n, p in pm.named_parameters() if p.requires_grad and p.grad is not None}


a, b = grads(False), grads(True)
assert a.keys() == b.keys() and len(a) > 0
worst = max(((a[k] - b[k]).abs().max() / (a[k].abs().max() + 1e-12)).item() for k in a)
print(f"{len(a)} LoRA tensors with grads; worst relative diff offload vs plain: {worst:.2e}")
assert worst < 1e-4, worst
print("OFFLOAD EQUIVALENCE OK")
