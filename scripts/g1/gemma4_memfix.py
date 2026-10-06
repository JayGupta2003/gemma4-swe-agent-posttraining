"""Memory fixes so a 31B Gemma 4 can be LoRA-trained on 4x L4 (23 GB each) at 16k+ tokens.

Root cause found in G1 run 4: full-attention layers have head_dim 512 (config `global_head_dim`), which PyTorch's fused SDPA kernels
do not support, so they fall back to the math kernel whose memory grows with T^2 ([32 heads, T, T] fp32). Even 5k tokens ran out of the
~7 GiB per GPU left after loading 15.6 GiB of bf16 weights.

Fixes (all exact, no approximation of the model):
  1. `chunked_sdpa` attention: queries processed in blocks, each block checkpointed, causal/sliding-window masks built on the fly from
     positions (no T x T mask tensor). Memory ~ O(block x T) instead of O(T^2); sliding layers only look at their window.
  2. Sequence-chunked MLP: the MLP is token-wise, so running it in chunks under checkpoint is exactly equivalent and removes the
     [T, 21504] x several activations that dominate at 16k.

Assumes batch size 1 and no padding (training on single unpadded sequences), which is how train_g1.py uses it.
"""

import os

import torch
from torch.utils.checkpoint import checkpoint

ATTN_NAME = "chunked_sdpa"
_cfg = {"chunk_global": 256, "chunk_sliding": 1024, "mlp_chunk": 2048}
calls = {"attn": 0, "mlp": 0}  # incremented per call; lets a log prove which implementation actually ran


def _block(qc, kc, vc, q0: int, k0: int, scale: float, window):
    """qc [B,Hkv,G,c,D], kc/vc [B,Hkv,L,D] -> [B,Hkv,G,c,D]; exact causal (+ sliding-window) softmax attention in fp32 scores."""
    s = torch.einsum("bhgcd,bhld->bhgcl", qc, kc).float() * scale
    qi = torch.arange(q0, q0 + qc.shape[3], device=qc.device)[:, None]
    kj = torch.arange(k0, k0 + kc.shape[2], device=qc.device)[None, :]
    allowed = kj <= qi
    if window:
        allowed = allowed & (kj > qi - window)
    s = s.masked_fill(~allowed, float("-inf"))
    p = torch.softmax(s, dim=-1).to(vc.dtype)
    return torch.einsum("bhgcl,bhld->bhgcd", p, vc)


def chunked_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, **kwargs):
    """Drop-in for transformers attention functions: query [B,H,T,D], key/value [B,Hkv,T,D] -> ([B,T,H,D], None)."""
    calls["attn"] += 1
    B, H, T, D = query.shape
    Hkv = key.shape[1]
    G = H // Hkv
    scale = 1.0 if scaling is None else scaling
    ch = _cfg["chunk_sliding"] if sliding_window else _cfg["chunk_global"]
    q = query.reshape(B, Hkv, G, T, D)
    outs = []
    for a in range(0, T, ch):
        b = min(a + ch, T)
        k0 = max(0, a - sliding_window + 1) if sliding_window else 0
        args = (q[:, :, :, a:b], key[:, :, k0:b], value[:, :, k0:b], a, k0, scale, sliding_window)
        outs.append(checkpoint(_block, *args, use_reentrant=False) if torch.is_grad_enabled() else _block(*args))
    out = torch.cat(outs, dim=3).reshape(B, H, T, D)
    return out.transpose(1, 2).contiguous(), None


def _no_mask(*args, **kwargs):
    return None  # chunked_attention builds its own causal / sliding-window masks


def install(chunk_global: int = 256, chunk_sliding: int = 1024, mlp_chunk: int = 2048) -> None:
    """Register the attention + mask functions and patch the Gemma 4 MLP. Call before building/loading the model."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    from transformers.models.gemma4 import modeling_gemma4 as mg

    _cfg.update(chunk_global=chunk_global, chunk_sliding=chunk_sliding, mlp_chunk=mlp_chunk)
    AttentionInterface.register(ATTN_NAME, chunked_attention)
    AttentionMaskInterface.register(ATTN_NAME, _no_mask)

    if not getattr(mg.Gemma4TextMLP, "_g1_chunked", False):
        orig = mg.Gemma4TextMLP.forward

        def forward(self, x):
            n = x.shape[-2]
            mc = _cfg["mlp_chunk"]
            if n <= mc:
                return orig(self, x)
            calls["mlp"] += 1
            parts = [checkpoint(orig, self, x[..., a:a + mc, :], use_reentrant=False) if torch.is_grad_enabled() else orig(self, x[..., a:a + mc, :])
                     for a in range(0, n, mc)]
            return torch.cat(parts, dim=-2)

        mg.Gemma4TextMLP.forward = forward
        mg.Gemma4TextMLP._g1_chunked = True
        mg.Gemma4TextMLP._g1_orig_forward = orig


def uninstall_mlp_patch() -> None:
    """For tests: restore the stock MLP forward."""
    from transformers.models.gemma4 import modeling_gemma4 as mg
    if getattr(mg.Gemma4TextMLP, "_g1_chunked", False):
        mg.Gemma4TextMLP.forward = mg.Gemma4TextMLP._g1_orig_forward
        mg.Gemma4TextMLP._g1_chunked = False
