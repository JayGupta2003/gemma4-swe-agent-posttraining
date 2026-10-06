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


# =====================================================================================================================
# Sequence-chunked decoder layer (G1d). Everything in a Gemma 4 layer is token-wise EXCEPT attention itself, so a layer can be
# processed in chunks of query positions with each chunk checkpointed. The big fp32 temporaries of RMSNorm / q-norm / RoPE / MLP
# then scale with the chunk instead of the whole sequence. Sliding layers recompute the K/V of their window per chunk (cheap);
# global layers compute K/V once (checkpointed in token chunks; only ~[T, 2048] per tensor to keep).
# Valid for the 31B config: no per-layer inputs, no MoE, no KV sharing, batch 1, no cache. Anything else falls back to stock.
# =====================================================================================================================
_lcfg = {"chunk_sliding": 1024, "chunk_global": 256, "kv_chunk": 2048, "min_len": 2048}


def _kv(layer, h, cos, sin):
    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
    a = layer.self_attn
    x = layer.input_layernorm(h)
    shape = (*x.shape[:-1], -1, a.head_dim)
    k_raw = a.k_proj(x).view(shape)
    v_raw = a.v_proj(x).view(shape) if a.v_proj is not None else k_raw   # k_eq_v layers: value = raw (pre-norm) key projection
    k = apply_rotary_pos_emb(a.k_norm(k_raw), cos, sin, unsqueeze_dim=2).transpose(1, 2)
    v = a.v_norm(v_raw).transpose(1, 2)
    return k, v


def _tail(layer, h_c, cos_c, sin_c, K, V, q0, k0):
    """The whole layer for the query tokens h_c (positions q0..q0+c): attention against K/V (positions k0..) + MLP, exactly as stock."""
    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
    a = layer.self_attn
    c = h_c.shape[1]
    x = layer.input_layernorm(h_c)
    q = apply_rotary_pos_emb(a.q_norm(a.q_proj(x).view(1, c, -1, a.head_dim)), cos_c, sin_c, unsqueeze_dim=2).transpose(1, 2)  # [1,H,c,hd]
    H, Hkv = q.shape[1], K.shape[1]
    o = _block(q.reshape(1, Hkv, H // Hkv, c, q.shape[-1]), K, V, q0, k0, a.scaling, a.sliding_window)
    o = a.o_proj(o.reshape(1, H, c, -1).transpose(1, 2).reshape(1, c, -1))
    h = h_c + layer.post_attention_layernorm(o)
    h = h + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(h)))
    return h * layer.layer_scalar


def _chunk_sliding(layer, h_kv, cos_kv, sin_kv, q0, k0):
    off = q0 - k0
    K, V = _kv(layer, h_kv, cos_kv, sin_kv)
    return _tail(layer, h_kv[:, off:], cos_kv[:, off:], sin_kv[:, off:], K, V, q0, k0)


def _chunk_global(layer, h_c, cos_c, sin_c, K, V, q0):
    b = q0 + h_c.shape[1]
    return _tail(layer, h_c, cos_c, sin_c, K[:, :, :b], V[:, :, :b], q0, 0)


def _layer_forward(self, hidden_states, per_layer_input=None, shared_kv_states=None, position_embeddings=None, attention_mask=None,
                   position_ids=None, past_key_values=None, **kwargs):
    B, T, _ = hidden_states.shape
    a = self.self_attn
    if (B != 1 or T < _lcfg["min_len"] or past_key_values is not None or not torch.is_grad_enabled() or self.enable_moe_block
            or self.hidden_size_per_layer_input or a.is_kv_shared_layer):
        return type(self)._g1_orig_forward(self, hidden_states, per_layer_input, shared_kv_states, position_embeddings, attention_mask,
                                           position_ids, past_key_values, **kwargs)
    calls["layer"] += 1
    cos, sin = position_embeddings
    h = hidden_states
    outs = []
    if a.sliding_window:
        ch, w = _lcfg["chunk_sliding"], a.sliding_window
        for q0 in range(0, T, ch):
            b, k0 = min(q0 + ch, T), max(0, q0 - w + 1)
            outs.append(checkpoint(_chunk_sliding, self, h[:, k0:b], cos[:, k0:b], sin[:, k0:b], q0, k0, use_reentrant=False))
    else:
        ks, vs = [], []
        for t0 in range(0, T, _lcfg["kv_chunk"]):
            t1 = min(t0 + _lcfg["kv_chunk"], T)
            k, v = checkpoint(_kv, self, h[:, t0:t1], cos[:, t0:t1], sin[:, t0:t1], use_reentrant=False)
            ks.append(k)
            vs.append(v)
        K, V = torch.cat(ks, dim=2), torch.cat(vs, dim=2)
        ch = _lcfg["chunk_global"]
        for q0 in range(0, T, ch):
            b = min(q0 + ch, T)
            outs.append(checkpoint(_chunk_global, self, h[:, q0:b], cos[:, q0:b], sin[:, q0:b], K, V, q0, use_reentrant=False))
    return torch.cat(outs, dim=1)


calls["layer"] = 0


def install_layers(chunk_sliding: int = 1024, chunk_global: int = 256, kv_chunk: int = 2048, min_len: int = 2048) -> None:
    """Replace Gemma4TextDecoderLayer.forward by the sequence-chunked version (use with HF gradient checkpointing DISABLED:
    the chunks are the checkpoints)."""
    from transformers.models.gemma4 import modeling_gemma4 as mg
    _lcfg.update(chunk_sliding=chunk_sliding, chunk_global=chunk_global, kv_chunk=kv_chunk, min_len=min_len)
    L = mg.Gemma4TextDecoderLayer
    if not getattr(L, "_g1_layer_chunked", False):
        L._g1_orig_forward = L.forward
        L.forward = _layer_forward
        L._g1_layer_chunked = True


def uninstall_layers() -> None:
    from transformers.models.gemma4 import modeling_gemma4 as mg
    L = mg.Gemma4TextDecoderLayer
    if getattr(L, "_g1_layer_chunked", False):
        L.forward = L._g1_orig_forward
        L._g1_layer_chunked = False
