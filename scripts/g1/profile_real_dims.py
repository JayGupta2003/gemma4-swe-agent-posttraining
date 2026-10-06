"""Local memory profile with the REAL 31B layer geometry (2 layers: 1 sliding + 1 global, LoRA r=16 on attn+MLP, vocab shrunk).
Compares stock layer (HF gradient checkpointing) vs the sequence-chunked layer. Peak = max allocated above the weights."""
import sys, time
from pathlib import Path
import torch
sys.modules["torchao"] = None
sys.path.insert(0, str(Path(__file__).parent))
import gemma4_memfix as MF
from peft import LoraConfig, get_peft_model
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

RE = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
rope = {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}

def build(layers):
    t = Gemma4TextConfig(vocab_size=1000, hidden_size=5376, intermediate_size=21504, num_hidden_layers=len(layers), num_attention_heads=32,
                         num_key_value_heads=16, head_dim=256, global_head_dim=512, num_global_key_value_heads=4, attention_k_eq_v=True,
                         sliding_window=1024, layer_types=layers, final_logit_softcapping=30.0, hidden_size_per_layer_input=0,
                         num_kv_shared_layers=0, use_double_wide_mlp=False, max_position_embeddings=65536, rope_parameters=rope)
    cfg = Gemma4Config(text_config=t.to_dict(), vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                       num_key_value_heads=2, head_dim=16, global_head_dim=16), audio_config=None)
    return Gemma4ForConditionalGeneration._from_config(cfg, attn_implementation="sdpa").to("cuda").bfloat16()

def run(mode, T, layers):
    MF.uninstall_layers(); MF.uninstall_mlp_patch()
    m = build(layers)
    if mode == "stock":
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        import os; MF.install_layers(chunk_sliding=int(os.environ.get("CS", "1024")), chunk_global=int(os.environ.get("CG", "256")))  # chunks are the checkpoints; HF layer checkpointing stays off
    m.enable_input_require_grads()
    pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=RE, task_type="CAUSAL_LM")); pm.train()
    x = torch.randint(0, 1000, (1, T), device="cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t0 = time.time()
    try:
        h = m.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).last_hidden_state
        h.float().pow(2).mean().backward(); torch.cuda.synchronize()
        r = f"{(torch.cuda.max_memory_allocated() - base) / 2**30:5.2f} GiB above weights, {time.time() - t0:5.1f}s"
    except torch.cuda.OutOfMemoryError:
        r = "OOM"
    del m, pm; MF.uninstall_layers(); torch.cuda.empty_cache()
    return r

if __name__ == "__main__":
    layers = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sliding_attention", "full_attention"]
    for T in (4096, 8192, 10240, 12288, 16384):
        print(f"T={T:6d}  stock: {run('stock', T, layers)}   |   chunked layer: {run('chunk', T, layers)}", flush=True)
