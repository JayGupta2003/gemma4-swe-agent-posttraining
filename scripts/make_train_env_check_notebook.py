"""CPU-only notebook (no GPU quota): what does Kaggle's official image offer for LoRA training, and what exactly is the
bf16 'qat-q4_0-unquantized' Gemma 4 31B checkpoint (config, shard sizes)? Answers the training-feasibility questions that
need no GPU: which libraries exist offline, and the model geometry for memory/throughput arithmetic."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "kaggle_assets" / "train_env_check"
OUT.mkdir(parents=True, exist_ok=True)

CELL = r'''import importlib.metadata as md, json, os, platform, subprocess, sys
from pathlib import Path
print('python', sys.version.split()[0], platform.platform())
for p in ['torch','transformers','peft','trl','accelerate','bitsandbytes','compressed-tensors','safetensors','datasets','deepspeed',
          'flash-attn','flash_attn','liger-kernel','xformers','vllm','unsloth','optimum','scipy','numpy']:
    try: print(f'{p:20s}', md.version(p))
    except md.PackageNotFoundError: print(f'{p:20s} -- not installed')
try:
    import torch; print('torch cuda available:', torch.cuda.is_available(), '| build cuda:', torch.version.cuda)
except Exception as e: print('torch import failed', e)
print(subprocess.run('df -h /kaggle/working /tmp | cat; free -g | cat; nproc', shell=True, capture_output=True, text=True).stdout)
cfgs = sorted(Path('/kaggle/input').rglob('config.json'))
print('config.json files under /kaggle/input:', [str(c) for c in cfgs][:10])
for c in cfgs:
    if 'unquantized' in str(c).lower() or '31b' in str(c).lower():
        d = c.parent
        print('\n=== model dir', d)
        files = sorted(d.iterdir())
        print({f.name: round(f.stat().st_size/1e9, 2) for f in files if f.is_file()})
        cfg = json.loads(c.read_text())
        txt = cfg.get('text_config', cfg)
        keep = ['architectures','model_type','hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','intermediate_size',
                'vocab_size','max_position_embeddings','sliding_window','layer_types','torch_dtype','dtype','quantization_config']
        print('top-level keys:', list(cfg.keys()))
        print(json.dumps({k: cfg[k] for k in keep if k in cfg}, indent=1)[:1500])
        print('text_config:', json.dumps({k: txt[k] for k in keep if k in txt}, indent=1)[:2500])
        idx = d / 'model.safetensors.index.json'
        if idx.exists():
            wm = json.loads(idx.read_text())['weight_map']; ks = list(wm)
            print('n tensors', len(ks), '| sample names:', ks[:3], ks[-3:])
            lay = sorted({k for k in ks if '.layers.0.' in k})
            print('layer-0 tensors:', lay)
'''

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Training environment check (CPU only)\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "train_env_check.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-train-env-check", "title": "Gemma 4 Train Env Check",
    "code_file": "train_env_check.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": [], "kernel_sources": [], "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": ["google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT)
