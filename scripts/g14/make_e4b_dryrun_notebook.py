"""Generate kaggle_assets/g14_dry/ : CPU-only (no GPU quota) dry run of the E4B training path on the REAL checkpoint (gemma-4-e4b-it-qat-q4_0-unquantized, bf16).

Checks, on the real architecture (per-layer inputs, shared-KV layers, 42 layers, hidden 2560):
  1. text config facts that decide memory and chunking (layer types, kv-shared layers, per-layer input size, softcap, tying);
  2. our LoRA target regex matches the expected modules (count per type, trainable parameters);
  3. one forward + backward on a short random sequence with the chunked lm_head loss path (gradients reach the LoRA matrices);
  4. the PEFT adapter exports to the scorer format (scripts/g1/make_adapters.py with SCORER_MODEL = the E4B scorer id) and every key matches the expected naming.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "scripts"))
import make_experiment1 as e1  # noqa: E402  (INSTALL cell)

OUT = ROOT / "kaggle_assets" / "g14_dry"
OUT.mkdir(parents=True, exist_ok=True)
MAKE_ADAPTERS = (ROOT / "scripts" / "g1" / "make_adapters.py").read_text(encoding="utf-8").replace("\r\n", "\n")

RUN = r'''import glob, json, os, re, sys, time
sys.modules["torchao"] = None   # Kaggle's torchao 0.10.0 is rejected by peft 0.19.1
from pathlib import Path
import torch, transformers
print('torch', torch.__version__, 'transformers', transformers.__version__, 'cpus', os.cpu_count())
cand = sorted(glob.glob('/kaggle/input/models/google/gemma-4/*/gemma-4-e4b-it-qat-q4_0-unquantized/*/config.json'))
print('model candidates:', cand); assert len(cand) == 1
MODEL = str(Path(cand[0]).parent)
cfg = transformers.AutoConfig.from_pretrained(MODEL)
tc = cfg.get_text_config()
facts = {k: getattr(tc, k, None) for k in ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'global_head_dim', 'sliding_window',
         'hidden_size_per_layer_input', 'vocab_size_per_layer_input', 'num_kv_shared_layers', 'use_double_wide_mlp', 'attention_k_eq_v', 'final_logit_softcapping', 'vocab_size',
         'intermediate_size', 'enable_moe_block', 'tie_word_embeddings')}
import collections
facts['layer_types'] = dict(collections.Counter(tc.layer_types)); facts['architectures'] = cfg.architectures; facts['tie_word_embeddings_top'] = getattr(cfg, 'tie_word_embeddings', None)
print('TEXT CONFIG:', json.dumps(facts, indent=1))
t0 = time.time()
model = transformers.Gemma4ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa')
print('loaded in', round(time.time() - t0), 's | params (B):', round(sum(p.numel() for p in model.parameters()) / 1e9, 2))
from peft import LoraConfig, get_peft_model
TARGET_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
model.config.use_cache = False
pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias='none', target_modules=TARGET_RE, task_type='CAUSAL_LM'))
mods = [n for n, _ in pm.named_modules() if n.endswith('.lora_A.default')]
by = collections.Counter(n.split('.lora_A')[0].split('.')[-1] for n in mods)
print('LoRA modules:', len(mods), dict(by), '| trainable params (M):', round(sum(p.numel() for p in pm.parameters() if p.requires_grad) / 1e6, 1))
print('example module names:', mods[:2], mods[-1])
# one forward + backward with the chunked lm_head loss on a short random sequence (CPU)
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
pm.train(); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
ids = torch.randint(5, 20000, (1, 192))
t0 = time.time()
hs = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0]
softcap = tc.final_logit_softcapping
def chunk_loss(h, tgt):
    lg = model.lm_head(h).float()
    if softcap: lg = torch.tanh(lg / softcap) * softcap
    return F.cross_entropy(lg, tgt, reduction='sum')
pos = list(range(50, 190)); tgt = ids[0, [p + 1 for p in pos]]
loss = checkpoint(chunk_loss, hs[pos], tgt, use_reentrant=False) / len(pos)
loss.backward()
gn = [float(p.grad.norm()) for p in pm.parameters() if p.requires_grad and p.grad is not None]
print('forward+backward OK in', round(time.time() - t0), 's | loss', round(float(loss), 3), '| LoRA params with grad', len(gn), 'of', sum(1 for p in pm.parameters() if p.requires_grad), '| B grads nonzero:', sum(g > 0 for g in gn))
# export in scorer format
out = Path('/kaggle/working/e4b_adapter_raw'); pm.save_pretrained(str(out))
os.environ['SCORER_MODEL'] = 'google/gemma-4-e4b-it-qat-w4a16-ct'
exec(MAKE_ADAPTERS_SRC.replace('if __name__ == "__main__":', 'if False:'))
rep = main(out, Path('/kaggle/working/e4b_adapters'), None)
print('ADAPTER EXPORT OK:', json.dumps({k: rep[k] for k in ('n_tensors', 'rank', 'alpha')}), '| g1_full tensors', rep['g1_full']['tensors'], 'bytes', rep['g1_full']['bytes'], '| modules', rep['g1_full']['target_modules'])
print(json.loads((Path('/kaggle/working/e4b_adapters/g1_full/adapter_config.json')).read_text())['base_model_name_or_path'])
print('E4B DRY RUN: PASS')'''


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


nb = {"cells": [code(e1.INSTALL), code("MAKE_ADAPTERS_SRC = " + json.dumps(MAKE_ADAPTERS)), code(RUN)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "g14_dry.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g14-e4b-dryrun", "title": "Gemma 4 G14 E4B Dryrun", "code_file": "g14_dry.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"], "kernel_sources": [], "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": ["google/gemma-4/Transformers/gemma-4-e4b-it-qat-q4_0-unquantized/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT)
