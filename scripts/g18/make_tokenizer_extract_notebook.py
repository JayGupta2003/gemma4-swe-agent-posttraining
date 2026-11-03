"""CPU-only Kaggle notebook: copy the E4B tokenizer + chat template out of the two E4B checkpoints (bf16 training checkpoint, served w4a16-ct) so the teacher-episode
renderer can be developed and tested locally against the REAL artifacts, and report whether training and serving share the template (sha256) and the tokenizer.
Usage: python scripts/g18/make_tokenizer_extract_notebook.py  ->  kaggle_assets/g18_tokenizer/ (push with `kaggle kernels push -p kaggle_assets/g18_tokenizer`)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "kaggle_assets" / "g18_tokenizer"
OUT.mkdir(parents=True, exist_ok=True)
CELL = r'''import hashlib, json, shutil
from pathlib import Path
W = Path('/kaggle/working'); (W / 'tok').mkdir(exist_ok=True)
rep = {}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
for cfg in sorted(Path('/kaggle/input').rglob('config.json')):
    d = cfg.parent
    if 'gemma-4-e4b' not in str(d): continue
    key = 'w4a16' if 'w4a16' in str(d) else 'bf16_unquantized'
    rep[key] = {'dir': str(d), 'files': sorted(p.name for p in d.iterdir())[:60]}
    for n in ('chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'special_tokens_map.json'):
        p = d / n
        rep[key][n + '_sha256'] = sha(p)
        if p.exists():
            (W / 'tok' / key).mkdir(parents=True, exist_ok=True)
            shutil.copy(p, W / 'tok' / key / n)
rep['identical_template'] = rep.get('w4a16', {}).get('chat_template.jinja_sha256') == rep.get('bf16_unquantized', {}).get('chat_template.jinja_sha256')
rep['identical_tokenizer'] = rep.get('w4a16', {}).get('tokenizer.json_sha256') == rep.get('bf16_unquantized', {}).get('tokenizer.json_sha256')
(W / 'tokenizer_check.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))'''
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "g18_tokenizer.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g18-tokenizer", "title": "Gemma 4 G18 Tokenizer", "code_file": "g18_tokenizer.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": [], "kernel_sources": [], "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": ["google/gemma-4/Transformers/gemma-4-e4b-it-qat-q4_0-unquantized/2", "google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT)
