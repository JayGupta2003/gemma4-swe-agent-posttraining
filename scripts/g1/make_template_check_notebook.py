"""CPU-only: is the chat template of the SERVED w4a16-ct checkpoint identical to the bf16 training checkpoint's? (training must use the serving template)"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "kaggle_assets" / "g1_template_check"
OUT.mkdir(parents=True, exist_ok=True)
CELL = r'''import hashlib, json, shutil
from pathlib import Path
W = Path('/kaggle/working')
rep = {}
for cfg in sorted(Path('/kaggle/input').rglob('config.json')):
    d = cfg.parent
    if 'gemma-4-31b' not in str(d): continue
    key = 'w4a16' if 'w4a16' in str(d) else 'bf16_unquantized'
    t = d / 'chat_template.jinja'
    tc = json.loads((d / 'tokenizer_config.json').read_text()) if (d / 'tokenizer_config.json').exists() else {}
    rep[key] = {'dir': str(d), 'files': sorted(p.name for p in d.iterdir())[:40],
                'chat_template_jinja_sha256': hashlib.sha256(t.read_bytes()).hexdigest() if t.exists() else None,
                'template_in_tokenizer_config': bool(tc.get('chat_template'))}
    if t.exists(): shutil.copy(t, W / f'chat_template_{key}.jinja')
    gc = d / 'generation_config.json'
    if gc.exists(): shutil.copy(gc, W / f'generation_config_{key}.json')
rep['identical_templates'] = (rep.get('w4a16', {}).get('chat_template_jinja_sha256') == rep.get('bf16_unquantized', {}).get('chat_template_jinja_sha256'))
(W / 'template_check.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))'''
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "g1_template_check.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g1-template-check", "title": "Gemma 4 G1 Template Check", "code_file": "g1_template_check.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": [], "kernel_sources": [], "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": ["google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2", "google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT)
