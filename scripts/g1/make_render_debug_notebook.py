"""CPU-only: dump the 31B checkpoint's real chat template/tokenizer config and diagnose why rendering was not prefix-stable."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "kaggle_assets" / "g1_render_debug"
OUT.mkdir(parents=True, exist_ok=True)
CELL = r'''import json, shutil, sys
from pathlib import Path
import transformers
from transformers import AutoTokenizer
print('transformers', transformers.__version__)
M = next(p.parent for p in Path('/kaggle/input').rglob('config.json') if 'unquantized' in str(p).lower())
D = next(Path('/kaggle/input').rglob('g1_train.jsonl'))
W = Path('/kaggle/working'); W.mkdir(exist_ok=True)
for f in ('chat_template.jinja', 'tokenizer_config.json', 'generation_config.json', 'config.json'):
    if (M / f).exists(): shutil.copy(M / f, W / f)
print('model dir files:', sorted(p.name for p in M.iterdir()))
tok = AutoTokenizer.from_pretrained(M)
print('has chat_template:', bool(getattr(tok, 'chat_template', None)), 'len', len(tok.chat_template or ''))
rows = [json.loads(l) for l in open(D, encoding='utf-8')]
def flat(msgs):
    out = []
    for m in msgs:
        if m['role'] == 'assistant' and m.get('tool_calls'):
            calls = '\n'.join(json.dumps({'name': c['function']['name'], 'arguments': c['function']['arguments']}) for c in m['tool_calls'])
            out.append({'role': 'assistant', 'content': (m.get('content') or '') + '\n<tool_call>\n' + calls + '\n</tool_call>'})
        elif m['role'] == 'tool':
            out.append({'role': 'user', 'content': '<tool_response>\n' + (m.get('content') or '') + '\n</tool_response>'})
        else:
            out.append({'role': m['role'], 'content': m.get('content') or ''})
    return out
def t(m, g=False): return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=g)
report = {'transformers': transformers.__version__}
for mode, fn in (('native', lambda x: x), ('flat', flat)):
    msgs = fn(rows[0]['messages'])
    try:
        full = t(msgs)
    except Exception as e:
        report[mode] = f'render error {type(e).__name__}: {str(e)[:300]}'; continue
    info = {'full_len': len(full), 'head': full[:200], 'fails': []}
    for i, m in enumerate(msgs):
        if m['role'] != 'assistant': continue
        p, q = t(msgs[:i], True), t(msgs[:i + 1])
        a, b = (p, q) if not q.startswith(p) else (q, full)
        if not (q.startswith(p) and full.startswith(q)):
            k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            info['fails'].append({'assistant_idx': i, 'q_startswith_p': q.startswith(p), 'full_startswith_q': full.startswith(q), 'A': a[max(0, k - 120):k + 120], 'B': b[max(0, k - 120):k + 120]})
            if len(info['fails']) >= 2: break
    report[mode] = info
    # first assistant turn rendering for inspection
    j = next(i for i, m in enumerate(msgs) if m['role'] == 'assistant')
    report[mode + '_sample_tail'] = t(msgs[:j + 3])[-700:]
(W / 'render_debug.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'head'}) for k, v in report.items()}, indent=1, ensure_ascii=False)[:5000])
'''
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "g1_render_debug.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g1-render-debug", "title": "Gemma 4 G1 Render Debug", "code_file": "g1_render_debug.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": ["noobhunterjay/gemma4-g1-data"], "kernel_sources": [], "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": ["google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT)
