"""Generate kaggle_assets/g14b0/ : does vLLM on Kaggle apply a LoRA adapter to the E4B (gemma-4-e4b-it-qat-w4a16-ct)? ~15 min, one L4x4 session (~1 quota h).

Why first: the E4B has shared-KV layers (k_proj/v_proj exist in only 24 of 42 layers) and per-layer inputs, and the 31B's LoRA path in vLLM was broken once ("silently reset to zero").
Before spending ~8 quota h on training an E4B adapter we check, with the CANARY adapter from the CPU dry run (attention-only, random lora_B, scorer format, base id = the E4B),
that (1) vLLM starts with --enable-lora and loads it, (2) a request to the adapter name returns different tokens/logprobs than the same request to the base model.

Derived from the generated G14a notebook (E4B serving on 4xL4, harness imports, capture proxy) by exact-string edits, each asserted.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g14a" / "g14a.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g14b0"
OUT.mkdir(parents=True, exist_ok=True)
WIRING_SRC = (ROOT / "scripts" / "g3" / "g3_wiring.py").read_text(encoding="utf-8").replace("\r\n", "\n")
E4B = "gemma-4-e4b-it-qat-w4a16-ct"
assert 'BASE = "gemma-4-31b-it-qat-w4a16-ct"' in WIRING_SRC
WIRING_SRC = WIRING_SRC.replace('BASE = "gemma-4-31b-it-qat-w4a16-ct"', f'BASE = "{E4B}"')


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}

# ---- setup: the arm gets the canary adapter (agent.yaml + sub_agents/code_analyzer.yaml declare `adapter: g3_lora`)
s = txt[3]
s = sub(s, "G2 = WORKING_DIR / 'g14a'", "G2 = WORKING_DIR / 'g14b0'")
s = sub(s, "'g14a_summary.json'", "'g14b0_summary.json'")
s = sub(s, "BUDGET_MIN = 130", "BUDGET_MIN = 45")
s = sub(s, "for must in ('agent.yaml',",
        "(CODE / 'g3_wiring.py').write_text(" + repr(WIRING_SRC) + ", encoding='utf-8')\n"
        "import g3_wiring as W\n"
        "CANARY = next(p.parent for p in Path('/kaggle/input').rglob('adapter_model.safetensors') if 'canary' in str(p).lower()); print('canary adapter dir:', CANARY)\n"
        "ARM = W.make_eval_arm(ARM, CANARY, WORKING_DIR / 'arm_canary'); RESULTS['wiring'] = W.check_wiring(ARM); print('wiring OK:', RESULTS['wiring'])\n"
        "for must in ('agent.yaml',")
txt[3] = s

# ---- vLLM: LoRA enabled
s = txt[5]
s = sub(s, "enable_lora=False,", "enable_lora=True, max_loras=8, max_lora_rank=128,")
txt[5] = s

# ---- check cell
CHECK = '''import requests, math
base_name = str(MODEL_PATH); adapter_name = W.ADAPTER
ms = requests.get(VLLM_URL + '/v1/models', timeout=60).json()
ids = [m['id'] for m in ms['data']]; RESULTS['served_models'] = ids; print('served models:', ids)
assert adapter_name in ids, f'adapter {adapter_name} not registered: {ids}'
PROMPTS = ['Write a Python function that returns the n-th Fibonacci number, then explain it in one sentence.',
           'The bug: `rich.table.Table` drops the caption when `show_header=False`. Which file would you look at first, and why?']
def ask(model, prompt):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'max_tokens': 64, 'logprobs': True, 'top_logprobs': 1,
            'chat_template_kwargs': {'enable_thinking': False}}
    r = requests.post(VLLM_URL + '/v1/chat/completions', json=body, timeout=600); r.raise_for_status()
    ch = r.json()['choices'][0]
    return ch['message'].get('content') or '', [t['logprob'] for t in (ch.get('logprobs') or {}).get('content', [])]
out = []
for p in PROMPTS:
    tb, lb = ask(base_name, p); ta, la = ask(adapter_name, p); tb2, lb2 = ask(base_name, p)
    n = min(len(lb), len(la))
    mean_abs = sum(abs(a - b) for a, b in zip(lb[:n], la[:n])) / max(n, 1)
    out.append({'prompt': p[:60], 'base_text': tb[:160], 'adapter_text': ta[:160], 'text_differs': tb != ta, 'base_repeatable': tb == tb2, 'mean_abs_logprob_diff': round(mean_abs, 4)})
    print(json.dumps(out[-1], indent=1))
RESULTS['canary'] = out
effect = any(o['text_differs'] or o['mean_abs_logprob_diff'] > 0.01 for o in out)
RESULTS['canary_verdict'] = 'ADAPTER APPLIED (outputs differ from the base)' if effect else 'ADAPTER HAS NO EFFECT (identical to the base): do NOT train for E4B before this is understood'
print('VERDICT:', RESULTS['canary_verdict']); RESULTS['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); save_results(); _wd.cancel()
server_instance.stop() if hasattr(server_instance, 'stop') else None
'''
txt[7] = CHECK
txt[0] = "# G14b0: does vLLM apply a LoRA adapter to the E4B? (canary adapter with random lora_B vs the base, same prompt, temperature 0)"
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
(OUT / "g14b0.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g14a" / "kernel-metadata.json").read_text(encoding="utf-8"))
meta["dataset_sources"] = meta["dataset_sources"] + ["noobhunterjay/gemma4-e4b-canary"]
meta.update({"id": "noobhunterjay/gemma-4-g14b0-e4b-lora-check", "title": "Gemma 4 G14b0 E4B LoRA Check", "code_file": "g14b0.ipynb"})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
