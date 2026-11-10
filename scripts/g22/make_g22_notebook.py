"""Generate kaggle_assets/g22/ : G22 = evaluation only of the G21 on-policy-distilled E4B adapter with its reasoning capped, plus a base-E4B control.

Question: G21 failed because the distilled E4B reasons ~2.3x longer (52/60 attempts hit the 8-minute cap). Did it learn anything besides length?
G22 serves the unchanged G21 adapter (dataset noobhunterjay/gemma4-g22-adapter = G21's train/adapter_full) with thinking_budget 4096 -> 512 and
max_output_tokens 8192 -> 4096 in the E4B arm's configs/sampling.yaml (the harness sends them as thinking_token_budget / max_completion_tokens; G12 showed
vLLM enforces the budget). Round e1: adapter @512 on all 75 tasks in G21's order (seed 201). Round b512: base E4B @512, same server, same order (control).

Derived from the generated G21 notebook (kaggle_assets/g21/g21.ipynb) by exact-string edits, each asserted: the training cell is replaced by loading the adapter;
everything else (wiring check, scorer-format packaging, vLLM + capture proxy, adapter-effect gate, smoke gate, rounds) is G21's proven path.

Pre-registered rule (docs/STRATEGY.md, written 2026-11-11 before any G22 result):
  SIGNAL   adapter resolved >= 12/75 with capped <= 8, and >= base@512 + 3 when the control finishes -> step 2 (OPD retrain, reasoning-close token masked)
  NULL     adapter resolved <= 11, or < base@512 + 3 -> the G21 adapter learned length, not skill; E4B final stays the base bundle
  FAILURE  adapter capped > 20 even at 512 -> the shift is not confined to the reasoning block
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g21" / "g21.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g22"
OUT.mkdir(parents=True, exist_ok=True)


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
kinds = [c["cell_type"] for c in cells]
assert kinds == ["markdown", "code"] * 7, kinds
SETUP, TRAIN, PACKAGE, EVAL = 3, 7, 9, 13

s = txt[SETUP]
s = sub(s, "G3 = WORK / 'g21'", "G3 = WORK / 'g22'")
s = sub(s, "'g21_summary.json'", "'g22_summary.json'")
s = sub(s, "BUDGET_MIN = 300", "BUDGET_MIN = 270")
s = sub(s, "shutil.copytree(_arms[0], _b1e); B1 = _b1e\n",
        "shutil.copytree(_arms[0], _b1e); B1 = _b1e\n"
        "_sp = B1 / 'configs' / 'sampling.yaml'; _s = _sp.read_text(encoding='utf-8')   # G22: cap the reasoning (the only change to the arm)\n"
        "assert _s.count('thinking_budget: 4096') == 1 and _s.count('max_output_tokens: 8192') == 1, _s\n"
        "_sp.write_text(_s.replace('thinking_budget: 4096', 'thinking_budget: 512').replace('max_output_tokens: 8192', 'max_output_tokens: 4096'), encoding='utf-8')\n"
        "print('sampling.yaml (capped):', _sp.read_text(encoding='utf-8')[-160:])\n")
txt[SETUP] = s

txt[TRAIN - 1] = "## Load the G21 adapter (no training)"
txt[TRAIN] = (
    "import hashlib\n"
    "t0 = time.time()\n"
    "_ad = sorted(p.parent for p in Path('/kaggle/input').rglob('adapter_model.safetensors') if 'g22-adapter' in str(p) and (p.parent / 'adapter_config.json').exists()); assert len(_ad) == 1, _ad\n"
    "assert hashlib.sha256((_ad[0] / 'adapter_model.safetensors').read_bytes()).hexdigest() == '348f020e3ab6f35f03f5edc267ac44e47a75730cb6d2c85a0165c1a3d28396b8', 'not the G21 adapter'\n"
    "shutil.copytree(_ad[0], G3 / 'train' / 'adapter_full', dirs_exist_ok=True)\n"
    "R['train'] = {'source': 'G21 adapter, kernel noobhunterjay/gemma-4-g21-e4b-opd v1 (122 steps coarse top-8 reverse KL)', 'path': str(_ad[0]),\n"
    "              'sha256': hashlib.sha256((_ad[0] / 'adapter_model.safetensors').read_bytes()).hexdigest()}\n"
    "R['train_wall_s'] = round(time.time() - t0); save_results(); print(R['train'])\n")

s = txt[PACKAGE]
s = sub(s, "EVAL_ARM = W.make_eval_arm(B1, G3 / 'adapters' / 'g1_full', WORK / 'eval_arm')\n",
        "EVAL_ARM = W.make_eval_arm(B1, G3 / 'adapters' / 'g1_full', WORK / 'eval_arm')\n"
        "_es = (EVAL_ARM / 'configs' / 'sampling.yaml').read_text(encoding='utf-8'); assert 'thinking_budget: 512' in _es and 'max_output_tokens: 4096' in _es, _es\n")
txt[PACKAGE] = s

s = txt[EVAL]
s = sub(s, "def run_round(tag, ids):\n    proxy.round_tag = tag; rd = G3 / tag\n",
        "def run_round(tag, ids, arm_dir=None):\n    arm_dir = arm_dir or EVAL_ARM\n    proxy.round_tag = tag; rd = G3 / tag\n")
s = sub(s, "submission_dir=EVAL_ARM, models=models", "submission_dir=arm_dir, models=models")
s = sub(s, "**budget_of(EVAL_ARM))", "**budget_of(arm_dir))")
s = sub(s, "        'main-agent requests use the adapter (>= 95%)': bool(main) and used / len(main) >= 0.95}",
        "        'main-agent requests use the adapter (>= 95%)': bool(main) and used / len(main) >= 0.95,\n"
        "        'requests carry thinking_token_budget 512': bool(main) and all(r['request'].get('thinking_token_budget') == 512 for r in main),\n"
        "        'requests carry max_completion_tokens 4096': bool(main) and all(r['request'].get('max_completion_tokens', r['request'].get('max_tokens')) == 4096 for r in main)}")
s = sub(s, "run_round('e1', ids)\nrep = R.get('e1', {})\n",
        "run_round('e1', ids)\nrep = R.get('e1', {})\n"
        "print('adapter @512 round e1:', {k: v for k, v in rep.items() if k not in ('resolved_ids', 'proxy')}, flush=True)\n"
        "# ---- control: base E4B @512 on the same server, same task order (the watchdog may stop it; e1 is already saved)\n"
        "run_round('b512', ids, arm_dir=B1)\n"
        "_b = [r for r in CP.read_records(str(G3 / 'capture.jsonl.gz')) if r.get('round') == 'b512']\n"
        "R['b512_model_names'] = sorted({r['request'].get('model') for r in _b}); save_results(); print('b512 model names:', R['b512_model_names'])\n")
s = sub(s, "===== G21 SUMMARY (OPD iteration 1) =====", "===== G22 SUMMARY (G21 adapter, reasoning capped at 512) =====")
s = sub(s, "print('training:', {k: v for k, v in R['train'].items() if k not in ('losses', 'grad_norms')})\n", "print('adapter:', R['train'])\n")
s = sub(s, "print('adapter eval round e1:', {k: v for k, v in rep.items() if k not in ('resolved_ids', 'proxy')})\n",
        "print('adapter @512 e1:', {k: v for k, v in rep.items() if k not in ('resolved_ids', 'proxy')})\n"
        "print('base @512 b512:', {k: v for k, v in R.get('b512', {}).items() if k not in ('resolved_ids', 'proxy')})\n")
s = sub(s, "print('RULE (pre-registered): CANDIDATE resolved >= 15 & patches >= 55 & capped <= 8 | WEAK SIGNAL 12-14 (same patch/cap conditions) -> OPD iteration 2 | NULL <= 11 | FAILURE capped > 20 or mean_task_s > 250')",
        "print('G21 (adapter @4096, 60 tasks): 6 resolved, 9 patches, 52 capped')\n"
        "print('RULE (pre-registered): SIGNAL adapter >= 12 & capped <= 8 & >= base@512 + 3 -> masked OPD retrain | NULL <= 11 or < base@512 + 3 | FAILURE capped > 20')")
txt[EVAL] = s

txt[0] = ("# G22: the G21 on-policy-distilled E4B adapter with its reasoning capped at 512 tokens, plus a base-E4B control at the same cap\n\n"
          "No training. Does the distilled adapter carry any skill beyond its longer reasoning? Wiring check -> load + package the G21 adapter -> vLLM + capture proxy -> "
          "adapter-effect + smoke gates (incl. the capped budget in real requests) -> 75-task round with the adapter -> 75-task control round with the base model.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
(OUT / "g22.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g21" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
meta.update({"id": "noobhunterjay/gemma-4-g22-e4b-opd-capped", "title": "Gemma 4 G22 E4B OPD capped", "code_file": "g22.ipynb",
             "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g21-data", "noobhunterjay/gemma4-g22-adapter"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
