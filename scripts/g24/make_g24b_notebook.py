"""Generate kaggle_assets/g24b/ : G24b = E4B self-RFT training + evaluation (rule pre-registered in docs/STRATEGY.md, 2026-11-17).

LoRA SFT of the E4B on its OWN test-verified successful attempts from the 171-task pool (G20 pass r1 + G24a passes r2-r4, built by
scripts/g24/build_rft_data.py: resolved, no test files touched, <= 2 shortest attempts per task, loss on the model's own tokens), 2 epochs,
lr 1e-4, warmup 10% of the planned steps; then the adapter in the deployed E4B arm on all 75 competition tasks (round e1, seed 201 as G18/G21),
and a second round e2 (seed 202) in the same kernel when time allows (the rule's confirmation round; saves a separate boot).

Derived from the generated G18 notebook (kaggle_assets/g18/g18.ipynb: same trainer, packaging, adapter-effect and smoke gates) by exact-string edits, each asserted.
Dataset noobhunterjay/gemma4-g24-data = sft_sequences.jsonl + stats.json + E4B/ (byte-identical to submission_e4b).
Budget: training ~0.8 M tokens x 2 epochs at ~300 tok/s ~ 1.5 h; staging/boot/smoke ~0.5 h; each 75-task round ~40 min. Watchdog 300 min.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g18" / "g18.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g24b"
OUT.mkdir(parents=True, exist_ok=True)


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
kinds = [c["cell_type"] for c in cells]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
SETUP, TRAIN, EVAL = 3, 7, 13
assert kinds[SETUP] == kinds[TRAIN] == kinds[EVAL] == "code" and "G3_EPOCHS" in txt[TRAIN] and "run_round('e1', ids)" in txt[EVAL]

s = txt[SETUP]
s = sub(s, "G3 = WORK / 'g18'", "G3 = WORK / 'g24b'")
s = sub(s, "'g18_summary.json'", "'g24b_summary.json'")
s = sub(s, "BUDGET_MIN = 480", "BUDGET_MIN = 300")
txt[SETUP] = s

s = txt[TRAIN]
s = sub(s, "t0 = time.time(); STAGED = Path('/tmp/ckpt_bf16')\n",
        "t0 = time.time(); STAGED = Path('/tmp/ckpt_bf16')\n"
        "N_SEQ = sum(1 for _ in open(SFT, encoding='utf-8')); PLANNED = 2 * -(-N_SEQ // 2); WARMUP = max(1, round(0.1 * PLANNED))   # 2 epochs, ACC 2\n"
        "R['plan'] = {'sequences': N_SEQ, 'optimizer_steps': PLANNED, 'warmup': WARMUP}; print('plan:', R['plan'])\n")
s = sub(s, "G3_ACC='2', G3_LR='1e-4', G3_EPOCHS='1', G3_WARMUP='10'", "G3_ACC='2', G3_LR='1e-4', G3_EPOCHS='2', G3_WARMUP=str(WARMUP)")
s = sub(s, "max_seconds=5.5 * 3600)", "max_seconds=3.0 * 3600)")
txt[TRAIN] = s

s = txt[EVAL]
s = sub(s, "run_round('e1', ids)\nrep = R.get('e1', {})\n",
        "run_round('e1', ids)\nrep = R.get('e1', {})\n"
        "print('round e1:', {k: v for k, v in rep.items() if k not in ('resolved_ids', 'proxy')}, flush=True)\n"
        "# ---- confirmation round e2 (different order) if at least 55 minutes remain before the watchdog\n"
        "if (time.time() - T_START) / 60 < BUDGET_MIN - 55:\n"
        "    ids2 = list(ALL_IDS); random.Random(202).shuffle(ids2); run_round('e2', ids2)\n"
        "else:\n"
        "    R['e2'] = {'skipped': f'{(time.time() - T_START) / 60:.0f} min elapsed'}; save_results()\n")
s = sub(s, "===== G18 SUMMARY =====", "===== G24b SUMMARY (E4B self-RFT) =====")
s = sub(s, "print('adapter eval round e1:', {k: v for k, v in rep.items() if k not in ('resolved_ids', 'proxy')})\n",
        "print('plan:', R.get('plan'))\n"
        "print('adapter round e1:', {k: v for k, v in rep.items() if k not in ('resolved_ids', 'proxy')})\n"
        "print('adapter round e2:', {k: v for k, v in R.get('e2', {}).items() if k not in ('resolved_ids', 'proxy')})\n")
s = sub(s, "print('BASE E4B (same arm, no adapter; G14c e4b_b2): 7/75 resolved, ~69 patches, <= 1 capped, ~120 s per task; other base rounds 10 and 8; the 31B base solves 25 and 28')\n",
        "print('BASE E4B (same arm, no adapter): 10, 7, 8 of 75 (budget 4096), 11 (512 cap); patches 65-69; capped <= 1; 114-125 s per task; 17 distinct over 4 rounds')\n")
s = sub(s, "print('RULE: interesting if resolved >= 16 and patches >= 55 and capped <= 8; failure if capped > 20 or mean_task_s > 250 (loop regression)')",
        "print('RULE (pre-registered): CANDIDATE e1 >= 14 & patches >= 55 & capped <= 8 & mean <= 200 s (submit if e1/e2 mean >= 12) | WEAK 12-13 | NULL <= 11 | FAILURE capped > 20 or mean > 250 s')")
txt[EVAL] = s

txt[0] = ("# G24b: E4B self-RFT (LoRA SFT on its own test-verified pool successes, 2 epochs) + two 75-task evaluation rounds with the adapter\n\n"
          "Wiring check -> train (staged checkpoint, supervised) -> package -> vLLM + capture proxy with the adapter -> adapter-effect + smoke gates -> rounds e1, e2.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
(OUT / "g24b.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g18" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
meta.update({"id": "noobhunterjay/gemma-4-g24b-e4b-rft-train", "title": "Gemma 4 G24b E4B RFT Train", "code_file": "g24b.ipynb",
             "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g24-data"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
