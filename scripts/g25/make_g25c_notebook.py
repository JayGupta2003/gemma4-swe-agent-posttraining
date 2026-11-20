"""Generate kaggle_assets/g25c/ : G25c = E4B self-RFT training + evaluation on the G25 data (rule pre-registered in docs/STRATEGY.md, 2026-11-19).

Identical recipe to G24b (LoRA r16 attention+MLP, lr 1e-4, 2 epochs, warmup 10% of planned steps, ACC 2; adapter in the deployed E4B arm on the 75
competition tasks, round e1 seed 201, confirmation round e2 seed 202 when time allows), so G24b is the matched control. Only the data changes
(G20 + G24a pool successes + G25b FreshSWE successes, <= 2 attempts per task, <= 20 tasks per repo) and, because there is ~4x more of it, the
training time limit and the watchdog, which are sized here from the data's token count (stats.json) at the measured G24b rate (326 tok/s).

Derived from the generated G24b notebook by exact-string edits, each asserted.
Usage: python scripts/g25/make_g25c_notebook.py <data dir with stats.json>
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g24b" / "g24b.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g25c"
OUT.mkdir(parents=True, exist_ok=True)
STATS = json.loads((Path(sys.argv[1]) / "stats.json").read_text(encoding="utf-8"))

TOK_PER_S = 300                                          # G24b measured 326; keep a margin
train_h = 2 * STATS["tokens"] / TOK_PER_S / 3600         # 2 epochs
MAX_TRAIN_H = round(min(7.0, math.ceil(train_h * 1.25 * 4) / 4), 2)
BUDGET_MIN = int(MAX_TRAIN_H * 60 + 40 + 2 * 50 + 20)    # train + staging/boot/gates + two 75-task rounds + margin
assert BUDGET_MIN <= 11.5 * 60, BUDGET_MIN               # Kaggle's 12 h session limit


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
SETUP, TRAIN, EVAL = 3, 7, 13

s = txt[SETUP]
s = sub(s, "G3 = WORK / 'g24b'", "G3 = WORK / 'g25c'")
s = sub(s, "'g24b_summary.json'", "'g25c_summary.json'")
s = sub(s, "BUDGET_MIN = 300", f"BUDGET_MIN = {BUDGET_MIN}")
txt[SETUP] = s

s = txt[TRAIN]
s = sub(s, "max_seconds=3.0 * 3600)", f"max_seconds={MAX_TRAIN_H} * 3600)")
txt[TRAIN] = s

s = txt[EVAL]
s = sub(s, "===== G24b SUMMARY (E4B self-RFT) =====", "===== G25c SUMMARY (E4B self-RFT, pool + FreshSWE-Py) =====")
s = sub(s, "print('BASE E4B (same arm, no adapter): 10, 7, 8 of 75 (budget 4096), 11 (512 cap); patches 65-69; capped <= 1; 114-125 s per task; 17 distinct over 4 rounds')\n",
        "print('BASE E4B (same arm, no adapter): 10, 7, 8 of 75 (budget 4096), 11 (512 cap); patches 65-69; capped <= 1; 114-125 s per task; 17 distinct over 4 rounds')\n"
        "print('G24b (same recipe, 37 pool tasks): e1 7, e2 10; 64/68 patches; 0/1 capped; 149/156 s')\n")
txt[EVAL] = s

txt[0] = ("# G25c: E4B self-RFT (LoRA SFT on its own test-verified successes: 171-task pool + FreshSWE-Py, 2 epochs) + two 75-task evaluation rounds\n\n"
          "Wiring check -> train (staged checkpoint, supervised) -> package -> vLLM + capture proxy with the adapter -> adapter-effect + smoke gates -> rounds e1, e2.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
(OUT / "g25c.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g24b" / "kernel-metadata.json").read_text(encoding="utf-8"))
meta.update({"id": "noobhunterjay/gemma-4-g25c-e4b-rft-train", "title": "Gemma 4 G25c E4B RFT Train", "code_file": "g25c.ipynb",
             "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g25-data"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT, "| tokens", STATS["tokens"], "| est train h", round(train_h, 2), "| max train h", MAX_TRAIN_H, "| watchdog min", BUDGET_MIN,
      "| est q-h (L4x4 billed 2x)", round(2 * BUDGET_MIN / 60, 1))
