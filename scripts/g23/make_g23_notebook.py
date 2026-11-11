"""Generate kaggle_assets/g23/ : G23 = action-only on-policy distillation of the E4B from the 31B, then the 75-task evaluation.

Same as G21 (kaggle_assets/g21/g21.ipynb, derived from it by asserted exact-string edits) except the training data: dataset noobhunterjay/gemma4-g23-data,
file opd_action_sequences.jsonl = the G21 rows with all reasoning-block targets removed from the loss (scripts/g23/build_action_only.py; 151k loss tokens).
The E4B arm (thinking budget 4096, B2 regime) still comes from noobhunterjay/gemma4-g21-data. Watchdog 330 min.

Pre-registered rule (docs/STRATEGY.md, written 2026-11-11 before any G23 result):
  CANDIDATE resolved >= 15 & patches >= 55 & capped <= 8 & "Wait" <= 1.5 per 10k chars -> second round, then E4B submission
  WEAK 12-14 (same guards) -> second round | NULL <= 11 -> stop E4B distillation | FAILURE capped > 20 or mean task > 250 s or "Wait" > 1.5 per 10k chars
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g21" / "g21.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g23"
OUT.mkdir(parents=True, exist_ok=True)


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
assert [c["cell_type"] for c in cells] == ["markdown", "code"] * 7
SETUP, EVAL = 3, 13

s = txt[SETUP]
s = sub(s, "G3 = WORK / 'g21'", "G3 = WORK / 'g23'")
s = sub(s, "'g21_summary.json'", "'g23_summary.json'")
s = sub(s, "BUDGET_MIN = 300", "BUDGET_MIN = 330")
s = sub(s, "rglob('opd_sequences.jsonl')", "rglob('opd_action_sequences.jsonl')")
txt[SETUP] = s

s = txt[EVAL]
s = sub(s, "===== G21 SUMMARY (OPD iteration 1) =====", "===== G23 SUMMARY (action-only OPD) =====")
s = sub(s, "print('RULE (pre-registered): CANDIDATE resolved >= 15 & patches >= 55 & capped <= 8 | WEAK SIGNAL 12-14 (same patch/cap conditions) -> OPD iteration 2 | NULL <= 11 | FAILURE capped > 20 or mean_task_s > 250')",
        "print('G21 (all tokens): 6/60, 52 capped | G22 (G21 adapter, reasoning capped 512): 9/75, 50 capped | base at 512: 11/75, 1 capped')\n"
        "print('RULE (pre-registered): CANDIDATE >= 15 & patches >= 55 & capped <= 8 & Wait <= 1.5/10k chars | WEAK 12-14 | NULL <= 11 | FAILURE capped > 20 or mean_task_s > 250 or Wait > 1.5/10k')")
txt[EVAL] = s

txt[0] = ("# G23: action-only on-policy distillation of the E4B from the 31B + evaluation round with the adapter\n\n"
          "G21's data with every reasoning-block target removed from the loss (tool calls, arguments, edit text and visible output only). "
          "Wiring check -> train (staged checkpoint, supervised) -> package -> vLLM + capture proxy with the adapter -> adapter-effect + smoke gates -> 75-task evaluation round.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
(OUT / "g23.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g21" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
meta.update({"id": "noobhunterjay/gemma-4-g23-e4b-opd-action", "title": "Gemma 4 G23 E4B OPD action", "code_file": "g23.ipynb",
             "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g21-data", "noobhunterjay/gemma4-g23-data"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
