"""Generate kaggle_assets/g21/ : G21 = on-policy distillation (OPD) of the E4B from the 31B, iteration 1, then the 75-task competition evaluation.

Training data (dataset noobhunterjay/gemma4-g21-data): 244 rows from G20 (the deployed E4B arm's OWN rollouts on the 171-task pool, solved or not) with the 31B's top-8
log-probs at every student token (G20b), built by scripts/g20/build_opd_dataset.py: 2.93 M tokens, 711.9k loss tokens. Trainer scripts/g20/train_opd.py (train_g3 with the
loss replaced by the coarse top-8 + rest KL of scripts/g20/opd_loss.py, reverse KL), LoRA r16 attn+MLP, lr 1e-4, 1 epoch (122 optimizer steps at ACC 2), HF gradient
checkpointing. Evaluation: the same arm with the adapter (submission_e4b, B2 regime 8 min / 28 calls, temperature 0.2), all 75 competition tasks (none trained on).

Derived from the generated G18 notebook (kaggle_assets/g18/g18.ipynb: E4B training -> scorer-format adapter -> wiring check -> vLLM with the adapter + capture ->
adapter-effect gate -> smoke gate -> 75-task round) by exact-string edits, each asserted. Expected: ~2.2 h training + ~0.2 h staging + ~0.3 h boot/smoke + ~0.7 h
evaluation = ~3.4 h wall (~7 quota h). Watchdog 300 min.

Pre-registered rule (written 2026-11-04, before any G21 result; base E4B on the same 75 tasks: 7, 10 and 8 resolved, ~68 patches, <= 1 capped, ~120 s per task):
  CANDIDATE   resolved >= 15 with patches >= 55 and capped <= 8   -> confirm with a second evaluation round, then an E4B leaderboard submission
  WEAK SIGNAL resolved 12-14 with patches >= 55 and capped <= 8   -> OPD iteration 2 (new rollouts from the adapter, re-scored by the 31B) before judging
  NULL        resolved <= 11
  FAILURE     capped > 20 or mean task > 250 s (loop regression), whatever the resolved count
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g18" / "g18.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g21"
OUT.mkdir(parents=True, exist_ok=True)
NEW_CODE = {n: (ROOT / "scripts" / "g20" / n).read_text(encoding="utf-8").replace("\r\n", "\n") for n in ("train_opd.py", "opd_loss.py")}


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
kinds = [c["cell_type"] for c in cells]
assert kinds == ["markdown", "code", "markdown", "code", "markdown", "code", "markdown", "code", "markdown", "code", "markdown", "code", "markdown", "code"], kinds
SETUP, TRAIN, EVAL = 3, 7, 13

s = txt[SETUP]
s = sub(s, "G3 = WORK / 'g18'", "G3 = WORK / 'g21'")
s = sub(s, "'g18_summary.json'", "'g21_summary.json'")
s = sub(s, "BUDGET_MIN = 480", "BUDGET_MIN = 300")
s = sub(s, "sys.path.insert(0, str(CODE))\n", "for n, txt in " + json.dumps(NEW_CODE) + ".items(): (CODE / n).write_text(txt, encoding='utf-8')   # OPD trainer + loss\nsys.path.insert(0, str(CODE))\n")
s = sub(s, "_sft = sorted(Path('/kaggle/input').rglob('sft_sequences.jsonl'))", "_sft = sorted(Path('/kaggle/input').rglob('opd_sequences.jsonl'))")
txt[SETUP] = s

s = txt[TRAIN]
s = sub(s, "G3_ACC='2', G3_LR='1e-4', G3_EPOCHS='1', G3_WARMUP='10', G3_HF_GC='1',", "G3_ACC='2', G3_LR='1e-4', G3_EPOCHS='1', G3_WARMUP='10', G3_HF_GC='1', OPD_DIV='rkl',")
s = sub(s, "R['train_run'] = U.run_supervised([sys.executable, str(CODE / 'train_g3.py')], env, G3 / 'train.log',\n"
           "                                  [('[g3] loaded in', 600), ('[g3] step 1/', 1800)], max_seconds=5.5 * 3600)   # 700 episodes ~ 4.3 M tokens ~ 4 h at ~300 tok/s",
        "R['train_run'] = U.run_supervised([sys.executable, str(CODE / 'train_opd.py')], env, G3 / 'train.log',\n"
        "                                  [('[opd] loaded in', 600), ('[opd] step 1/', 1800)], max_seconds=3.5 * 3600)   # 244 rows, 2.93 M tokens ~ 2.2 h at ~375 tok/s (G18 measured)")
txt[TRAIN] = s

s = txt[EVAL]
s = sub(s, "===== G18 SUMMARY =====", "===== G21 SUMMARY (OPD iteration 1) =====")
s = sub(s, "print('RULE: interesting if resolved >= 16 and patches >= 55 and capped <= 8; failure if capped > 20 or mean_task_s > 250 (loop regression)')",
        "print('RULE (pre-registered): CANDIDATE resolved >= 15 & patches >= 55 & capped <= 8 | WEAK SIGNAL 12-14 (same patch/cap conditions) -> OPD iteration 2 | NULL <= 11 | FAILURE capped > 20 or mean_task_s > 250')")
txt[EVAL] = s

txt[0] = ("# G21: on-policy distillation of the E4B from the 31B (iteration 1) + evaluation round with the adapter\n\n"
          "Data: the E4B's own G20 rollouts (244 context segments) with the 31B's top-8 log-probs at every student token (G20b). Loss: coarse top-8 + rest reverse KL. "
          "Wiring check -> train (staged checkpoint, supervised) -> package -> vLLM + capture proxy with the adapter -> adapter-effect + smoke gates -> 75-task evaluation round.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
(OUT / "g21.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g18" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
meta.update({"id": "noobhunterjay/gemma-4-g21-e4b-opd", "title": "Gemma 4 G21 E4B OPD", "code_file": "g21.ipynb",
             "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g21-data"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
