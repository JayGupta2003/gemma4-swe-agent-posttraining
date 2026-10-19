"""Generate kaggle_assets/g11/ : G11 = LoRA SFT of the 31B on the resolved POOL attempts of G10 (71 sequences, 54 tasks from 54 repos, 0.87M tokens), 2 epochs,
then the 75-task competition evaluation round with the adapter. Training repos != evaluation repos, so all 75 evaluation tasks are unseen.

Derived from the G3 notebook (kaggle_assets/g3/g3.ipynb, same train_g3.py / memfix / wiring / smoke gate / evaluation code) by exact-string edits, each asserted:
  - data: dataset noobhunterjay/gemma4-g11-sft (sft_sequences.jsonl from scripts/g10/build_sft_general.py on the G10 capture)
  - 2 epochs (71 optimizer steps at accumulation 2), supervisor limit 7 h, watchdog 560 min
Expected: ~6.0 h training + ~0.4 h staging/boot/smoke + ~1.6 h evaluation = ~8 h of L4x4 (~16 quota h).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g3" / "g3.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g11"
OUT.mkdir(parents=True, exist_ok=True)


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
changed = set()
for i, s in txt.items():
    if cells[i]["cell_type"] != "code":
        continue
    n = s
    if "BUDGET_MIN = 340" in n:
        n = sub(n, "BUDGET_MIN = 340", "BUDGET_MIN = 560")
    if "DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')" in n:
        # the competition is now mounted at /kaggle/input/gemma-4-developer-agent/ (seen in the G10 v2 log); discover it instead of hard-coding the old path
        n = sub(n, "DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent');",
                "DATA_DIR = next(p.parent for p in Path('/kaggle/input').rglob('tasks.jsonl') if (p.parent / 'sample_submission').is_dir() and (p.parent / 'snapshots').is_dir());")
        assert "competitions/gemma-4-developer-agent" not in n
    if "G3_EPOCHS='1'" in n:
        n = sub(n, "G3_EPOCHS='1'", "G3_EPOCHS='2'")
        n = sub(n, "max_seconds=4.0 * 3600", "max_seconds=7.0 * 3600")
    if n != s:
        txt[i] = n
        changed.add(i)
assert len(changed) == 2, changed   # SETUP (watchdog) and TRAIN (epochs, supervisor limit)
txt[0] = ("# G11: LoRA SFT on resolved POOL attempts (G10: 71 sequences, 54 tasks, 54 repos) x 2 epochs + evaluation round on the 75 competition tasks (all unseen)\n\n"
          "Wiring check -> train -> package -> vLLM + capture proxy with the adapter -> smoke gate -> 75-task evaluation round. Base reference on the same tasks: 25/75 and 28/75.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")

(OUT / "g11.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g3" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
assert "noobhunterjay/gemma4-g3-sft" in meta["dataset_sources"]
meta["dataset_sources"] = [("noobhunterjay/gemma4-g11-sft" if d == "noobhunterjay/gemma4-g3-sft" else d) for d in meta["dataset_sources"]]
meta.update({"id": "noobhunterjay/gemma-4-g11-train", "title": "Gemma 4 G11 Train", "code_file": "g11.ipynb"})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
