"""Generate kaggle_assets/g16/ : G16 = pool round p3, identical to G15's p2 (31B, B1 arm, 5-minute cap, temperature 1.0 / top_p 0.95 / top_k 64, capture proxy, 171-task pool)
except the round tag (p3) and therefore the task order (shuffle seed 100 + 3). Third sample of every pool task under the same configuration, for the preference pair builder
(docs/PREFERENCE_RUN.md: G13b, gate >= 40 pairs after the pair rules). Derived from the generated G15 notebook by exact-string edits, each asserted.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g15" / "g15.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g16"
OUT.mkdir(parents=True, exist_ok=True)


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
s = txt[3]
s = sub(s, "G2 = WORKING_DIR / 'g15'", "G2 = WORKING_DIR / 'g16'")
s = sub(s, "'g15_summary.json'", "'g16_summary.json'")
txt[3] = s
s = txt[7]
s = sub(s, "for k in (2,):   # p2 only: p1 (temperature 0.2) already exists from G10\n", "for k in (3,):   # p3 only: p1 (temperature 0.2, G10) and p2 (temperature 1.0, G15) already exist\n")
s = sub(s, "for tag in ('smoke', 'p2'):", "for tag in ('smoke', 'p3'):")
s = sub(s, "G15 SUMMARY", "G16 SUMMARY")
txt[7] = s
txt[0] = ("# G16: pool round p3 = G15's configuration (temperature 1.0, top_p 0.95, top_k 64; B1 arm, 5-minute cap) with exact request/response capture\n\n"
          "Third attempt per pool task for the preference pair builder. Output: `g16/capture.jsonl.gz`, `g16/p3/task_results.jsonl`, `g16/g16_summary.json`.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
(OUT / "g16.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g15" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
meta.update({"id": "noobhunterjay/gemma-4-g16-pool-p3", "title": "Gemma 4 G16 Pool P3", "code_file": "g16.ipynb"})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
