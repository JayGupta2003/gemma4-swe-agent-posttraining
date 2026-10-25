"""Generate kaggle_assets/g15/ : G15 = pool round p2, the SAME configuration as G10's p1 (31B, B1 arm, 5-minute cap, capture proxy, 171-task pool, same stub graphs and task dependencies)
except the sampling: temperature 1.0 / top_p 0.95 / top_k 64 (arm T4 of scripts/g6/g6_arms.py: output cap 8192 and thinking budget 4096 unchanged).

Purpose (docs/PREFERENCE_RUN.md, gate of the preference run): p1 (temperature 0.2) vs p2 (temperature 1.0) on the same tasks under an otherwise identical config measures the
number of tasks with success in one round and failure in the other, i.e. whether temperature 1.0 gives enough contrast for preference pairs. Gate: >= 25 such tasks.
Derived from the generated G10 notebook by exact-string edits, each asserted. Cost = p1: ~7.4 quota h (300-minute watchdog).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g10" / "g10.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g15"
OUT.mkdir(parents=True, exist_ok=True)
ARMS_SRC = (ROOT / "scripts" / "g6" / "g6_arms.py").read_text(encoding="utf-8").replace("\r\n", "\n")
assert '"t4": dict(temperature=1.0, top_p=0.95, top_k=64, max_output_tokens=8192, thinking_budget=4096)' in ARMS_SRC


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}

# ---- setup: output directory, summary file, sampling of the arm
s = txt[3]
s = sub(s, "G2 = WORKING_DIR / 'g10'", "G2 = WORKING_DIR / 'g15'")
s = sub(s, "'g10_summary.json'", "'g15_summary.json'")
s = sub(s, "ARM = WORKING_DIR / 'arm'; shutil.rmtree(ARM, ignore_errors=True); shutil.copytree(arm_dirs[0], ARM)\n",
        "ARM = WORKING_DIR / 'arm'; shutil.rmtree(ARM, ignore_errors=True); shutil.copytree(arm_dirs[0], ARM)\n"
        "ARMS_NS = {}; exec(" + repr(ARMS_SRC) + ", ARMS_NS)   # ARMS, sampling_yaml, request_matches\n"
        "SAMPLING = ARMS_NS['ARMS']['t4']; assert SAMPLING == dict(temperature=1.0, top_p=0.95, top_k=64, max_output_tokens=8192, thinking_budget=4096), SAMPLING\n"
        "(ARM / 'configs' / 'sampling.yaml').write_text(ARMS_NS['sampling_yaml'](**SAMPLING), encoding='utf-8', newline=chr(10))\n"
        "print('sampling.yaml of the arm:', (ARM / 'configs' / 'sampling.yaml').read_text().replace(chr(10), ' | '))\n")
txt[3] = s

# ---- run: config gate on the smoke capture, then ONE full round tagged p2
s = txt[7]
s = sub(s, "RESULTS['smoke_gate'] = gate\n",
        "MAIN_PFX = 'You are an expert autonomous software engineer'\n"
        "_reqs = [r['request'] for r in CP.read_records(str(G2 / 'capture.jsonl.gz')) if r['round'] == 'smoke' and (next((m.get('content') for m in r['request']['messages'] if m.get('role') == 'system'), '') or '').startswith(MAIN_PFX)]\n"
        "_bad = [m for r in _reqs for m in ARMS_NS['request_matches'](r, SAMPLING)]\n"
        "gate['sampling settings reach vLLM (T1.0 / top_p / top_k / caps)'] = bool(_reqs) and not _bad\n"
        "RESULTS['config_gate'] = {'main_agent_requests': len(_reqs), 'mismatches': _bad[:5]}; print('config gate:', RESULTS['config_gate'])\n"
        "RESULTS['smoke_gate'] = gate\n")
s = sub(s, "for k in (1, 2):\n", "for k in (2,):   # p2 only: p1 (temperature 0.2) already exists from G10\n")
s = sub(s, "for tag in ('smoke', 'p1', 'p2'):", "for tag in ('smoke', 'p2'):")
s = sub(s, "G10 SUMMARY", "G15 SUMMARY")
txt[7] = s

txt[0] = ("# G15: pool round p2 = G10's p1 configuration at temperature 1.0 (top_p 0.95, top_k 64), with exact request/response capture\n\n"
          "Gate for the preference run: tasks with success in exactly one of p1 and p2 (>= 25). Output: `g15/capture.jsonl.gz`, `g15/p2/task_results.jsonl`, `g15/g15_summary.json`.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
(OUT / "g15.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g10" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
meta.update({"id": "noobhunterjay/gemma-4-g15-pool-p2", "title": "Gemma 4 G15 Pool P2", "code_file": "g15.ipynb"})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
