"""Generate kaggle_assets/g24a/ : G24a = rollout stage of E4B self-RFT (rejection-sampling fine-tuning / expert iteration).

Idea: E4B already solves many more distinct tasks than it solves in any single round (base rounds on the 75 competition tasks: 10, 7, 8, 11 resolved,
17 distinct). Sampling it several times on the training pool, keeping only the attempts that the hidden tests verify, and fine-tuning it on its OWN
successful tokens pushes probability mass toward the behaviour that already works (pass@k -> pass@1), with no imported style (the failure mode of
G14b, G18, G21-G23, which all trained on another model's behaviour).

G24a = three more passes of the DEPLOYED E4B arm (submission_e4b: B2 prompt/budget, temperature 0.2, thinking 4096, no adapter; on-policy for the
submission) over the 171-task pool with the capture proxy. G20 stage 1 was pass 1 (25/171 resolved), so G24a gives 4 samples per task in total.
Each pass uses a different shuffle seed and runs in 6 chunks so the watchdog can stop between chunks.

Derived from the generated G20 notebook (kaggle_assets/g20/g20.ipynb) by exact-string edits, each asserted; the teacher stage is removed.
Budget: watchdog 300 min (~10 quota h); no new chunk after 280 min (a chunk takes 10-14 min).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g20" / "g20.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g24a"
OUT.mkdir(parents=True, exist_ok=True)


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
assert [c["cell_type"] for c in cells] == ["markdown", "code"] * 4

s = txt[3]
s = sub(s, "G2 = WORKING_DIR / 'g20'", "G2 = WORKING_DIR / 'g24a'")
s = sub(s, "'g20_summary.json'", "'g24a_summary.json'")
s = sub(s, "BUDGET_MIN = 195", "BUDGET_MIN = 300")
txt[3] = s

s = txt[7]
cut = s.index("# ------------------------------------------------------------------ G20 stage 1")
TAIL = '''# ------------------------------------------------------------------ G24a: passes 2-4 of the deployed E4B over the 171-task pool (pass 1 = G20)
CHUNK_STOP_MIN = 280
PASSES = {'r2': 241, 'r3': 242, 'r4': 243}
stopped = False
for p, seed in PASSES.items():
    ids = list(ALL_IDS); random.Random(seed).shuffle(ids)
    for k, ch in enumerate([ids[i::6] for i in range(6)]):
        elapsed_min = (time.time() - T_START) / 60
        if elapsed_min > CHUNK_STOP_MIN:
            print(f'not starting {p}_{k}: elapsed {elapsed_min:.0f} min > {CHUNK_STOP_MIN}'); stopped = True; break
        run_round(f'{p}_{k}', ch)
    _rs = [RESULTS[t] for t in RESULTS if t.startswith(p + '_') and 'n' in RESULTS[t]]
    RESULTS[p + '_total'] = {'n': sum(r['n'] for r in _rs), 'resolved': sum(r['resolved'] for r in _rs), 'patch_written': sum(r['patch_written'] for r in _rs),
                             'time_cap_hit': sum(r['time_cap_hit'] for r in _rs), 'chunks': len(_rs), 'minutes': round((time.time() - T_START) / 60, 1)}
    save_results(); print(p, 'pass:', RESULTS[p + '_total'], '| capture:', proxy.stats(), flush=True)
    if stopped: break
proxy.stop()
print('\\n===== G24a SUMMARY =====')
for p in PASSES:
    print(p, RESULTS.get(p + '_total'))
_solved = set()
for t, r in RESULTS.items():
    if isinstance(r, dict) and r.get('resolved_ids') is not None and t[:2] in PASSES: _solved |= set(r['resolved_ids'])
RESULTS['distinct_tasks_solved_g24a'] = len(_solved)
print('distinct tasks solved in G24a:', len(_solved), '| capture MB:', round((G2 / 'capture.jsonl.gz').stat().st_size / 1e6, 1))
RESULTS['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); save_results()
_wd.cancel()'''
txt[7] = s[:cut] + TAIL

txt[0] = ("# G24a: E4B self-RFT rollouts (passes 2-4 of the deployed E4B arm over the 171-task pool, with capture)\n\n"
          "Pass 1 was G20 stage 1. Output: `g24a/capture.jsonl.gz`, `g24a/r{2,3,4}_*/`, `g24a/g24a_summary.json`.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
(OUT / "g24a.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g20" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
meta.update({"id": "noobhunterjay/gemma-4-g24a-e4b-rft-rollouts", "title": "Gemma 4 G24a E4B RFT Rollouts", "code_file": "g24a.ipynb",
             "model_sources": ["google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
