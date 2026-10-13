"""Generate kaggle_assets/g6_12b/ : config screen on the 12B variant (arms T1, T2 of scripts/g6/g6_arms.py) over the 32 screen tasks (docs/g6_screen_tasks.json).

Derived from the G5 notebook (kaggle_assets/g5/g5.ipynb: 12B serving with the staff override + guarded loader patch, capture proxy, decode probe) by exact-string
edits, each asserted. Per arm: a 2-task smoke round whose captured requests are CHECKED against the arm's settings (fail fast: a config that does not reach vLLM
would silently waste the quota), then one round over the 32 tasks. A mechanism summary (time share in long turns, capped share, calls per task) is computed
in the notebook from the capture so the result is readable straight from g6_summary.json.

Expected cost: boot ~10 min + 2 x (2-task smoke ~5 min + <= 42 min round) ~ 1.7 h wall ~ 3.3 quota h (L4x4 bills 2x). 150-minute watchdog.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC = json.loads((ROOT / "kaggle_assets" / "g5" / "g5.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g6_12b"
OUT.mkdir(parents=True, exist_ok=True)
SCREEN = json.loads((ROOT / "docs" / "g6_screen_tasks.json").read_text(encoding="utf-8"))
assert len(SCREEN) == 32 and len(set(SCREEN)) == 32
ARMS_SRC = (HERE / "g6_arms.py").read_text(encoding="utf-8").replace("\r\n", "\n")
ARMS_TO_RUN = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("t1", "t2")   # e.g. `make_g6_notebook.py t1` for a T1-only run
assert set(ARMS_TO_RUN) <= {"t1", "t2"}


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}

# ---------------------------------------------------------------- SETUP (cell 3)
s = txt[3]
s = sub(s, "G2 = WORKING_DIR / 'g5'", "G2 = WORKING_DIR / 'g6'")
s = sub(s, "'g5_summary.json'", "'g6_summary.json'")
s = sub(s, "BUDGET_MIN = 160", "BUDGET_MIN = 150")
i0 = s.index("SMOKE_IDS = random.Random(5)")
i1 = s.index("\n", i0)
s = s[:i0] + (f"SCREEN_IDS = {json.dumps(SCREEN)}\nassert all(i in by_id for i in SCREEN_IDS)\n"
              "SMOKE_IDS = random.Random(6).sample([i for i in ALL_IDS if i not in SCREEN_IDS and i.startswith(('fastapi_', 'rich_'))], 2)") + s[i1:]
txt[3] = s

# ---------------------------------------------------------------- RUN (cell 7)
s = txt[7]
s = sub(s, "def run_round(tag, ids):\n    proxy.round_tag = tag\n", "def run_round(tag, ids, arm_dir=None):\n    arm_dir = arm_dir or ARM\n    proxy.round_tag = tag\n")
s = sub(s, "submission_dir=ARM,", "submission_dir=arm_dir,")
s = sub(s, "**budget_of(ARM))", "**budget_of(arm_dir))")
cut = s.index("# ------------------------------------------------------------------ SMOKE")
TAIL = '''# ------------------------------------------------------------------ G6: two config arms over the 32 screen tasks, each preceded by a checked 2-task smoke
exec(ARMS_SRC)   # defines ARMS, make_arm, request_matches
ARM_DIRS = {n: make_arm(ARM, WORKING_DIR / f'arm_{n}', **kw) for n, kw in ARMS.items()}
for n, d in ARM_DIRS.items(): print(n, ARMS[n], '|', (d / 'configs/sampling.yaml').read_text().replace(chr(10), ' | '))
MAIN = 'You are an expert autonomous software engineer'

def config_gate(tag, arm):
    reqs = []
    for rec in CP.read_records(str(G2 / 'capture.jsonl.gz')):
        if rec['round'] != tag: continue
        sysm = next((m.get('content') for m in rec['request']['messages'] if m.get('role') == 'system'), '') or ''
        if sysm.startswith(MAIN): reqs.append(rec['request'])
    bad = [m for r in reqs for m in request_matches(r, ARMS[arm])]
    RESULTS['config_gate_' + tag] = {'main_agent_requests': len(reqs), 'mismatches': bad[:5], 'n_mismatches': len(bad)}
    save_results(); print('config gate', tag, RESULTS['config_gate_' + tag])
    if not reqs or bad:
        raise RuntimeError(f'CONFIG GATE FAILED for arm {arm}: {len(reqs)} requests, mismatches {bad[:3]}')

for arm in __ARMS_TO_RUN__:
    elapsed = (time.time() - T_START) / 60
    if arm == 't2' and elapsed + 50 > BUDGET_MIN - 5:
        print(f'not starting t2: elapsed {elapsed:.0f} min + ~50 min would exceed the budget'); break
    rows = run_round(f'smoke_{arm}', SMOKE_IDS, ARM_DIRS[arm])
    if arm == 't1':
        st = proxy.stats(); print('capture after smoke:', st)
        assert rows and st['records'] >= 6 and st['errors'] == 0, f'capture/serving problem after smoke: {st}'
        try:
            first = next(CP.read_records(str(G2 / 'capture.jsonl.gz'))); rq = first['request']
            tk = {k: rq[k] for k in ('model', 'messages', 'tools', 'chat_template_kwargs') if k in rq}; tk['add_generation_prompt'] = True
            tok = requests.post(VLLM_URL + '/tokenize', json=tk, timeout=300).json().get('tokens')
            got = first['response'].get('prompt_token_ids')
            RESULTS['token_id_crosscheck'] = {'tokenize_len': len(tok or []), 'captured_len': len(got or []), 'identical': tok == got}
        except Exception:
            RESULTS['token_id_crosscheck'] = {'error': traceback.format_exc()[-600:]}
        print('cross-check:', RESULTS['token_id_crosscheck'])
    config_gate(f'smoke_{arm}', arm)
    run_round(arm, SCREEN_IDS, ARM_DIRS[arm])

# ------------------------------------------------------------------ mechanism summary from the capture (main-agent requests only)
import collections, statistics as _st
mech = collections.defaultdict(lambda: {'turns': 0, 'sec': 0.0, 'long_sec': 0.0, 'long_turns': 0, 'ct': [], 'length': 0})
for rec in CP.read_records(str(G2 / 'capture.jsonl.gz')):
    if rec['round'] not in ('t1', 't2'): continue
    sysm = next((m.get('content') for m in rec['request']['messages'] if m.get('role') == 'system'), '') or ''
    if not sysm.startswith(MAIN): continue
    ct = (rec['response'].get('usage') or {}).get('completion_tokens', 0); dt = rec['t1'] - rec['t0']
    m = mech[rec['round']]; m['turns'] += 1; m['sec'] += dt; m['ct'].append(ct)
    if ct > 2000: m['long_turns'] += 1; m['long_sec'] += dt
    if rec['response']['choices'][0].get('finish_reason') == 'length': m['length'] += 1
RESULTS['mechanism'] = {r: {'turns': m['turns'], 'mean_completion_tokens': round(_st.mean(m['ct']), 1), 'turns_over_2000': m['long_turns'], 'length_finishes': m['length'],
                            'time_share_in_turns_over_2000': round(m['long_sec'] / max(m['sec'], 1e-9), 3)} for r, m in mech.items()}
print('mechanism:', json.dumps(RESULTS['mechanism'], indent=1))
print('\\n===== G6 (12B config screen) SUMMARY =====')
for tag in ('smoke_t1', 't1', 'smoke_t2', 't2'):
    if tag in RESULTS and 'n' in RESULTS[tag]: print(tag, {k: v for k, v in RESULTS[tag].items() if k not in ('resolved_ids', 'proxy')})
print('capture:', proxy.stats(), '| file MB:', round((G2 / 'capture.jsonl.gz').stat().st_size / 1e6, 1))
RESULTS['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); RESULTS['capture'] = proxy.stats(); save_results()
_wd.cancel()'''
s = s[:cut] + TAIL.replace('__ARMS_TO_RUN__', repr(ARMS_TO_RUN))
# embed the arms source as a module-level string defined at the top of the cell
s = "ARMS_SRC = " + json.dumps(ARMS_SRC) + "\n" + s
txt[7] = s

txt[0] = ("# G6 (12B): config screen — thinking budget / output cap / sampling (arms T1, T2) on 32 tasks\n\n"
          "Control = G5 round b12 restricted to the same 32 tasks. Output: `g6/g6_summary.json` (incl. config gates and mechanism summary), `g6/capture.jsonl.gz`, `g6/t1`, `g6/t2`.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")

(OUT / "g6.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g5" / "kernel-metadata.json").read_text(encoding="utf-8"))
meta.update({"id": "noobhunterjay/gemma-4-g6-12b-screen", "title": "Gemma 4 G6 12B Screen", "code_file": "g6.ipynb"})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
