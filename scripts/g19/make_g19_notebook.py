"""Generate kaggle_assets/g19/ : the 31B in the REAL SCORER REGIME. The scorer is sequential (one task at a time, ~37 tok/s per request) and our B2 submission allows 8 min / 28 tool calls /
150 s per command; every earlier local round ran 4 tasks at once (~24 tok/s per request) under a 5-minute cap, which probably overstates clock pressure and hides the call cap.
Arm b2seq = the submitted B2 agent (swe_scaffold_v3_b2, sampling temp 0.2 / top_p 0.95 / 8192 / thinking 4096) at concurrency 1 on the first 24 screen tasks (docs/g6_screen_tasks.json).
Measures: resolved, calls used (does the 28-call cap bind?), seconds per task, capped by time vs by calls, tokens per request.

Derived from the G2 notebook (kaggle_assets/g2/g2.ipynb: 31B serving, capture proxy, smoke gate) by exact-string edits, each asserted, mirroring scripts/g6/make_g6_notebook.py:
  - per arm: a 2-task smoke whose captured requests are CHECKED against the arm's settings (temperature, top_p, top_k, output cap, thinking_token_budget), then the 32 tasks
  - the competition mount is discovered (it is /kaggle/input/gemma-4-developer-agent/ now), the 4xL4 machine is asserted in the first seconds
  - a mechanism summary (time share in turns > 2,000 tokens; the 31B control share is ~23%) computed from the capture
Expected: ~15 min boot + 2-task smoke (~10 min) + 24 tasks x <= ~6 min = ~2.9 h wall = ~5.8 quota h. 230-minute watchdog.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC = json.loads((ROOT / "kaggle_assets" / "g2" / "g2.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g19"
OUT.mkdir(parents=True, exist_ok=True)
SCREEN = json.loads((ROOT / "docs" / "g6_screen_tasks.json").read_text(encoding="utf-8"))
assert len(SCREEN) == 32 and len(set(SCREEN)) == 32
SCREEN = SCREEN[:24]
ARMS_SRC = (ROOT / "scripts" / "g6" / "g6_arms.py").read_text(encoding="utf-8").replace("\r\n", "\n")
ARMS_TO_RUN = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("b2seq",)
assert set(ARMS_TO_RUN) <= {"b2seq"}


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
assert [c["cell_type"] for c in cells] == ["markdown", "code", "markdown", "code", "markdown", "code", "markdown", "code"]

# ---------------------------------------------------------------- SETUP (cell 3)
s = txt[3]
s = sub(s, "G2 = WORKING_DIR / 'g2'", "G2 = WORKING_DIR / 'g12'")
s = sub(s, "'g2_summary.json'", "'g12_summary.json'")
s = sub(s, "BUDGET_MIN = 235", "BUDGET_MIN = 190")
s = sub(s, "RESULTS = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "RESULTS = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "import torch\n"
        "_gpus = [(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]\n"
        "print('GPUs:', _gpus)\n"
        "assert len(_gpus) == 4 and all(c[0] >= 8 for _, c in _gpus), f'expected 4x L4 (compute capability >= 8), got {_gpus}'\n")
s = sub(s, "DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')\n",
        "DATA_DIR = next(p.parent for p in Path('/kaggle/input').rglob('tasks.jsonl') if (p.parent / 'sample_submission').is_dir() and (p.parent / 'snapshots').is_dir())\n")
i0 = s.index("SMOKE_IDS = random.Random(5)")
i1 = s.index("\n", i0)
s = s[:i0] + (f"SCREEN_IDS = {json.dumps(SCREEN)}\nassert all(i in by_id for i in SCREEN_IDS)\n"
              "SMOKE_IDS = random.Random(6).sample([i for i in ALL_IDS if i not in SCREEN_IDS and i.startswith(('fastapi_', 'rich_'))], 2)") + s[i1:]
s = sub(s, "'swe_scaffold_v2_b1' in p.read_text", "'swe_scaffold_v3_b2' in p.read_text")
s = sub(s, "expected exactly one B1 arm", "expected exactly one B2 arm")
s = sub(s, "G2 = WORKING_DIR / 'g12'", "G2 = WORKING_DIR / 'g19'")
s = sub(s, "'g12_summary.json'", "'g19_summary.json'")
s = sub(s, "BUDGET_MIN = 190", "BUDGET_MIN = 230")
txt[3] = s

# ---------------------------------------------------------------- RUN (cell 7)
s = txt[7]
s = sub(s, "def run_round(tag, ids):\n    proxy.round_tag = tag\n", "def run_round(tag, ids, arm_dir=None):\n    arm_dir = arm_dir or ARM\n    proxy.round_tag = tag\n")
s = sub(s, "submission_dir=ARM,", "submission_dir=arm_dir,")
s = sub(s, "concurrency=4,", "concurrency=1,")   # the scorer is sequential: one task at a time
s = sub(s, "**budget_of(ARM))", "**budget_of(arm_dir))")
cut = s.index("# ------------------------------------------------------------------ SMOKE")
TAIL = '''# ------------------------------------------------------------------ G12: config arms over the 32 screen tasks, each preceded by a checked 2-task smoke
exec(ARMS_SRC)   # defines ARMS, make_arm, request_matches
ARM_DIRS = {n: make_arm(ARM, WORKING_DIR / f'arm_{n}', **ARMS[n]) for n in __ARMS_TO_RUN__}
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

for k, arm in enumerate(__ARMS_TO_RUN__):
    elapsed = (time.time() - T_START) / 60
    if k > 0 and elapsed + 55 > BUDGET_MIN - 5:
        print(f'not starting {arm}: elapsed {elapsed:.0f} min + ~55 min would exceed the budget'); break
    rows = run_round(f'smoke_{arm}', SMOKE_IDS, ARM_DIRS[arm])
    if k == 0:
        st = proxy.stats(); print('capture after smoke:', st)
        assert rows and st['records'] >= 6 and st['errors'] == 0, f'capture/serving problem after smoke: {st}'
        try:
            first = next(CP.read_records(str(G2 / 'capture.jsonl.gz'))); rq = first['request']
            tk = {kk: rq[kk] for kk in ('model', 'messages', 'tools', 'chat_template_kwargs') if kk in rq}; tk['add_generation_prompt'] = True
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
    if rec['round'] not in __ARMS_TO_RUN__: continue
    sysm = next((m.get('content') for m in rec['request']['messages'] if m.get('role') == 'system'), '') or ''
    if not sysm.startswith(MAIN): continue
    ct = (rec['response'].get('usage') or {}).get('completion_tokens', 0); dt = rec['t1'] - rec['t0']
    m = mech[rec['round']]; m['turns'] += 1; m['sec'] += dt; m['ct'].append(ct)
    if ct > 2000: m['long_turns'] += 1; m['long_sec'] += dt
    if rec['response']['choices'][0].get('finish_reason') == 'length': m['length'] += 1
RESULTS['mechanism'] = {r: {'turns': m['turns'], 'mean_completion_tokens': round(_st.mean(m['ct']), 1), 'turns_over_2000': m['long_turns'], 'length_finishes': m['length'],
                            'time_share_in_turns_over_2000': round(m['long_sec'] / max(m['sec'], 1e-9), 3), 'mean_request_s': round(m['sec'] / max(m['turns'], 1), 2)} for r, m in mech.items()}
print('mechanism:', json.dumps(RESULTS['mechanism'], indent=1))
print('\\n===== G19 (31B, sequential scorer regime, B2) SUMMARY =====')
for tag in [f'{p}{a}' for a in __ARMS_TO_RUN__ for p in ('smoke_', '')]:
    if tag in RESULTS and 'n' in RESULTS[tag]: print(tag, {kk: v for kk, v in RESULTS[tag].items() if kk not in ('resolved_ids', 'proxy')})
print('capture:', proxy.stats(), '| file MB:', round((G2 / 'capture.jsonl.gz').stat().st_size / 1e6, 1))
RESULTS['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); RESULTS['capture'] = proxy.stats(); save_results()
_wd.cancel()'''
s = s[:cut] + TAIL.replace('__ARMS_TO_RUN__', repr(ARMS_TO_RUN))
s = "ARMS_SRC = " + json.dumps(ARMS_SRC) + "\n" + s
txt[7] = s

txt[0] = ("# G19 (31B): the B2 agent at concurrency 1 (scorer-like) on 24 competition tasks\n\n"
          "Output: `g19/g19_summary.json` (config gate, mechanism summary, per-arm results), `g19/capture.jsonl.gz`, one results dir per arm.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")

(OUT / "g19.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g2" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4" and "31b" in meta["model_sources"][0]
meta.update({"id": "noobhunterjay/gemma-4-g19-sequential-b2", "title": "Gemma 4 G19 Sequential B2", "code_file": "g19.ipynb"})
assert "noobhunterjay/gemma4-g2-arm" in meta["dataset_sources"]
meta["dataset_sources"] = [d.replace("gemma4-g2-arm", "gemma4-g19-arm") for d in meta["dataset_sources"]]
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT, "| arms", ARMS_TO_RUN)
