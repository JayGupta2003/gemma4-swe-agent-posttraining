"""Generate kaggle_assets/g14a/ : E4B (gemma-4-e4b-it-qat-w4a16-ct, no adapter) baseline over ALL 75 competition tasks with the B2 budget (8 min / 28 tool calls / 150 s command timeout) and the B1/B2 sampling
(temperature 0.2, thinking 4096, output 8192). Derived from the G13 generator. [G13 text follows:] temperature probe on the 31B over ALL 75 competition tasks, arm T4 = the B1 settings except temperature 1.0 / top_p 0.95 / top_k 64. Purpose: (1) does Gemma's usual sampling change the resolve rate vs temperature 0.2 (base 25 and 28 of 75, adapters 28 and 22), and
(2) does it decorrelate attempts, i.e. how many tasks become mixed (success in some attempts, not in others)? Gate for the preference run (docs/PREFERENCE_RUN.md).

Derived from the G2 notebook (kaggle_assets/g2/g2.ipynb: 31B serving, capture proxy, smoke gate) by exact-string edits, each asserted, mirroring scripts/g6/make_g6_notebook.py:
  - per arm: a 2-task smoke whose captured requests are CHECKED against the arm's settings (temperature, top_p, top_k, output cap, thinking_token_budget), then the 32 tasks
  - the competition mount is discovered (it is /kaggle/input/gemma-4-developer-agent/ now), the 4xL4 machine is asserted in the first seconds
  - a mechanism summary (time share in turns > 2,000 tokens; the 31B control share is ~23%) computed from the capture
Expected: ~15 min boot + 5 min smoke + ~90 min round = ~1.9 h wall = ~3.8 quota h. 150-minute watchdog.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC = json.loads((ROOT / "kaggle_assets" / "g2" / "g2.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g14a"
OUT.mkdir(parents=True, exist_ok=True)
ARMS_SRC = (ROOT / "scripts" / "g6" / "g6_arms.py").read_text(encoding="utf-8").replace("\r\n", "\n")
B2_EVAL = (ROOT / "submission_b2" / "eval_config.yaml").read_text(encoding="utf-8").replace("\r\n", "\n")   # 8 min / 28 tool calls / 150 s command timeout
ARMS_TO_RUN = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("e4b",)
assert set(ARMS_TO_RUN) <= {"t1", "t2", "t3", "t4", "e4b"}


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
assert [c["cell_type"] for c in cells] == ["markdown", "code", "markdown", "code", "markdown", "code", "markdown", "code"]

# ---------------------------------------------------------------- SETUP (cell 3)
s = txt[3]
s = sub(s, "G2 = WORKING_DIR / 'g2'", "G2 = WORKING_DIR / 'g14a'")
s = sub(s, "'g2_summary.json'", "'g14a_summary.json'")
s = sub(s, "BUDGET_MIN = 235", "BUDGET_MIN = 130")
s = sub(s, "RESULTS = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "RESULTS = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "import torch\n"
        "_gpus = [(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]\n"
        "print('GPUs:', _gpus)\n"
        "assert len(_gpus) == 4 and all(c[0] >= 8 for _, c in _gpus), f'expected 4x L4 (compute capability >= 8), got {_gpus}'\n")
M31, M4 = "gemma-4-31b-it-qat-w4a16-ct", "gemma-4-e4b-it-qat-w4a16-ct"
s = sub(s, "shutil.copytree(arm_dirs[0], ARM)\n",
        "shutil.copytree(arm_dirs[0], ARM)\n"
        "for _rel in ('agent.yaml', 'sub_agents/code_analyzer.yaml'):   # one variant for every agent and sub-agent\n"
        f"    _p = ARM / _rel; _t = _p.read_text(encoding='utf-8'); assert _t.count('{M31}') == 1, _rel; _p.write_text(_t.replace('{M31}', '{M4}'), encoding='utf-8', newline='\\n')\n"
        f"assert '{M31}' not in (ARM / 'agent.yaml').read_text()\n"
        "(ARM / 'eval_config.yaml').write_text(" + repr(B2_EVAL) + ", encoding='utf-8', newline='\\n')   # B2 budget: 8 min / 28 tool calls / 150 s command timeout\n")
s = sub(s, "DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')\n",
        "DATA_DIR = next(p.parent for p in Path('/kaggle/input').rglob('tasks.jsonl') if (p.parent / 'sample_submission').is_dir() and (p.parent / 'snapshots').is_dir())\n")
i0 = s.index("SMOKE_IDS = random.Random(5)")
i1 = s.index("\n", i0)
s = s[:i0] + ("SCREEN_IDS = random.Random(77).sample(ALL_IDS, len(ALL_IDS))   # all 75 competition tasks, shuffled\nassert len(SCREEN_IDS) == 75 and all(i in by_id for i in SCREEN_IDS)\n"
              "SMOKE_IDS = random.Random(6).sample(ALL_IDS, 2)") + s[i1:]
txt[3] = s


# ---------------------------------------------------------------- VLLM (cell 5): E4B, no adapter in the baseline
s = txt[5]
s = sub(s, "TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'", "TARGET_MODEL_NAME = 'gemma-4-e4b-it-qat-w4a16-ct'")
s = sub(s, "MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')", "MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2')\nassert MODEL_PATH.is_dir(), MODEL_PATH")
s = sub(s, "enable_lora=True, max_loras=8, max_lora_rank=128,", "enable_lora=False,")
s = sub(s, "startup_timeout=60 * 20)", "startup_timeout=60 * 12)")
txt[5] = s

# ---------------------------------------------------------------- RUN (cell 7)
s = txt[7]
s = sub(s, "def run_round(tag, ids):\n    proxy.round_tag = tag\n", "def run_round(tag, ids, arm_dir=None):\n    arm_dir = arm_dir or ARM\n    proxy.round_tag = tag\n")
s = sub(s, "submission_dir=ARM,", "submission_dir=arm_dir,")
s = sub(s, "**budget_of(ARM))", "**budget_of(arm_dir))")
cut = s.index("# ------------------------------------------------------------------ SMOKE")
TAIL = '''# ------------------------------------------------------------------ G14a: E4B baseline arm over all 75 tasks, each preceded by a checked 2-task smoke
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
    if k > 0 and elapsed + 100 > BUDGET_MIN - 5:
        print(f'not starting {arm}: elapsed {elapsed:.0f} min + ~100 min would exceed the budget'); break
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
print('\\n===== G14a (E4B baseline) SUMMARY =====')
for tag in [f'{p}{a}' for a in __ARMS_TO_RUN__ for p in ('smoke_', '')]:
    if tag in RESULTS and 'n' in RESULTS[tag]: print(tag, {kk: v for kk, v in RESULTS[tag].items() if kk not in ('resolved_ids', 'proxy')})
print('capture:', proxy.stats(), '| file MB:', round((G2 / 'capture.jsonl.gz').stat().st_size / 1e6, 1))
RESULTS['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); RESULTS['capture'] = proxy.stats(); save_results()
_wd.cancel()'''
s = s[:cut] + TAIL.replace('__ARMS_TO_RUN__', repr(ARMS_TO_RUN))
s = "ARMS_SRC = " + json.dumps(ARMS_SRC) + "\n" + s
txt[7] = s

txt[0] = ("# G13 (31B): temperature probe, arm " + ", ".join(ARMS_TO_RUN) + " on all 75 competition tasks\n\n"
          "Base reference at temperature 0.2: r1 25/75, r2 28/75 (+ adapters 28, 22). Output: `g13/g13_summary.json` (config gates + mechanism summary), `g13/capture.jsonl.gz`, `g13/t4/`.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")

(OUT / "g14a.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g2" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
meta.update({"id": "noobhunterjay/gemma-4-g14a-e4b-base", "title": "Gemma 4 G14a E4B Base", "code_file": "g14a.ipynb", "model_sources": ["google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT, "| arms", ARMS_TO_RUN)
