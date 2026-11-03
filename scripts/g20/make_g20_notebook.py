"""Generate kaggle_assets/g20/ : G20 = data stage of on-policy distillation 31B -> E4B (STRATEGY.md, 2026-11-04 review, suggestion A).

Stage 1: the deployed E4B arm (submission_e4b: B2 prompt and budget 8 min / 28 tool calls / 150 s, temperature 0.2, thinking 4096, agent swe_scaffold_v5_e4b_base,
no adapter) runs the 171-task pool (competition tasks never used) with the capture proxy, in 6 chunks so the round can stop early and leave time for stage 2.
Stage 2: the E4B server is stopped, the 31B (w4a16, the scorer's checkpoint) is started from the harness's own build_cmd() plus scoring flags, and scores every
main-agent leaf (scripts/g20/opd_score.py) with prompt_logprobs (actual-token log p + top-8). The 31B and E4B share tokenizer and chat template (60/60 exact).
Training (later, next quota): per-token reverse-KL / log-ratio advantage on the student's own tokens.

Derived from the generated G16 notebook (pool round + capture) by exact-string edits, each asserted; the E4B model edits mirror scripts/g14/make_g14a_notebook.py.
Budget: watchdog 195 min (6.5 quota h of the 6.83 left before the 2026-11-07 refresh); no new chunk after 125 min; scoring stops 8 min before the watchdog.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g16" / "g16.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g20"
OUT.mkdir(parents=True, exist_ok=True)
OPD = (ROOT / "scripts" / "g20" / "opd_score.py").read_text(encoding="utf-8").replace("\r\n", "\n")
M31, M4 = "gemma-4-31b-it-qat-w4a16-ct", "gemma-4-e4b-it-qat-w4a16-ct"


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
assert [c["cell_type"] for c in cells] == ["markdown", "code", "markdown", "code", "markdown", "code", "markdown", "code"]

# ---------------------------------------------------------------- SETUP (cell 3)
s = txt[3]
s = sub(s, "G2 = WORKING_DIR / 'g16'", "G2 = WORKING_DIR / 'g20'")
s = sub(s, "'g16_summary.json'", "'g20_summary.json'")
s = sub(s, "BUDGET_MIN = 300", "BUDGET_MIN = 195")
s = sub(s, "sys.path.insert(0, str(CODE))\n", "(CODE / 'opd_score.py').write_text(" + json.dumps(OPD) + ", encoding='utf-8')\nsys.path.insert(0, str(CODE))\n")
s = sub(s, "if 'swe_scaffold_v2_b1' in p.read_text(encoding='utf-8')]\nassert len(arm_dirs) == 1, f'expected exactly one B1 arm, found {arm_dirs}'",
        "if 'swe_scaffold_v5_e4b_base' in p.read_text(encoding='utf-8')]\nassert len(arm_dirs) == 1, f'expected exactly one E4B arm, found {arm_dirs}'")
s = sub(s, "SAMPLING = ARMS_NS['ARMS']['t4']; assert SAMPLING == dict(temperature=1.0, top_p=0.95, top_k=64, max_output_tokens=8192, thinking_budget=4096), SAMPLING\n"
           "(ARM / 'configs' / 'sampling.yaml').write_text(ARMS_NS['sampling_yaml'](**SAMPLING), encoding='utf-8', newline=chr(10))\n",
        "SAMPLING = ARMS_NS['ARMS']['e4b']   # the deployed E4B arm is used UNCHANGED (on-policy for the submission); the config gate checks its requests against this\n"
        "_ev = (ARM / 'eval_config.yaml').read_text(); assert 'max_tool_calls: 28' in _ev and 'max_time_minutes: 8' in _ev, _ev\n"
        f"assert (ARM / 'agent.yaml').read_text().count('{M4}') == 1 and '{M31}' not in (ARM / 'agent.yaml').read_text()\n")
txt[3] = s

# ---------------------------------------------------------------- VLLM (cell 5): the E4B student, no adapter
s = txt[5]
s = sub(s, f"TARGET_MODEL_NAME = '{M31}'", f"TARGET_MODEL_NAME = '{M4}'")
s = sub(s, f"MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/{M31}/2')", f"MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/{M4}/2')\nassert MODEL_PATH.is_dir(), MODEL_PATH")
s = sub(s, "enable_lora=True, max_loras=8, max_lora_rank=128,", "enable_lora=False,")
s = sub(s, "startup_timeout=60 * 20)", "startup_timeout=60 * 12)")
txt[5] = s

# ---------------------------------------------------------------- RUN (cell 7)
s = txt[7]
s = sub(s, "gate['sampling settings reach vLLM (T1.0 / top_p / top_k / caps)']", "gate['E4B arm sampling settings reach vLLM']")
cut = s.index("# ------------------------------------------------------------------ full rounds over all 75 tasks")
TAIL = '''# ------------------------------------------------------------------ G20 stage 1: the student over the 171-task pool, in chunks (no new chunk after ROUND_STOP_MIN)
ROUND_STOP_MIN = 125
ids = list(ALL_IDS); random.Random(120).shuffle(ids)
CHUNKS = [ids[i::6] for i in range(6)]
for k, ch in enumerate(CHUNKS):
    elapsed_min = (time.time() - T_START) / 60
    if elapsed_min > ROUND_STOP_MIN:
        print(f'not starting chunk {k}: elapsed {elapsed_min:.0f} min > {ROUND_STOP_MIN}'); break
    run_round(f'e1_{k}', ch)
_rs = [RESULTS[t] for t in RESULTS if t.startswith('e1_') and 'n' in RESULTS[t]]
RESULTS['e1_total'] = {'n': sum(r['n'] for r in _rs), 'resolved': sum(r['resolved'] for r in _rs), 'patch_written': sum(r['patch_written'] for r in _rs),
                       'time_cap_hit': sum(r['time_cap_hit'] for r in _rs), 'chunks': len(_rs), 'minutes': round((time.time() - T_START) / 60, 1)}
save_results(); print('student round:', RESULTS['e1_total'], '| capture:', proxy.stats(), flush=True)

# ------------------------------------------------------------------ G20 stage 2: the 31B teacher scores the student's leaves (prompt_logprobs)
import subprocess
import opd_score as O
proxy.stop()
try:
    server_instance.stop()
except Exception as e:
    print('server_instance.stop() failed:', e)
def gpu_used():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]
for _ in range(36):
    if max(gpu_used()) < 1500: break
    time.sleep(5)
else:
    subprocess.run(['pkill', '-f', 'vllm'], check=False); time.sleep(30)
RESULTS['gpu_mib_after_student'] = gpu_used(); print('GPU MiB after stopping the student:', RESULTS['gpu_mib_after_student'], flush=True)
LEAVES = O.leaves(str(G2 / 'capture.jsonl.gz'), 'e1_')
RESULTS['leaves'] = {'n': len(LEAVES), 'tokens': sum(len(l['ids']) for l in LEAVES), 'max_len': max((len(l['ids']) for l in LEAVES), default=0)}
save_results(); print('leaves:', RESULTS['leaves'], flush=True)
K = 8
T_MODEL = Path('/kaggle/input/models/google/gemma-4/other/__M31__/2'); assert T_MODEL.is_dir(), T_MODEL
tcfg = VllmConfig(model=str(T_MODEL), port=8010, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4', max_model_len=32768, dtype='bfloat16',
                  gpu_memory_utilization=0.80, enable_auto_tool_choice=True, enable_lora=False, tensor_parallel_size=4, startup_timeout=60 * 15)
TCMD = list(VllmServer(tcfg, adapter_manifest=None).build_cmd()) + ['--max-logprobs', '20', '--max-num-batched-tokens', '2048', '--max-model-len', '33000']
RESULTS['teacher_cmd'] = ' '.join(map(str, TCMD)); save_results(); print('teacher cmd:', RESULTS['teacher_cmd'], flush=True)
TURL = 'http://127.0.0.1:8010'
tlog = open(G2 / 'teacher_vllm.log', 'w')
tproc = subprocess.Popen([str(c) for c in TCMD], stdout=tlog, stderr=subprocess.STDOUT)
t_boot = time.time(); ready = False
while time.time() - t_boot < 15 * 60 and tproc.poll() is None:
    try:
        if requests.get(TURL + '/health', timeout=5).status_code == 200: ready = True; break
    except Exception:
        pass
    time.sleep(10)
RESULTS['teacher_boot_s'] = round(time.time() - t_boot); RESULTS['teacher_ready'] = ready; save_results()
STORE = O.Store(K)
def _save():
    STORE.save(G2 / 'teacher_scores.npz', G2 / 'teacher_meta.json'); RESULTS['teacher'] = O.summary(STORE); save_results()
if not ready:
    RESULTS['teacher_error'] = 'teacher vLLM did not become ready: ' + (G2 / 'teacher_vllm.log').read_text(errors='ignore')[-3000:]; save_results(); print(RESULTS['teacher_error'])
elif LEAVES:
    TMODEL = requests.get(TURL + '/v1/models', timeout=30).json()['data'][0]['id']
    def job(leaf):
        t = time.time()
        act, ti, tl = O.parse_prompt_logprobs(O.score_leaf(TURL, TMODEL, leaf['ids'], K), leaf['ids'], K)
        return act, ti, tl, time.time() - t
    try:   # probe on the shortest leaf: shape and speed before the full pass
        a, ti, tl, sec = job(LEAVES[0]); STORE.add(LEAVES[0], a, ti, tl, sec); _save()
        RESULTS['teacher_probe'] = {'tokens': len(LEAVES[0]['ids']), 'seconds': round(sec, 1), 'tok_per_s': round(len(LEAVES[0]['ids']) / max(sec, 1e-6)), 'act_lp_head': [float(x) for x in a[1:6]],
                                    'top_ids_head': ti[1].tolist()}
        print('teacher probe:', RESULTS['teacher_probe'], flush=True)
        n_ok, errs = O.score_all(LEAVES[1:], job, STORE, lambda: (time.time() - T_START) / 60 > BUDGET_MIN - 8, _save, workers=2, every=10)
        RESULTS['teacher_pass'] = {'scored': n_ok + 1, 'of': len(LEAVES), 'errors': len(errs), 'first_errors': errs[:3], 'minutes': round((time.time() - T_START) / 60, 1)}
    except Exception:
        RESULTS['teacher_error'] = traceback.format_exc()[-2500:]
    _save(); print('teacher pass:', RESULTS.get('teacher_pass'), RESULTS.get('teacher_error', ''), '| summary:', RESULTS.get('teacher'), flush=True)
tproc.terminate()
print('\\n===== G20 SUMMARY =====')
for tag in ['smoke'] + [f'e1_{k}' for k in range(6)]:
    if tag in RESULTS and 'n' in RESULTS[tag]: print(tag, {kk: v for kk, v in RESULTS[tag].items() if kk not in ('resolved_ids', 'proxy')})
print('student total:', RESULTS.get('e1_total'), '| leaves:', RESULTS.get('leaves'), '| teacher:', RESULTS.get('teacher_pass'), RESULTS.get('teacher'))
print('capture MB:', round((G2 / 'capture.jsonl.gz').stat().st_size / 1e6, 1))
RESULTS['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); save_results()
_wd.cancel()'''
s = s[:cut] + TAIL.replace("__M31__", M31)
txt[7] = s

txt[0] = ("# G20: on-policy distillation data stage (student E4B rollouts on the 171-task pool + 31B teacher scoring)\n\n"
          "Stage 1: deployed E4B arm with capture, 6 chunks. Stage 2: 31B scores every main-agent leaf with prompt_logprobs (top-8). "
          "Output: `g20/capture.jsonl.gz`, `g20/teacher_scores.npz`, `g20/teacher_meta.json`, `g20/e1_*/`, `g20/g20_summary.json`.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
(OUT / "g20.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g16" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
meta.update({"id": "noobhunterjay/gemma-4-g20-opd-data", "title": "Gemma 4 G20 OPD Data", "code_file": "g20.ipynb",
             "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g18-data"],
             "model_sources": ["google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2", "google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
