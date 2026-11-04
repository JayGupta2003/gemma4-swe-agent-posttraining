"""Generate kaggle_assets/g20b/ : G20b = the teacher-scoring stage of G20, rerun on its own.

G20 stage 2 crashed on the first request: vLLM computes prompt logprobs as log_softmax over [max_num_batched_tokens x 262,144] fp32 = 2 GiB per GPU for a
2,048-token chunk, with 1.95 GiB free (teacher_vllm.log: torch.OutOfMemoryError in sampler.compute_logprobs). Fix: 512-token chunks (~0.5 GiB), --enforce-eager (no
CUDA-graph memory; a prefill-only job gains nothing from graphs), gpu_memory_utilization 0.72 (weights 5.96 GiB + ~8 GiB KV per GPU, ~5 GiB headroom), max 8 sequences.
Input: G20's capture (kernel source noobhunterjay/gemma-4-g20-opd-data); the 244 main-agent leaves (2,930,160 tokens) are rebuilt with scripts/g20/opd_score.py,
the shortest first as a probe, the rest in a seeded random order so that a time-limited partial pass is not biased towards short leaves.
Output: g20b/teacher_scores.npz + teacher_meta.json (saved every 10 leaves), g20b/g20b_summary.json, g20b/teacher_vllm.log. Watchdog 85 min (<= ~2.9 quota h).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G20 = json.loads((ROOT / "kaggle_assets" / "g20" / "g20.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g20b"
OUT.mkdir(parents=True, exist_ok=True)
OPD = (ROOT / "scripts" / "g20" / "opd_score.py").read_text(encoding="utf-8").replace("\r\n", "\n")
INSTALL = "".join(G20["cells"][1]["source"])
assert "pip" in INSTALL and "WHEELHOUSE_DIR" in INSTALL

MAIN = '''import json, os, random, subprocess, sys, threading, time, traceback
from pathlib import Path
import requests, torch
W = Path('/kaggle/working/g20b'); W.mkdir(parents=True, exist_ok=True); CODE = W / 'code'; CODE.mkdir(exist_ok=True)
RESULTS = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}
def save_results(): (W / 'g20b_summary.json').write_text(json.dumps(RESULTS, indent=1, default=str))
_gpus = [(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
print('GPUs:', _gpus); assert len(_gpus) == 4 and all(c[0] >= 8 for _, c in _gpus), _gpus
BUDGET_MIN = 85; T_START = time.time()
def _watchdog():
    RESULTS['watchdog'] = f'fired after {BUDGET_MIN} min'; save_results(); os._exit(3)
_wd = threading.Timer(BUDGET_MIN * 60, _watchdog); _wd.daemon = True; _wd.start()
(CODE / 'opd_score.py').write_text(__OPD__, encoding='utf-8'); sys.path.insert(0, str(CODE))
import opd_score as O
caps = [p for p in Path('/kaggle/input').rglob('capture.jsonl.gz') if p.parent.name == 'g20']
assert len(caps) == 1, caps
LEAVES = O.leaves(str(caps[0]), 'e1_')
RESULTS['leaves'] = {'n': len(LEAVES), 'tokens': sum(len(l['ids']) for l in LEAVES), 'max_len': max(len(l['ids']) for l in LEAVES)}
print('leaves:', RESULTS['leaves']); assert RESULTS['leaves']['n'] == 244 and RESULTS['leaves']['tokens'] == 2930160, RESULTS['leaves']
T_MODEL = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2'); assert (T_MODEL / 'config.json').is_file(), T_MODEL   # path used by G20
K = 8
TCMD = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(T_MODEL), '--host', '127.0.0.1', '--port', '8010', '--max-model-len', '33000',
        '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.72', '--tensor-parallel-size', '4', '--max-logprobs', '20', '--max-num-batched-tokens', '512',
        '--max-num-seqs', '8', '--enforce-eager']
RESULTS['teacher_cmd'] = ' '.join(TCMD); save_results(); print(RESULTS['teacher_cmd'], flush=True)
TURL = 'http://127.0.0.1:8010'
tproc = subprocess.Popen(TCMD, stdout=open(W / 'teacher_vllm.log', 'w'), stderr=subprocess.STDOUT)
t_boot = time.time(); ready = False
while time.time() - t_boot < 15 * 60 and tproc.poll() is None:
    try:
        if requests.get(TURL + '/health', timeout=5).status_code == 200: ready = True; break
    except Exception:
        pass
    time.sleep(10)
RESULTS['teacher_boot_s'] = round(time.time() - t_boot); RESULTS['teacher_ready'] = ready; save_results(); print('teacher ready:', ready, RESULTS['teacher_boot_s'], 's', flush=True)
STORE = O.Store(K)
def _save():
    STORE.save(W / 'teacher_scores.npz', W / 'teacher_meta.json'); RESULTS['teacher'] = O.summary(STORE); save_results()
if not ready:
    RESULTS['teacher_error'] = 'not ready: ' + (W / 'teacher_vllm.log').read_text(errors='ignore')[-3000:]
else:
    TMODEL = requests.get(TURL + '/v1/models', timeout=30).json()['data'][0]['id']
    def job(leaf):
        t = time.time()
        act, ti, tl = O.parse_prompt_logprobs(O.score_leaf(TURL, TMODEL, leaf['ids'], K), leaf['ids'], K)
        return act, ti, tl, time.time() - t
    try:
        a, ti, tl, sec = job(LEAVES[0]); STORE.add(LEAVES[0], a, ti, tl, sec); _save()
        RESULTS['teacher_probe'] = {'tokens': len(LEAVES[0]['ids']), 'seconds': round(sec, 1), 'tok_per_s': round(len(LEAVES[0]['ids']) / max(sec, 1e-6)),
                                    'act_lp_head': [float(x) for x in a[1:6]], 'top_ids_head': ti[1].tolist()}
        print('probe:', RESULTS['teacher_probe'], flush=True); save_results()
        rest = LEAVES[1:]; random.Random(20).shuffle(rest)
        n_ok, errs = O.score_all(rest, job, STORE, lambda: (time.time() - T_START) / 60 > BUDGET_MIN - 6, _save, workers=2, every=10)
        RESULTS['teacher_pass'] = {'scored': n_ok + 1, 'of': len(LEAVES), 'errors': len(errs), 'first_errors': errs[:3], 'minutes': round((time.time() - T_START) / 60, 1)}
    except Exception:
        RESULTS['teacher_error'] = traceback.format_exc()[-2500:]
        RESULTS['teacher_log_tail'] = (W / 'teacher_vllm.log').read_text(errors='ignore')[-2500:]
    _save()
tproc.terminate()
print('\\n===== G20b SUMMARY =====\\n', json.dumps({k: RESULTS.get(k) for k in ('leaves', 'teacher_boot_s', 'teacher_probe', 'teacher_pass', 'teacher', 'teacher_error')}, indent=1, default=str))
RESULTS['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); save_results(); _wd.cancel()
'''.replace("__OPD__", json.dumps(OPD))

compile(MAIN, "main", "exec")
md = ("# G20b: 31B teacher scoring of the G20 student leaves (prompt_logprobs top-8)\n\nRerun of G20 stage 2 after its OOM (2 GiB log_softmax for a 2,048-token chunk); "
      "512-token chunks, --enforce-eager, 0.72 GPU memory. Output `g20b/teacher_scores.npz`.")
cells = [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(True)},
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": INSTALL.splitlines(True)},
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": MAIN.splitlines(True)}]
(OUT / "g20b.ipynb").write_text(json.dumps({"cells": cells, "metadata": G20["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g20" / "kernel-metadata.json").read_text(encoding="utf-8"))
meta.update({"id": "noobhunterjay/gemma-4-g20b-teacher-score", "title": "Gemma 4 G20b Teacher Score", "code_file": "g20b.ipynb",
             "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"], "kernel_sources": ["noobhunterjay/gemma-4-g20-opd-data"],
             "model_sources": ["google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"]})
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
