"""Generate kaggle_assets/g1/ : the G1 training+serving feasibility notebook (needs L4x4; ~1.5 h expected, 150 min hard watchdog).

Stages (each guarded; results.json is rewritten after every stage so a later failure never loses earlier numbers):
  A  baseline serving, NO adapters (what a scorer without LoRA looks like): KV cache, decode speed, concurrency, long-prompt
  B  LoRA training spike on the bf16 QAT checkpoint (scripts/g1/train_g1.py in a subprocess so GPU memory is fully released)
  C  adapter post-processing + serving WITH adapters on the scorer's own VllmServer: adapter active?, KV cache, decode speed, long-prompt
The code files are embedded verbatim from scripts/g1/ (the exact files tested locally on a tiny random Gemma 4 / a mock server).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_experiment1 as e1  # noqa: E402  (INSTALL cell: wheelhouse install, same as every other notebook)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "kaggle_assets" / "g1"
OUT.mkdir(parents=True, exist_ok=True)

CODE_FILES = {n: (HERE / n).read_text(encoding="utf-8").replace("\r\n", "\n") for n in ("train_g1.py", "make_adapters.py", "g1_probe.py")}

SETUP = '''import json, os, re, shutil, subprocess, sys, threading, time, traceback
from pathlib import Path
WORK = Path('/kaggle/working'); G1 = WORK / 'g1'; CODE = G1 / 'code'; CODE.mkdir(parents=True, exist_ok=True)
RES = G1 / 'results.json'; results = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}
BUDGET_MIN = 150
def save_results(): RES.write_text(json.dumps(results, indent=1, default=str))
def _watchdog():   # a hung stage must not burn GPU quota: hard stop (kills the kernel; /kaggle/working is kept)
    results['watchdog'] = f'fired after {BUDGET_MIN} min'; save_results(); os._exit(3)
_wd = threading.Timer(BUDGET_MIN * 60, _watchdog); _wd.daemon = True; _wd.start()
CODE_FILES = __CODE_FILES__
for n, txt in CODE_FILES.items(): (CODE / n).write_text(txt, encoding='utf-8')
sys.path.insert(0, str(CODE))
DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
MODEL_W4 = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
bf16 = sorted(p.parent for p in Path('/kaggle/input').rglob('config.json') if 'unquantized' in str(p).lower())
assert len(bf16) == 1, f'expected one unquantized checkpoint, found {bf16}'
MODEL_BF16 = bf16[0]
G1_DATA = next(Path('/kaggle/input').rglob('g1_train.jsonl'))
print('bf16 checkpoint:', MODEL_BF16, '| w4a16:', MODEL_W4.exists(), '| data:', G1_DATA)
results['gpu'] = subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True, capture_output=True, text=True).stdout.strip().splitlines()
print(results['gpu']); save_results()'''

SERVE = '''import g1_probe as P
def serve_and_probe(tag, adapter_names):
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    sub = WORK / f'sub_{tag}'; shutil.rmtree(sub, ignore_errors=True); shutil.copytree(DATA_DIR / 'sample_submission', sub)
    shutil.rmtree(sub / 'adapters', ignore_errors=True)
    manifest = None
    if adapter_names:
        (sub / 'adapters').mkdir()
        for n in adapter_names: shutil.copytree(G1 / 'adapters' / n, sub / 'adapters' / n)
        manifest = discover_adapters(str(sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    cfg = VllmConfig(model=str(MODEL_W4), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
        max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=True,
        max_loras=8, max_lora_rank=128, tensor_parallel_size=4, startup_timeout=60 * 20)
    srv = VllmServer(cfg, adapter_manifest=manifest)
    rec = {'tag': tag, 'adapters': adapter_names, 'cmd': ' '.join(srv.build_cmd())}
    t0 = time.time()
    try:
        srv.start()
    except Exception as e:
        rec['start_error'] = (str(e) + '\\n' + str(getattr(e, 'output', '') or ''))[-3500:]
        return rec
    try:
        rec['boot_s'] = round(time.time() - t0)
        log = Path(srv.log_path).read_text(errors='ignore')
        rec['kv_lines'] = re.findall(r'(?:GPU KV cache size|Available KV cache memory|Maximum concurrency)[^\\n]*', log)[:6]
        rec['lora_log_lines'] = [l for l in log.splitlines() if 'lora' in l.lower()][:8]
        ids = P.list_models(srv.base_url); rec['model_ids'] = ids
        base_id = next(i for i in ids if i not in (adapter_names or []))
        first = json.loads(open(G1_DATA, encoding='utf-8').readline())['messages']
        agent_prompt = [m for m in first[:2]]           # real system + task prompt (~4-5k tokens)
        rec['decode'] = {base_id: P.decode_probe(srv.base_url, base_id, agent_prompt)}
        for a in adapter_names or []: rec['decode'][a] = P.decode_probe(srv.base_url, a, agent_prompt)
        rec['concurrency4'] = {base_id: P.concurrency_probe(srv.base_url, base_id, agent_prompt)}
        if adapter_names: rec['concurrency4'][adapter_names[0]] = P.concurrency_probe(srv.base_url, adapter_names[0], agent_prompt)
        rec['long_prompt'] = {base_id: P.long_prompt_probe(srv.base_url, base_id, 40000)}
        if adapter_names: rec['long_prompt'][adapter_names[0]] = P.long_prompt_probe(srv.base_url, adapter_names[0], 40000)
        if adapter_names:
            lp = {i: P.logprob_probe(srv.base_url, i) for i in [base_id] + adapter_names}
            rec['adapter_activity'] = {a: P.compare_logprobs(lp[base_id], lp[a]) for a in adapter_names}
    except Exception:
        rec['probe_error'] = traceback.format_exc()[-2500:]
    finally:
        srv.stop(); time.sleep(15)
    return rec'''

STAGE_A = '''# Baseline measured in run 1 (same notebook, same wheelhouse, same day): reused to save ~10 min of L4x4. Source: results of kernel version 1.
results['A_baseline_serving'] = {'reused_from': 'g1 run 1', 'boot_s': 534,
    'kv_lines': ['Available KV cache memory: 10.61 GiB', 'GPU KV cache size: 46,352 tokens', 'Maximum concurrency for 32,768 tokens per request: 8.74x'],
    'decode': {'BASE': {'decode_tok_s_median': 37.71, 'ttft_s_median': 0.08}}, 'concurrency4': {'BASE': {'aggregate_tok_s': 131.9, 'wall_s': 6.07}},
    'cmd': '--max-model-len 32768 --gpu-memory-utilization 0.9 --tensor-parallel-size 4 --reasoning-config {...} --no-scheduler-reserve-full-isl'}
save_results(); print('baseline reused:', results['A_baseline_serving']['decode'], results['A_baseline_serving']['kv_lines'])'''

STAGE_B = '''t0 = time.time()
env = dict(os.environ, G1_MODEL=str(MODEL_BF16), G1_DATA=str(G1_DATA), G1_OUT=str(G1), G1_MAXLEN='16384', G1_MICRO='12', G1_ACC='4',
           G1_MAXMEM=json.dumps({'0': '16GiB', '1': '16GiB', '2': '16GiB', '3': '16GiB'}), G1_TRAIN_SECONDS='1200',
           G1_PROBES='16384,24576,32768', TOKENIZERS_PARALLELISM='false')
try:
    with open(G1 / 'train.log', 'w') as lf:
        p = subprocess.run([sys.executable, str(CODE / 'train_g1.py')], env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=80 * 60)
    results['B_train_returncode'] = p.returncode
except subprocess.TimeoutExpired:
    results['B_train_returncode'] = 'timeout'
tm = G1 / 'train_metrics.json'
results['B_train'] = json.loads(tm.read_text()) if tm.exists() else {'error': 'no train_metrics.json'}
results['B_seconds'] = round(time.time() - t0); save_results()
print(subprocess.run(f'grep "\\\\[g1\\\\]" {G1}/train.log | tail -40', shell=True, capture_output=True, text=True).stdout)
b = results['B_train']; print({k: b.get(k) for k in ('stage', 'tokens_per_second', 'peak_gib_per_gpu', 'load_seconds', 'trainable_params_m', 'error')}); print('probes:', b.get('probes'))'''

STAGE_C = '''t0 = time.time()
src = G1 / 'adapter_full'
if not (src / 'adapter_model.safetensors').exists():
    results['C_adapter_serving'] = {'skipped': 'no trained adapter (stage B failed)'}
else:
    import make_adapters
    try:
        results['C_adapters'] = make_adapters.main(src, G1 / 'adapters', DATA_DIR / 'sample_submission' / 'adapters' / 'main_lora')
    except Exception:
        results['C_adapters'] = {'error': traceback.format_exc()[-2000:]}
    save_results()
    results['C_adapter_serving'] = None
    for names in (['g1_full', 'g1_attn', 'g1_canary'], ['g1_attn', 'g1_canary']):   # fall back to attention-only if MLP LoRA is unsupported by this vLLM
        rec = serve_and_probe('lora_' + '_'.join(n[3:] for n in names), names)
        results['C_adapter_serving'] = rec
        save_results()
        if 'start_error' not in rec: break
results['C_seconds'] = round(time.time() - t0); save_results()
c = results['C_adapter_serving']; print(json.dumps({k: c.get(k) for k in ('adapters', 'boot_s', 'kv_lines', 'lora_log_lines', 'start_error', 'probe_error', 'skipped')}, indent=1)[:3500])
for m, d in (c.get('decode') or {}).items(): print('decode', m, d.get('decode_tok_s_median'), 'tok/s; ttft', d.get('ttft_s_median'))
print('long prompt:', c.get('long_prompt')); print('adapter activity:', {a: (v['max_logprob_diff'], v['any_text_differs']) for a, v in (c.get('adapter_activity') or {}).items()})'''

SUMMARY = '''def g(d, *ks):
    for k in ks:
        d = (d or {}).get(k) if isinstance(d, dict) else None
    return d
A, B, C = results.get('A_baseline_serving') or {}, results.get('B_train') or {}, results.get('C_adapter_serving') or {}
def first_decode(r, adapter=False):
    ds = r.get('decode') or {}
    for m, d in ds.items():
        if adapter == (m.startswith('g1_')): return d.get('decode_tok_s_median')
base_tps = first_decode(A); ad_tps = None
for m, d in (C.get('decode') or {}).items():
    if m == 'g1_attn' or (ad_tps is None and m.startswith('g1_') and m != 'g1_canary'): ad_tps = d.get('decode_tok_s_median')
gates = {
  'train tokens/s >= 60': (B.get('tokens_per_second') or 0) >= 60,
  'training fits 16k tokens (no OOM events)': not B.get('oom_events'),
  'adapter active in serving (canary logprob diff > 0.05)': (g(C, 'adapter_activity', 'g1_canary', 'max_logprob_diff') or 0) > 0.05,
  'adapter decode >= 85% of no-adapter baseline': bool(base_tps and ad_tps and ad_tps / base_tps >= 0.85),
  'long prompt completes with adapter': bool(g(C, 'long_prompt', 'g1_attn', 'ok') or g(C, 'long_prompt', 'g1_full', 'ok')),
}
print('=== G1 SUMMARY ===')
print('baseline decode tok/s:', base_tps, '| adapter(attn) decode tok/s:', ad_tps, '| ratio:', round(ad_tps / base_tps, 3) if base_tps and ad_tps else None)
print('train tok/s:', B.get('tokens_per_second'), '| peak GiB/GPU:', B.get('peak_gib_per_gpu'), '| probes:', [(p.get('len'), p.get('ok')) for p in (B.get('probes') or [])])
print('KV lines baseline:', A.get('kv_lines')); print('KV lines with adapters:', C.get('kv_lines'))
for k, v in gates.items(): print(('PASS ' if v else 'FAIL ') + k)
results['gates'] = gates; results['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); save_results()
for d in WORK.glob('sub_*'): shutil.rmtree(d, ignore_errors=True)   # keep the output small
_wd.cancel()'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


nb = {"cells": [
    md("# G1: LoRA training + serving feasibility spike (Gemma 4 31B, L4x4)\n\nStage A baseline (reused from run 1) -> B training spike on the bf16 QAT checkpoint -> C adapter serving on the scorer's VllmServer. "
       "Hard 150-minute watchdog. Plan and pass criteria: docs/STRATEGY.md 'POST-TRAINING PLAN'."),
    code(e1.INSTALL), code(SETUP.replace("__CODE_FILES__", json.dumps(CODE_FILES))), md("## Serving helper"), code(SERVE),
    md("## Stage A: baseline serving, no adapters"), code(STAGE_A), md("## Stage B: training spike"), code(STAGE_B),
    md("## Stage C: adapter serving"), code(STAGE_C), md("## Summary and gates"), code(SUMMARY)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5}
(OUT / "g1.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g1-spike", "title": "Gemma 4 G1 Spike", "code_file": "g1.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g1-data"], "kernel_sources": [],
    "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": ["google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2", "google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    "machine_shape": "NvidiaL4"}, indent=2), encoding="utf-8")
print("wrote", OUT)
