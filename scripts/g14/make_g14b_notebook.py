"""Generate kaggle_assets/g14b/ : G14b = LoRA SFT of the E4B (bf16 qat-q4_0-unquantized sibling, HF gradient checkpointing) on the 31B-resolved POOL sequences of G10 (the G11 dataset: 71 sequences, 54 tasks,
0.87M tokens, produced by the 31B with the B1 prompt; the E4B shares the tokenizer and chat template), 2 epochs, then the 75-task competition evaluation with the adapter (B1 prompt + B2 budget, exactly like G14a:
base E4B = 10/75). Derived from the G3 generator by exact-string edits.
[G3 docstring follows] G3 = LoRA SFT on the G2 sequences (1 epoch, all 30 solved tasks, no hold-out) + evaluation round with the adapter.

Stages: INSTALL -> SETUP -> WIRING CHECK (placeholder adapter through the real compile path, seconds, before any training) -> stage checkpoint to local disk and
        train (supervised subprocess, milestone deadlines) -> package adapter + wiring check on the REAL adapter -> vLLM with the adapter + capture proxy ->
        smoke round + gate (requests must really use the adapter) -> one evaluation round over all 75 tasks -> summary vs the G2 base rounds.
Expected: ~3.0 h training + ~0.4 h staging/boot/smoke + ~1.6 h evaluation = ~5 h of L4x4 (~10 quota h). 340-minute watchdog.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / 'g3'   # the shared trainer/wiring sources live in scripts/g3
sys.path.insert(0, str(HERE.parent))
import make_experiment1 as e1  # noqa: E402  (INSTALL + VLLM cells shared with every earlier notebook)

ROOT = HERE.parents[1]
OUT = ROOT / "kaggle_assets" / "g14b"
OUT.mkdir(parents=True, exist_ok=True)
CODE_FILES = {
    "train_g3.py": HERE / "train_g3.py", "g3_wiring.py": HERE / "g3_wiring.py",
    "gemma4_memfix.py": HERE.parent / "g1" / "gemma4_memfix.py", "g1_util.py": HERE.parent / "g1" / "g1_util.py",
    "make_adapters.py": HERE.parent / "g1" / "make_adapters.py", "capture_proxy.py": HERE.parent / "g2" / "capture_proxy.py"}
CODE = {n: p.read_text(encoding="utf-8").replace("\r\n", "\n") for n, p in CODE_FILES.items()}
assert 'BASE = "gemma-4-31b-it-qat-w4a16-ct"' in CODE["g3_wiring.py"]
CODE["g3_wiring.py"] = CODE["g3_wiring.py"].replace('BASE = "gemma-4-31b-it-qat-w4a16-ct"', 'BASE = "gemma-4-e4b-it-qat-w4a16-ct"')
gold = json.loads((ROOT / "docs" / "kaggle_env_gold_results.json").read_text(encoding="utf-8"))
TASK_IDS = sorted(i for i, v in gold.items() if v["resolved"])
assert len(TASK_IDS) == 75

B2EVAL_TEXT = (ROOT / 'submission_b2' / 'eval_config.yaml').read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))
SETUP = '''import json, os, random, shutil, subprocess, sys, threading, time, traceback
from pathlib import Path
WORK = Path('/kaggle/working'); G3 = WORK / 'g14b'; CODE = G3 / 'code'; CODE.mkdir(parents=True, exist_ok=True); WORKING_DIR = WORK
R = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}
def save_results(): (G3 / 'g14b_summary.json').write_text(json.dumps(R, indent=1, default=str))
BUDGET_MIN = 480; T_START = time.time()
def _watchdog():
    R['watchdog'] = f'fired after {BUDGET_MIN} min'; save_results(); os._exit(3)
_wd = threading.Timer(BUDGET_MIN * 60, _watchdog); _wd.daemon = True; _wd.start()
for n, txt in __CODE__.items(): (CODE / n).write_text(txt, encoding='utf-8')
sys.path.insert(0, str(CODE))
DATA_DIR = next(p.parent for p in Path('/kaggle/input').rglob('tasks.jsonl') if (p.parent / 'sample_submission').is_dir() and (p.parent / 'snapshots').is_dir()); TASKS_PATH = DATA_DIR / 'tasks.jsonl'
_sft = sorted(Path('/kaggle/input').rglob('sft_sequences.jsonl')); assert len(_sft) == 1, _sft; SFT = _sft[0]
B1 = next(p.parent for p in Path('/kaggle/input').rglob('agent.yaml') if 'swe_scaffold_v2_b1' in p.read_text(encoding='utf-8') and 'adapter:' not in p.read_text(encoding='utf-8'))
_B2EVAL = __B2EVAL__
_b1e = WORK / 'b1_e4b'; shutil.rmtree(_b1e, ignore_errors=True); shutil.copytree(B1, _b1e)
for _rel in ('agent.yaml', 'sub_agents/code_analyzer.yaml'):
    _p = _b1e / _rel; _t = _p.read_text(encoding='utf-8'); assert _t.count('gemma-4-31b-it-qat-w4a16-ct') == 1, _rel; _p.write_text(_t.replace('gemma-4-31b-it-qat-w4a16-ct', 'gemma-4-e4b-it-qat-w4a16-ct'), encoding='utf-8', newline=chr(10))
(_b1e / 'eval_config.yaml').write_text(_B2EVAL, encoding='utf-8', newline=chr(10)); B1 = _b1e
import torch
_gpus = [(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]; print('GPUs:', _gpus)
assert len(_gpus) == 4 and all(c[0] >= 8 for _, c in _gpus), f'expected 4x L4, got {_gpus}'
bf = sorted(p.parent for p in Path('/kaggle/input').rglob('config.json') if 'unquantized' in str(p).lower()); assert len(bf) == 1, bf
MODEL_BF16 = bf[0]
ALL_IDS = __TASK_IDS__
TRAINED = sorted({json.loads(l)['task'] for l in open(SFT, encoding='utf-8')}); UNSEEN = [i for i in ALL_IDS if i not in TRAINED]
print('sft:', SFT, '| B1 arm:', B1, '| bf16:', MODEL_BF16, '| trained tasks:', len(TRAINED), '| unseen tasks:', len(UNSEEN))
R['trained_tasks'] = TRAINED; save_results()
from swegemma.models import load_tasks
by_id = {t.instance_id: t for t in load_tasks(TASKS_PATH)}
assert not [i for i in ALL_IDS if i not in by_id]
SMOKE_IDS = random.Random(5).sample([i for i in ALL_IDS if i.startswith(('fastapi_', 'rich_'))], 4)'''

WIRING = '''import g3_wiring as W
# Fail fast, before ~3 h of training: does an adapter flow through the harness's own compile path? (placeholder = the organizers' sample adapter)
t0 = time.time()
arm0 = W.make_eval_arm(B1, DATA_DIR / 'sample_submission' / 'adapters' / 'main_lora', Path('/tmp/wiring_arm'))
R['wiring_placeholder'] = W.check_wiring(arm0); print('WIRING OK with placeholder adapter:', R['wiring_placeholder'], round(time.time() - t0, 1), 's'); save_results()'''

TRAIN = '''import g1_util as U
t0 = time.time(); STAGED = Path('/tmp/ckpt_bf16')
R['stage'] = U.stage_checkpoint(MODEL_BF16, STAGED, workers=12); print('staged:', R['stage']); save_results()
env = dict(os.environ, G3_MODEL=str(STAGED), G3_DATA=str(SFT), G3_OUT=str(G3 / 'train'), G3_ACC='2', G3_LR='2e-4', G3_EPOCHS='2', G3_WARMUP='3', G3_HF_GC='1',
           G3_MAXMEM=json.dumps({'0': '20GiB', '1': '20GiB', '2': '20GiB', '3': '20GiB'}), G3_CKPT_EVERY='10', TOKENIZERS_PARALLELISM='false')
R['train_run'] = U.run_supervised([sys.executable, str(CODE / 'train_g3.py')], env, G3 / 'train.log',
                                  [('[g3] loaded in', 600), ('[g3] step 1/', 1800)], max_seconds=5.0 * 3600)
tm = G3 / 'train' / 'train_metrics.json'
M = json.loads(tm.read_text()) if tm.exists() else {'error': 'no train_metrics.json'}
R['train'] = {k: M.get(k) for k in ('stage', 'error', 'optimizer_steps_planned', 'tokens_per_second', 'peak_gib_per_gpu', 'oom_events', 'leak_gib_at_end', 'train_seconds', 'trainable_params_m', 'data')}
R['train']['losses'] = [s['loss'] for s in M.get('steps', [])]; R['train']['grad_norms'] = [s['grad_norm'] for s in M.get('steps', [])]
R['train_wall_s'] = round(time.time() - t0); save_results(); shutil.rmtree(STAGED, ignore_errors=True)
print(R['train_run']); print({k: v for k, v in R['train'].items() if k not in ('losses', 'grad_norms')}); print('losses:', R['train']['losses']); print('grad norms:', R['train']['grad_norms'])
planned = M.get('optimizer_steps_planned') or 1
if M.get('stage') != 'done' or len(M.get('steps', [])) < 0.8 * planned:   # a few OOM-skipped groups are tolerated (logged); a failed or mostly-skipped run is not
    raise RuntimeError(f"TRAINING FAILED, not spending GPU time on evaluation: stage={M.get('stage')} steps={len(M.get('steps', []))}/{planned} oom={M.get('oom_events')} error={M.get('error')}")'''

PACKAGE = '''os.environ['SCORER_MODEL'] = 'google/gemma-4-e4b-it-qat-w4a16-ct'
import make_adapters
assert make_adapters.SCORER_MODEL == 'google/gemma-4-e4b-it-qat-w4a16-ct', make_adapters.SCORER_MODEL
rep = make_adapters.main(G3 / 'train' / 'adapter_full', G3 / 'adapters', DATA_DIR / 'sample_submission' / 'adapters' / 'main_lora')
R['adapter'] = {'bytes': rep['g1_full']['bytes'], 'tensors': rep['g1_full']['tensors'], 'target_modules': rep['g1_full']['target_modules']}
from safetensors.torch import load_file
tt = load_file(str(G3 / 'adapters' / 'g1_full' / 'adapter_model.safetensors'))
bs = [v.float().abs().mean().item() for k, v in tt.items() if 'lora_B' in k]
R['adapter']['mean_abs_lora_B'] = sum(bs) / len(bs); R['adapter']['nonzero_B_fraction'] = sum(1 for x in bs if x > 0) / len(bs)
EVAL_ARM = W.make_eval_arm(B1, G3 / 'adapters' / 'g1_full', WORK / 'eval_arm')
R['wiring_real'] = W.check_wiring(EVAL_ARM); print('adapter:', R['adapter']); print('WIRING OK with the trained adapter:', R['wiring_real']); save_results()
assert R['adapter']['mean_abs_lora_B'] > 0, 'trained adapter has all-zero B matrices'
(WORK / 'eval_arm' / 'adapters' / 'g3_lora' / 'adapter_config.json').read_text()'''

VLLM = (e1.VLLM.replace("first_arm = ARMS[sorted(ARMS)[0]]", "first_arm = EVAL_ARM")
        .replace("TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'", "TARGET_MODEL_NAME = 'gemma-4-e4b-it-qat-w4a16-ct'")
        .replace("MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')", "MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2')")
        .replace("startup_timeout=60 * 20)", "startup_timeout=60 * 12)")
        .replace("models = server_instance.create_model_registry(aliases=[declared_model, TARGET_MODEL_NAME], model_prefix='openai/', api_key='EMPTY')",
                 "import capture_proxy as CP\n"
                 "VLLM_URL = f'http://127.0.0.1:{server_instance.config.port}'\n"
                 "proxy = CP.CaptureProxy(VLLM_URL, str(G3 / 'capture.jsonl.gz'), port=8001).start()\n"
                 "server_instance.config.port = 8001      # the harness now talks to the capture proxy\n"
                 "models = server_instance.create_model_registry(aliases=[declared_model, TARGET_MODEL_NAME], model_prefix='openai/', api_key='EMPTY')\n"
                 "print('registry aliases:', sorted(getattr(models, '_registry', {}).keys()) if hasattr(models, '_registry') else 'n/a')\n"
                 "print('harness endpoint (proxy):', server_instance.base_url, '-> upstream', VLLM_URL, '| server cmd has --enable-lora:', '--enable-lora' in ' '.join(server_instance.build_cmd()))"))
assert "first_arm = EVAL_ARM" in VLLM and "capture_proxy" in VLLM and "gemma-4-e4b-it-qat-w4a16-ct/2" in VLLM and "31b" not in VLLM and "startup_timeout=60 * 12" in VLLM and "enable_lora=True" in VLLM

EVAL = '''import asyncio, concurrent.futures, yaml
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
def run_sync(coro_fn):
    try: loop = asyncio.get_running_loop()
    except RuntimeError: loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool: return pool.submit(lambda: asyncio.run(coro_fn())).result()
    return asyncio.run(coro_fn())
def budget_of(d):
    cfg = yaml.safe_load((d / 'eval_config.yaml').read_text(encoding='utf-8')).get('evaluation', {}); out = {}
    for k, cast in (('timeout_seconds', int), ('max_tool_calls', int), ('max_time_minutes', float)):
        if k in cfg: out[k] = cast(cfg[k])
    t = cfg.get('max_turns', cfg.get('max_llm_calls'))
    if t is not None: out['max_turns'] = int(t)
    return out
limits, gen_constraints = build_submission_limits()
def run_round(tag, ids):
    proxy.round_tag = tag; rd = G3 / tag
    cfg = EvalConfig(tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / 'snapshots', results_dir=rd, submission_dir=EVAL_ARM, models=models, sandbox='subprocess',
        task_ids=list(ids), concurrency=4, display_mode='quiet', limits=limits, generation_constraints=gen_constraints, adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
        graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'), wheels_dir=DATA_DIR / 'wheels', verbose=False, **budget_of(EVAL_ARM))
    t0 = time.perf_counter()
    try: run_sync(Evaluator(cfg).run)
    except Exception:
        traceback.print_exc(); R[tag] = {'error': 'run raised', 'trace': traceback.format_exc()[-1500:]}; save_results(); return None
    rows = [json.loads(l) for l in open(rd / 'task_results.jsonl', encoding='utf-8')]
    ok = {r['instance_id'] for r in rows if r['resolved']}
    R[tag] = {'n': len(rows), 'resolved': len(ok), 'patch_written': sum(1 for r in rows if r['agent_patch_size'] > 0),
              'time_cap_hit': sum(1 for r in rows if 'session timeout' in (r['error'] or '')), 'wall_s': round(time.perf_counter() - t0, 1),
              'mean_task_s': round(sum(r['duration_seconds'] for r in rows) / max(1, len(rows)), 1), 'resolved_ids': sorted(ok),
              'resolved_on_trained_tasks': len(ok & set(TRAINED)), 'resolved_on_unseen_tasks': len(ok & set(UNSEEN)), 'proxy': proxy.stats()}
    save_results(); print(tag, {k: v for k, v in R[tag].items() if k != 'resolved_ids'}, flush=True); return rows

# ---- adapter-effect gate: the same temperature-0 request to the base and to the adapter must differ (a silently inactive LoRA would pass the name check below)
import requests
def _ask(model, prompt):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'max_tokens': 64, 'logprobs': True, 'top_logprobs': 1, 'chat_template_kwargs': {'enable_thinking': False}}
    r = requests.post(VLLM_URL + '/v1/chat/completions', json=body, timeout=600); r.raise_for_status(); ch = r.json()['choices'][0]
    return ch['message'].get('content') or '', [t['logprob'] for t in (ch.get('logprobs') or {}).get('content', [])]
_eff = []
for _p in ('Write a Python function that returns the n-th Fibonacci number, then explain it in one sentence.', 'The bug: rich.table.Table drops the caption when show_header=False. Which file would you look at first, and why?'):
    _tb, _lb = _ask(str(MODEL_PATH), _p); _ta, _la = _ask('g3_lora', _p); _n = min(len(_lb), len(_la))
    _eff.append({'text_differs': _tb != _ta, 'mean_abs_logprob_diff': round(sum(abs(a - b) for a, b in zip(_lb[:_n], _la[:_n])) / max(_n, 1), 4)})
R['adapter_effect'] = _eff; save_results(); print('adapter effect (base vs adapter, temperature 0):', _eff)
if not any(e['text_differs'] or e['mean_abs_logprob_diff'] > 0.01 for e in _eff): raise RuntimeError(f'ADAPTER HAS NO EFFECT on the E4B in vLLM: {_eff}')

# ---- smoke round + gate: requests must really go to the adapter (model name g3_lora), capture must work
rows = run_round('smoke', SMOKE_IDS)
st = proxy.stats(); recs = list(CP.read_records(str(G3 / 'capture.jsonl.gz')))
main = [r for r in recs if (next((m['content'] for m in r['request']['messages'] if m['role'] == 'system'), '') or '').startswith('You are an expert autonomous')]
used = sum(1 for r in main if r['request'].get('model') == 'g3_lora')
gate = {'rows': bool(rows) and len(rows) == len(SMOKE_IDS), 'records >= 12': st['records'] >= 12, 'no proxy errors': st['errors'] == 0,
        'main-agent requests use the adapter (>= 95%)': bool(main) and used / len(main) >= 0.95}
R['smoke_gate'] = gate; R['smoke_model_names'] = sorted({r['request'].get('model') for r in recs}); save_results(); print('smoke gate:', gate, '| model names seen:', R['smoke_model_names'], f'| adapter used in {used}/{len(main)} main requests')
if not all(gate.values()): raise RuntimeError(f'SMOKE GATE FAILED, stopping before the evaluation round: {gate} {st}')

# ---- evaluation round over all 75 tasks with the adapter
ids = list(ALL_IDS); random.Random(201).shuffle(ids)
run_round('e1', ids)
rep = R.get('e1', {})
print('\\n===== G14b SUMMARY =====')
print('training:', {k: v for k, v in R['train'].items() if k not in ('losses', 'grad_norms')})
print('adapter eval round e1:', {k: v for k, v in rep.items() if k not in ('resolved_ids', 'proxy')})
print('BASE E4B (G14a, same prompt and budget): 10/75 resolved, 68 patches, 1 capped, 114 s per task; the 31B base solves 25 and 28')
R['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); R['capture'] = proxy.stats(); save_results(); _wd.cancel()'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


nb = {"cells": [
    md("# G14b: E4B LoRA SFT on 31B-resolved pool sequences (G11 data, 2 epochs) + evaluation round with the adapter\n\n"
       "Wiring check -> train (staged checkpoint, supervised) -> package -> vLLM + capture proxy with the adapter -> smoke gate -> 75-task evaluation round."),
    code(e1.INSTALL), md("## Setup"), code(SETUP.replace("__B2EVAL__", repr(B2EVAL_TEXT)).replace("__CODE__", json.dumps(CODE)).replace("__TASK_IDS__", json.dumps(TASK_IDS))),
    md("## Fail-fast: adapter wiring through the harness compile path"), code(WIRING), md("## Train"), code(TRAIN),
    md("## Package adapter, re-check wiring with the trained adapter"), code(PACKAGE), md("## vLLM with the adapter + capture proxy"), code(VLLM),
    md("## Smoke gate and evaluation round"), code(EVAL)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5}
(OUT / "g14b.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g14b-e4b-train", "title": "Gemma 4 G14b E4B Train", "code_file": "g14b.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g11-sft", "noobhunterjay/gemma4-g2-arm"], "kernel_sources": [],
    "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": ["google/gemma-4/Transformers/gemma-4-e4b-it-qat-q4_0-unquantized/2", "google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    "machine_shape": "NvidiaL4"}, indent=2), encoding="utf-8")
print("wrote", OUT, "| tasks", len(TASK_IDS))
