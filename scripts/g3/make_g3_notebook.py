"""Generate kaggle_assets/g3/ : G3 = LoRA SFT on the G2 sequences (1 epoch, all 30 solved tasks, no hold-out) + evaluation round with the adapter.

Stages: INSTALL -> SETUP -> WIRING CHECK (placeholder adapter through the real compile path, seconds, before any training) -> stage checkpoint to local disk and
        train (supervised subprocess, milestone deadlines) -> package adapter + wiring check on the REAL adapter -> vLLM with the adapter + capture proxy ->
        smoke round + gate (requests must really use the adapter) -> one evaluation round over all 75 tasks -> summary vs the G2 base rounds.
Expected: ~3.0 h training + ~0.4 h staging/boot/smoke + ~1.6 h evaluation = ~5 h of L4x4 (~10 quota h). 340-minute watchdog.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_experiment1 as e1  # noqa: E402  (INSTALL + VLLM cells shared with every earlier notebook)

ROOT = HERE.parents[1]
OUT = ROOT / "kaggle_assets" / "g3"
OUT.mkdir(parents=True, exist_ok=True)
CODE_FILES = {
    "train_g3.py": HERE / "train_g3.py", "g3_wiring.py": HERE / "g3_wiring.py",
    "gemma4_memfix.py": HERE.parent / "g1" / "gemma4_memfix.py", "g1_util.py": HERE.parent / "g1" / "g1_util.py",
    "make_adapters.py": HERE.parent / "g1" / "make_adapters.py", "capture_proxy.py": HERE.parent / "g2" / "capture_proxy.py"}
CODE = {n: p.read_text(encoding="utf-8").replace("\r\n", "\n") for n, p in CODE_FILES.items()}
gold = json.loads((ROOT / "docs" / "kaggle_env_gold_results.json").read_text(encoding="utf-8"))
TASK_IDS = sorted(i for i, v in gold.items() if v["resolved"])
assert len(TASK_IDS) == 75

SETUP = '''import json, os, random, shutil, subprocess, sys, threading, time, traceback
from pathlib import Path
WORK = Path('/kaggle/working'); G3 = WORK / 'g3'; CODE = G3 / 'code'; CODE.mkdir(parents=True, exist_ok=True); WORKING_DIR = WORK
R = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}
def save_results(): (G3 / 'g3_summary.json').write_text(json.dumps(R, indent=1, default=str))
BUDGET_MIN = 340; T_START = time.time()
def _watchdog():
    R['watchdog'] = f'fired after {BUDGET_MIN} min'; save_results(); os._exit(3)
_wd = threading.Timer(BUDGET_MIN * 60, _watchdog); _wd.daemon = True; _wd.start()
for n, txt in __CODE__.items(): (CODE / n).write_text(txt, encoding='utf-8')
sys.path.insert(0, str(CODE))
DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent'); TASKS_PATH = DATA_DIR / 'tasks.jsonl'
SFT = next(Path('/kaggle/input').rglob('sft_sequences.jsonl'))
B1 = next(p.parent for p in Path('/kaggle/input').rglob('agent.yaml') if 'swe_scaffold_v2_b1' in p.read_text(encoding='utf-8') and 'adapter:' not in p.read_text(encoding='utf-8'))
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
env = dict(os.environ, G3_MODEL=str(STAGED), G3_DATA=str(SFT), G3_OUT=str(G3 / 'train'), G3_ACC='2', G3_LR='2e-4', G3_EPOCHS='1', G3_WARMUP='3',
           G3_MAXMEM=json.dumps({'0': '16GiB', '1': '16GiB', '2': '16GiB', '3': '16GiB'}), G3_CKPT_EVERY='10', TOKENIZERS_PARALLELISM='false')
R['train_run'] = U.run_supervised([sys.executable, str(CODE / 'train_g3.py')], env, G3 / 'train.log',
                                  [('[g3] loaded in', 600), ('[g3] step 1/', 1800)], max_seconds=4.0 * 3600)
tm = G3 / 'train' / 'train_metrics.json'
M = json.loads(tm.read_text()) if tm.exists() else {'error': 'no train_metrics.json'}
R['train'] = {k: M.get(k) for k in ('stage', 'error', 'optimizer_steps_planned', 'tokens_per_second', 'peak_gib_per_gpu', 'oom_events', 'leak_gib_at_end', 'train_seconds', 'trainable_params_m', 'data')}
R['train']['losses'] = [s['loss'] for s in M.get('steps', [])]; R['train']['grad_norms'] = [s['grad_norm'] for s in M.get('steps', [])]
R['train_wall_s'] = round(time.time() - t0); save_results(); shutil.rmtree(STAGED, ignore_errors=True)
print(R['train_run']); print({k: v for k, v in R['train'].items() if k not in ('losses', 'grad_norms')}); print('losses:', R['train']['losses']); print('grad norms:', R['train']['grad_norms'])
planned = M.get('optimizer_steps_planned') or 1
if M.get('stage') != 'done' or len(M.get('steps', [])) < 0.8 * planned:   # a few OOM-skipped groups are tolerated (logged); a failed or mostly-skipped run is not
    raise RuntimeError(f"TRAINING FAILED, not spending GPU time on evaluation: stage={M.get('stage')} steps={len(M.get('steps', []))}/{planned} oom={M.get('oom_events')} error={M.get('error')}")'''

PACKAGE = '''import make_adapters
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
        .replace("models = server_instance.create_model_registry(aliases=[declared_model, TARGET_MODEL_NAME], model_prefix='openai/', api_key='EMPTY')",
                 "import capture_proxy as CP\n"
                 "VLLM_URL = f'http://127.0.0.1:{server_instance.config.port}'\n"
                 "proxy = CP.CaptureProxy(VLLM_URL, str(G3 / 'capture.jsonl.gz'), port=8001).start()\n"
                 "server_instance.config.port = 8001      # the harness now talks to the capture proxy\n"
                 "models = server_instance.create_model_registry(aliases=[declared_model, TARGET_MODEL_NAME], model_prefix='openai/', api_key='EMPTY')\n"
                 "print('registry aliases:', sorted(getattr(models, '_registry', {}).keys()) if hasattr(models, '_registry') else 'n/a')\n"
                 "print('harness endpoint (proxy):', server_instance.base_url, '-> upstream', VLLM_URL, '| server cmd has --enable-lora:', '--enable-lora' in ' '.join(server_instance.build_cmd()))"))
assert "first_arm = EVAL_ARM" in VLLM and "capture_proxy" in VLLM

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
print('\\n===== G3 SUMMARY =====')
print('training:', {k: v for k, v in R['train'].items() if k not in ('losses', 'grad_norms')})
print('adapter eval round e1:', {k: v for k, v in rep.items() if k not in ('resolved_ids', 'proxy')})
print('BASE (G2, same prompt, same harness): r1 25/75 (patch 41, cap 44, 273.7 s), r2 28/75 (patch 47, cap 39, 269.3 s); the 45 never-solved tasks: base 0 of 45 in both rounds')
R['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); R['capture'] = proxy.stats(); save_results(); _wd.cancel()'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


nb = {"cells": [
    md("# G3: LoRA SFT on G2 trajectories (all 30 solved tasks, 1 epoch) + evaluation round with the adapter\n\n"
       "Wiring check -> train (staged checkpoint, supervised) -> package -> vLLM + capture proxy with the adapter -> smoke gate -> 75-task evaluation round."),
    code(e1.INSTALL), md("## Setup"), code(SETUP.replace("__CODE__", json.dumps(CODE)).replace("__TASK_IDS__", json.dumps(TASK_IDS))),
    md("## Fail-fast: adapter wiring through the harness compile path"), code(WIRING), md("## Train"), code(TRAIN),
    md("## Package adapter, re-check wiring with the trained adapter"), code(PACKAGE), md("## vLLM with the adapter + capture proxy"), code(VLLM),
    md("## Smoke gate and evaluation round"), code(EVAL)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5}
(OUT / "g3.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g3-train", "title": "Gemma 4 G3 Train", "code_file": "g3.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g3-sft", "noobhunterjay/gemma4-g2-arm"], "kernel_sources": [],
    "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": ["google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2", "google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    "machine_shape": "NvidiaL4"}, indent=2), encoding="utf-8")
print("wrote", OUT, "| tasks", len(TASK_IDS))
