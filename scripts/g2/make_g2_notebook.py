"""Generate kaggle_assets/g2/ : G2 data-collection notebook (L4x4). Runs the B1 policy on all 75 Kaggle-gold-passing public tasks, every vLLM
request/response captured exactly by scripts/g2/capture_proxy.py (token ids included), for later rejection-sampling SFT.

Stages: INSTALL -> SETUP (B1 arm + 75 task ids) -> vLLM + capture proxy -> SMOKE round (4 tasks) with a fail-fast capture gate ->
        rounds r1, r2 over all 75 tasks (separate results dirs; arm order/ task order shuffled per round) -> summary.
Budget: expected ~3.4 h of L4x4 (~6.8 quota h); 235-minute watchdog; a new round is not started if it could not finish inside the watchdog.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_experiment1 as e1  # noqa: E402  (INSTALL and VLLM cells, identical to every earlier notebook)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kaggle_assets" / "g2"
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
CODE_FILES = {"capture_proxy.py": (HERE / "capture_proxy.py").read_text(encoding="utf-8").replace("\r\n", "\n")}
gold = json.loads((ROOT / "docs" / "kaggle_env_gold_results.json").read_text(encoding="utf-8"))
TASK_IDS = sorted(i for i, v in gold.items() if v["resolved"])
assert len(TASK_IDS) == 75, len(TASK_IDS)

SETUP = '''import json, os, random, shutil, sys, threading, time, traceback
from pathlib import Path
WORKING_DIR = Path('/kaggle/working'); WORKING_DIR.mkdir(parents=True, exist_ok=True)
G2 = WORKING_DIR / 'g2'; G2.mkdir(exist_ok=True); CODE = G2 / 'code'; CODE.mkdir(exist_ok=True)
RESULTS = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}
def save_results(): (G2 / 'g2_summary.json').write_text(json.dumps(RESULTS, indent=1, default=str))
BUDGET_MIN = 235; T_START = time.time()
def _watchdog():   # a hung round must not burn GPU quota
    RESULTS['watchdog'] = f'fired after {BUDGET_MIN} min'; save_results(); os._exit(3)
_wd = threading.Timer(BUDGET_MIN * 60, _watchdog); _wd.daemon = True; _wd.start()
for n, txt in __CODE_FILES__.items(): (CODE / n).write_text(txt, encoding='utf-8')
sys.path.insert(0, str(CODE))
DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
arm_dirs = [p.parent for p in Path('/kaggle/input').rglob('agent.yaml') if 'swe_scaffold_v2_b1' in p.read_text(encoding='utf-8')]
assert len(arm_dirs) == 1, f'expected exactly one B1 arm, found {arm_dirs}'
ARM = WORKING_DIR / 'arm'; shutil.rmtree(ARM, ignore_errors=True); shutil.copytree(arm_dirs[0], ARM)
for must in ('agent.yaml', 'prompts/system.md', 'prompts/analyzer.md', 'configs/sampling.yaml', 'sub_agents/code_analyzer.yaml', 'eval_config.yaml'):
    assert (ARM / must).is_file(), f'arm missing {must}'
print('arm:', arm_dirs[0], '|', (ARM / 'configs/sampling.yaml').read_text().replace(chr(10), ' ')[-120:])
from swegemma.models import load_tasks
ALL_IDS = __TASK_IDS__
by_id = {t.instance_id: t for t in load_tasks(TASKS_PATH)}
missing = [i for i in ALL_IDS if i not in by_id]
assert not missing, f'task ids missing from the dataset: {missing}'
SMOKE_IDS = random.Random(5).sample([i for i in ALL_IDS if i.startswith(('fastapi_', 'rich_'))], 4)
print(len(ALL_IDS), 'tasks;', 'smoke ids:', SMOKE_IDS)'''

VLLM = (e1.VLLM
        .replace("first_arm = ARMS[sorted(ARMS)[0]]", "first_arm = ARM")
        .replace("models = server_instance.create_model_registry(aliases=[declared_model, TARGET_MODEL_NAME], model_prefix='openai/', api_key='EMPTY')",
                 "import capture_proxy as CP\n"
                 "VLLM_URL = f'http://127.0.0.1:{server_instance.config.port}'\n"
                 "proxy = CP.CaptureProxy(VLLM_URL, str(G2 / 'capture.jsonl.gz'), port=8001).start()\n"
                 "server_instance.config.port = 8001      # the model registry reads base_url at creation time: from now on the harness talks to the capture proxy\n"
                 "models = server_instance.create_model_registry(aliases=[declared_model, TARGET_MODEL_NAME], model_prefix='openai/', api_key='EMPTY')\n"
                 "print('harness endpoint (proxy):', server_instance.base_url, '-> upstream', VLLM_URL)"))
assert "first_arm = ARM\n" in VLLM and "capture_proxy" in VLLM

RUN = '''import asyncio, concurrent.futures, requests, yaml
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator

def run_sync(coro_fn):
    try: loop = asyncio.get_running_loop()
    except RuntimeError: loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro_fn())).result()
    return asyncio.run(coro_fn())

def budget_of(arm_dir):
    cfg = yaml.safe_load((arm_dir / 'eval_config.yaml').read_text(encoding='utf-8')).get('evaluation', {})
    out = {}
    for k, cast in (('timeout_seconds', int), ('max_tool_calls', int), ('max_time_minutes', float)):
        if k in cfg: out[k] = cast(cfg[k])
    t = cfg.get('max_turns', cfg.get('max_llm_calls'))
    if t is not None: out['max_turns'] = int(t)
    return out

limits, gen_constraints = build_submission_limits()
print('per-task budget:', budget_of(ARM))

def run_round(tag, ids):
    proxy.round_tag = tag
    rd = G2 / tag
    cfg = EvalConfig(tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / 'snapshots', results_dir=rd, submission_dir=ARM, models=models, sandbox='subprocess',
        task_ids=list(ids), concurrency=4, display_mode='quiet', limits=limits, generation_constraints=gen_constraints, adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
        graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'), wheels_dir=DATA_DIR / 'wheels', verbose=False, **budget_of(ARM))
    t0 = time.perf_counter()
    try:
        run_sync(Evaluator(cfg).run)
    except Exception:
        traceback.print_exc(); RESULTS[tag] = {'error': 'run raised', 'trace': traceback.format_exc()[-1500:]}; save_results(); return None
    rows = [json.loads(l) for l in open(rd / 'task_results.jsonl', encoding='utf-8')]
    RESULTS[tag] = {'n': len(rows), 'resolved': sum(r['resolved'] for r in rows), 'patch_written': sum(1 for r in rows if r['agent_patch_size'] > 0),
                    'time_cap_hit': sum(1 for r in rows if 'session timeout' in (r['error'] or '')), 'wall_s': round(time.perf_counter() - t0, 1),
                    'mean_task_s': round(sum(r['duration_seconds'] for r in rows) / max(1, len(rows)), 1), 'resolved_ids': sorted(r['instance_id'] for r in rows if r['resolved']),
                    'proxy': proxy.stats()}
    save_results(); print(tag, {k: v for k, v in RESULTS[tag].items() if k != 'resolved_ids'}, flush=True)
    return rows

# ------------------------------------------------------------------ SMOKE: 4 tasks, then a fail-fast capture gate (saves quota if capture is broken)
rows = run_round('smoke', SMOKE_IDS)
st = proxy.stats(); print('capture after smoke:', st)
gate = {'rows': bool(rows) and len(rows) == len(SMOKE_IDS), 'records >= 12': st['records'] >= 12, 'no proxy errors': st['errors'] == 0}
if st['with_token_ids'] == 0 and st['records'] > 0:      # vLLM returned no ids: switch on the /tokenize fallback for the rest of the run
    proxy.tokenize_fallback = True; print('WARNING: no token ids from vLLM; /tokenize fallback enabled')
try:   # cross-check: ids returned by vLLM == ids from /tokenize for the same request (the exactness of the whole capture rests on this)
    first = next(CP.read_records(str(G2 / 'capture.jsonl.gz')))
    rq = first['request']
    tk = {k: rq[k] for k in ('model', 'messages', 'tools', 'chat_template_kwargs') if k in rq}; tk['add_generation_prompt'] = True
    tok = requests.post(VLLM_URL + '/tokenize', json=tk, timeout=300).json().get('tokens')
    got = first['response'].get('prompt_token_ids') or first.get('prompt_token_ids_tokenize')
    RESULTS['token_id_crosscheck'] = {'tokenize_len': len(tok or []), 'captured_len': len(got or []), 'identical': tok == got}
except Exception:
    RESULTS['token_id_crosscheck'] = {'error': traceback.format_exc()[-800:]}
print('cross-check:', RESULTS['token_id_crosscheck']); save_results()
RESULTS['smoke_gate'] = gate
if not all(gate.values()):
    raise RuntimeError(f'SMOKE GATE FAILED, stopping before the expensive rounds: {gate} {st}')
print('smoke gate passed:', gate)

# ------------------------------------------------------------------ full rounds over all 75 tasks (2 attempts per task, order shuffled per round)
for k in (1, 2):
    elapsed_min = (time.time() - T_START) / 60
    est_round_min = 100
    if elapsed_min + est_round_min > BUDGET_MIN - 10:
        print(f'not starting round {k}: elapsed {elapsed_min:.0f} min + ~{est_round_min} min would exceed the budget'); break
    ids = list(ALL_IDS); random.Random(100 + k).shuffle(ids)
    run_round(f'r{k}', ids)

print('\\n===== G2 SUMMARY =====')
for tag in ('smoke', 'r1', 'r2'):
    if tag in RESULTS and 'n' in RESULTS[tag]: print(tag, {k: v for k, v in RESULTS[tag].items() if k not in ('resolved_ids', 'proxy')})
print('capture:', proxy.stats(), '| file MB:', round((G2 / 'capture.jsonl.gz').stat().st_size / 1e6, 1))
RESULTS['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); RESULTS['capture'] = proxy.stats(); save_results()
_wd.cancel()'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


nb = {"cells": [
    md("# G2: data collection with exact request/response capture (B1 policy, 75 Kaggle-gold-passing tasks, 2 rounds)\n\nSmoke round with a fail-fast capture gate, then rounds r1, r2. "
       "Output: `g2/capture.jsonl.gz` (every vLLM chat request + response incl. token ids), `g2/<round>/task_results.jsonl` + traces, `g2/g2_summary.json`."),
    code(e1.INSTALL), md("## Setup"), code(SETUP.replace("__CODE_FILES__", json.dumps(CODE_FILES)).replace("__TASK_IDS__", json.dumps(TASK_IDS))),
    md("## vLLM (real 31B, no adapter) + capture proxy"), code(VLLM), md("## Smoke round, capture gate, full rounds"), code(RUN)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5}
(OUT / "g2.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g2-collect", "title": "Gemma 4 G2 Collect", "code_file": "g2.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g2-arm"], "kernel_sources": [],
    "competition_sources": ["gemma-4-developer-agent"], "model_sources": ["google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    "machine_shape": "NvidiaL4"}, indent=2), encoding="utf-8")
print("wrote", OUT, "| tasks:", len(TASK_IDS))
