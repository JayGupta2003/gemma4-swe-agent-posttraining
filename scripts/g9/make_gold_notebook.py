"""Generate kaggle_assets/g9_gold/ : OFFLINE CPU notebook that runs the harness's own two-phase verification (swegemma 0.2.11 verify_task, subprocess sandbox,
shared wheel set) over the pool built by g9_build:
  1. the gold patch must be RESOLVED (all tests in the test-patch files pass and every FAIL_TO_PASS / PASS_TO_PASS node passes, by the harness's own junit check)
  2. for the tasks that pass, an EMPTY patch must NOT be resolved (otherwise the task does not test the fix)
Tasks that satisfy both form the rollout pool. Mirrors scripts/make_gold_check_notebook.py, ported to swegemma 0.2.11 (no setup_gemma_model_registry).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import make_experiment1 as e1  # noqa: E402  (INSTALL cell)

OUT = ROOT / "kaggle_assets" / "g9_gold"
OUT.mkdir(parents=True, exist_ok=True)

RUN = '''import asyncio, json, logging, time, traceback
from adk_eval_core.tracing import SessionTrace
from adk_submission import ModelRegistry, discover_declared_models
from google.adk.models.lite_llm import LiteLlm
from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks

class _DropEnsurepipNoise(logging.Filter):
    def filter(self, record): return 'ensurepip unavailable' not in record.getMessage()
logging.getLogger('swegemma.sandbox.subprocess').addFilter(_DropEnsurepipNoise())

POOL = Path(next(Path('/kaggle/input').rglob('tasks.jsonl'))).parent
ARM = next(p.parent for p in Path('/kaggle/input').rglob('agent.yaml') if 'swe_scaffold_v2_b1' in p.read_text(encoding='utf-8'))
print('pool:', POOL, '| arm:', ARM, '| wheels:', len(list((POOL / 'wheels').glob('*.whl'))), '| snapshots:', len(list((POOL / 'snapshots').glob('*.tgz'))))
WORK = Path('/kaggle/working'); (WORK / 'gold').mkdir(exist_ok=True)
# In subprocess mode swegemma does NOT inject the wheel set (install_test_dependencies returns early); the sandbox venv only sees the host's site-packages through a
# _host_env.pth built from sys.path. So unpack the shared wheels into a side directory and APPEND it to sys.path (host packages keep priority).
import subprocess, sys
exec(__TASKDEPS__)   # defines build_taskdeps / classify (pytest, its core deps and pytest plugins must NOT go into the side dir; see scripts/g9/taskdeps.py)
TD = Path('/tmp/taskdeps/site-packages')
_rep = build_taskdeps(POOL / 'wheels', TD)
print('taskdeps: rc', _rep['rc'], '| installed', _rep['installed'], '| denied', len(_rep['denied']), _rep['denied'][:12], '| pytest plugins excluded', len(_rep['plugins_excluded']), _rep['plugins_excluded'][:12], '| pytest11 entry points stripped', len(_rep['pytest_entry_points_stripped']), _rep['pytest_entry_points_stripped'][:15])
assert _rep['rc'] == 0, _rep['stderr_tail']
sys.path.append(str(TD))
reg = ModelRegistry()
for m in discover_declared_models(ARM): reg.register(m, LiteLlm(model='openai/' + m, api_base='http://127.0.0.1:8000/v1', api_key='EMPTY'))
config = EvalConfig(tasks_path=POOL / 'tasks.jsonl', snapshots_dir=POOL / 'snapshots', results_dir=WORK / 'gold', submission_dir=ARM, models=reg, sandbox='subprocess',
    graph_dir=str(WORK / 'no_graphs'), embeddings_dir=str(WORK / 'no_embeddings'), wheels_dir=POOL / 'wheels')
evaluator = Evaluator(config)
tasks = load_tasks(config.tasks_path)
LIMIT = __LIMIT__
if LIMIT: tasks = tasks[:LIMIT]
print('tasks:', len(tasks), flush=True)
results = {}
sem = asyncio.Semaphore(4)
BUDGET_S = __BUDGET_H__ * 3600; T0 = time.time()

async def one(task, patch):
    t0 = time.perf_counter()
    try:
        snap, base, patch_path = resolve_task_snapshot_paths(config.snapshots_dir, task.instance_id, task.repo)
        res = await verify_task(evaluator.docker, config, task, snap, base_snapshot_path=base, patch_path=patch_path, agent_patch=patch, agent_error=None,
                                trace=SessionTrace(), start_time=t0)
        return {'resolved': bool(res.resolved), 'error': res.error, 'exit': res.test_exit_code, 'tail': (res.test_output or '')[-1200:] if not res.resolved else '', 'sec': round(time.perf_counter() - t0, 1)}
    except Exception as e:
        return {'resolved': False, 'error': f'{type(e).__name__}: {e}'[:400], 'exit': -1, 'tail': '', 'sec': round(time.perf_counter() - t0, 1)}

async def check(task):
    async with sem:
        if time.time() - T0 > BUDGET_S:
            results[task.instance_id] = {'repo': task.repo, 'gold': {'resolved': False, 'error': 'skipped_budget'}}; return
        g = await one(task, task.patch)
        rec = {'repo': task.repo, 'gold': g}
        if g['resolved']:
            rec['empty'] = await one(task, '')
        results[task.instance_id] = rec
        done = len(results)
        print(f"[{done}/{len(tasks)}] {task.instance_id}: gold {'PASS' if g['resolved'] else 'FAIL'} ({g['sec']}s)" + (f" | empty {'RESOLVED(!)' if rec['empty']['resolved'] else 'fails (good)'}" if 'empty' in rec else ''), flush=True)
        (WORK / 'gold_results.json').write_text(json.dumps(results, indent=1), encoding='utf-8')

async def main():
    await asyncio.gather(*(check(t) for t in tasks))

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = None
if loop is not None and loop.is_running():
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(lambda: asyncio.run(main())).result()
else:
    asyncio.run(main())

(WORK / 'gold_results.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
good = sorted(i for i, r in results.items() if r['gold']['resolved'] and 'empty' in r and not r['empty']['resolved'])
(WORK / 'rollout_pool.json').write_text(json.dumps(good), encoding='utf-8')
print('\\n===== harness gold check on the new pool =====')
print(f"gold passes {sum(r['gold']['resolved'] for r in results.values())}/{len(results)} | empty patch also resolved {sum(1 for r in results.values() if 'empty' in r and r['empty']['resolved'])} | ROLLOUT POOL {len(good)} tasks, {len({results[i]['repo'] for i in good})} repos")
errs = {}
for i, r in results.items():
    if not r['gold']['resolved']: errs.setdefault((r['gold']['error'] or 'tests failed')[:90], []).append(i)
print('\\nFAIL reasons:')
for k, v in sorted(errs.items(), key=lambda kv: -len(kv[1]))[:20]: print(f'  {len(v):3d}  {k}   e.g. {v[:2]}')'''


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
budget_h = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
TASKDEPS_SRC = (ROOT / "scripts" / "g9" / "taskdeps.py").read_text(encoding="utf-8").replace(chr(13) + chr(10), chr(10))
nb = {"cells": [code(e1.INSTALL), code(RUN.replace("__LIMIT__", str(limit)).replace("__BUDGET_H__", str(budget_h)).replace("__TASKDEPS__", json.dumps(TASKDEPS_SRC)))],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "gold.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g9-gold", "title": "Gemma 4 G9 Gold", "code_file": "gold.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g2-arm"],
    "kernel_sources": ["noobhunterjay/gemma-4-g9-build"], "competition_sources": [], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT, "| limit", limit, "| budget h", budget_h)
