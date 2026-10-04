"""Generate kaggle_assets/gold_check/ : a CPU-ONLY notebook (no GPU, no model, no L4x4 quota) that runs the
REFERENCE patch of every public task through the same two-phase verification the scorer uses, inside
Kaggle's own environment (official image, official wheelhouse, subprocess sandbox).

Why: locally only 52/129 public tasks pass their own gold patch, but staff say local failures are caused by our
local environment, not the scorer. Every agent notebook already runs inside Kaggle's environment, so the set of
tasks whose gold patch passes THERE is the set that is actually usable as dev/training data. Growing that pool
is what makes small effects in scaffold experiments detectable at all.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle_assets" / "gold_check"
OUT.mkdir(parents=True, exist_ok=True)

INSTALL = '''import glob, importlib, os, subprocess, sys
from pathlib import Path
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['OTEL_SDK_DISABLED'] = 'true'
WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
for pat in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth', '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for pth in glob.glob(pat):
        try: os.unlink(pth)
        except OSError: pass
tmp = Path('/tmp/wheelhouse'); tmp.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower(): continue
    name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    t = tmp / name
    if not t.exists(): os.symlink(w, t)
wheels = sorted(str(w) for w in tmp.glob('*.whl'))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()
print('installed', len(wheels), 'wheels; cpu count =', os.cpu_count())'''

RUN = '''import asyncio, json, logging, time, traceback
from adk_eval_core.tracing import SessionTrace
from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks, setup_gemma_model_registry

class _DropEnsurepipNoise(logging.Filter):   # verified-harmless fallback warning, 2 lines per sandbox
    def filter(self, record): return 'ensurepip unavailable' not in record.getMessage()
logging.getLogger('swegemma.sandbox.subprocess').addFilter(_DropEnsurepipNoise())

DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WORK = Path('/kaggle/working'); (WORK / 'gold').mkdir(exist_ok=True)
config = EvalConfig(tasks_path=DATA_DIR / 'tasks.jsonl', snapshots_dir=DATA_DIR / 'snapshots', results_dir=WORK / 'gold',
    submission_dir=DATA_DIR / 'sample_submission', models=setup_gemma_model_registry(), sandbox='subprocess',
    graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'), wheels_dir=DATA_DIR / 'wheels')
evaluator = Evaluator(config)
tasks = load_tasks(config.tasks_path)
print('tasks:', len(tasks))
results = {}
sem = asyncio.Semaphore(4)

async def check(task):
    async with sem:
        t0 = time.perf_counter()
        try:
            snap, base, patch_path = resolve_task_snapshot_paths(config.snapshots_dir, task.instance_id, task.repo)
            res = await verify_task(evaluator.docker, config, task, snap, base_snapshot_path=base, patch_path=patch_path,
                                    agent_patch=task.patch, agent_error=None, trace=SessionTrace(), start_time=t0)
            rec = {'resolved': bool(res.resolved), 'error': res.error, 'exit': res.test_exit_code,
                   'tail': (res.test_output or '')[-1500:] if not res.resolved else ''}
        except Exception as e:
            rec = {'resolved': False, 'error': f'{type(e).__name__}: {e}'[:400], 'exit': -1, 'tail': ''}
        rec.update(repo=task.repo, seconds=round(time.perf_counter() - t0, 1))
        results[task.instance_id] = rec
        done = len(results)
        print(f"[{done}/{len(tasks)}] {task.instance_id}: {'PASS' if rec['resolved'] else 'FAIL'} ({rec['seconds']}s)", flush=True)
        (WORK / 'gold_results.json').write_text(json.dumps(results, indent=1), encoding='utf-8')   # incremental

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

by = {}
for iid, r in results.items():
    by.setdefault(r['repo'], [0, 0]); by[r['repo']][1] += 1; by[r['repo']][0] += r['resolved']
print('\\n===== gold patch passes in Kaggle env =====')
print(f"TOTAL {sum(r['resolved'] for r in results.values())}/{len(results)}")
for repo, (p, n) in sorted(by.items()): print(f'  {repo}: {p}/{n}')
print('\\nPASSING:', sorted(i for i, r in results.items() if r['resolved']))
errs = {}
for i, r in results.items():
    if not r['resolved']: errs.setdefault((r['error'] or 'tests failed')[:90], []).append(i)
print('\\nFAIL reasons:')
for k, v in sorted(errs.items(), key=lambda kv: -len(kv[1]))[:12]: print(f'  {len(v):3d}  {k}   e.g. {v[:3]}')'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


nb = {"cells": [
    md("# Gold-patch check in Kaggle's own environment (CPU only)\n\nRuns each public task's reference patch through the scorer's two-phase verification "
       "(subprocess sandbox, official wheelhouse) to find which of the 129 tasks are actually usable as dev/training data. Locally only 52/129 pass; staff say "
       "local failures are our environment's fault. No GPU, no model."),
    code(INSTALL), code(RUN)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5}
(OUT / "gold_check.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-gold-check", "title": "Gemma 4 Gold Check",
    "code_file": "gold_check.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"], "kernel_sources": [],
    "competition_sources": ["gemma-4-developer-agent"], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT)
