"""Generate kaggle_assets/g25a/ : G25a = FreshSWE-Py environment build + harness gold check (CPU kernel, internet ON, no GPU quota).

Stage 1: install the competition wheelhouse (same host packages as the GPU rollout kernels: same image, same INSTALL cell).
Stage 2: scripts/g25/fresh_env.build_store over all 663 FreshSWE-Py locks -> /kaggle/working/fresh_store (cp312 pins; FreshSWE's wheelhouse is cp313).
Stage 3: the harness's own two-phase check (swegemma verify_task, subprocess sandbox, fresh_env hook on): gold patch must resolve, empty patch must not.
Outputs: fresh_store.tar.gz (the store, one file), envs.json, build_report.json, gold_results.json, rollout_pool.json (usable instance ids).
Usage: python scripts/g25/make_g25a_notebook.py [limit | id,id,...] [gold_budget_hours]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import make_experiment1 as e1  # noqa: E402  (INSTALL cell)

OUT = ROOT / "kaggle_assets" / "g25a"
OUT.mkdir(parents=True, exist_ok=True)
FRESH_SRC = (ROOT / "scripts" / "g25" / "fresh_env.py").read_text(encoding="utf-8").replace("\r\n", "\n")

BUILD = '''import json, os, subprocess, sys, time
from pathlib import Path
WORK = Path('/kaggle/working'); Path('/kaggle/working/fresh_env.py').write_text(FRESH_SRC, encoding='utf-8'); sys.path.insert(0, '/kaggle/working')
import fresh_env as fe
FRESH = next(p.parent for p in Path('/kaggle/input').rglob('tasks.jsonl') if (p.parent / 'locks').is_dir())
WH = FRESH / 'wheelhouse' if (FRESH / 'wheelhouse').is_dir() else None
TASKS = [json.loads(l) for l in open(FRESH / 'tasks.jsonl', encoding='utf-8')]
LIMIT, IDS = __LIMIT__, __IDS__
if IDS:
    TASKS = [t for t in TASKS if t['instance_id'] in set(IDS)]
elif LIMIT:   # spread a small run over repos
    by = {}
    for t in TASKS: by.setdefault(t['repo'], []).append(t)
    TASKS = [t for i in range(LIMIT) for ts in by.values() if i < len(ts) for t in [ts[i]]][:LIMIT]
print('python', sys.version.split()[0], '| FreshSWE at', FRESH, '| wheelhouse', WH, len(list(WH.glob('*.whl'))) if WH else 0, 'wheels', '| tasks', len(TASKS), '| snapshots', len(list((FRESH / 'snapshots').glob('*.tgz'))))
print('top-level:', sorted(p.name for p in FRESH.iterdir()))
STORE = WORK / 'fresh_store'
T0 = time.time()
rep = fe.build_store(TASKS, FRESH / 'locks', STORE, WH, workers=8)
(WORK / 'envs.json').write_text(json.dumps(rep['envs']), encoding='utf-8')
(WORK / 'build_report.json').write_text(json.dumps({k: v for k, v in rep.items() if k != 'envs'}, indent=1), encoding='utf-8')
du = subprocess.run(['du', '-sm', str(STORE)], capture_output=True, text=True).stdout.split()[0]
print('STORE', rep['stats'], '| MB', du, '| tasks with missing pins', len(rep['missing']), '| no lock', len(rep['no_lock']))
print('failed pins (first 30):', list(rep['failed_pins'])[:30])'''

GOLD = '''import asyncio, logging, traceback
from adk_eval_core.tracing import SessionTrace
from adk_submission import ModelRegistry, discover_declared_models
from google.adk.models.lite_llm import LiteLlm
from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import Evaluator
import swegemma.harness.verification
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks

class _DropEnsurepipNoise(logging.Filter):
    def filter(self, record): return 'ensurepip unavailable' not in record.getMessage()
logging.getLogger('swegemma.sandbox.subprocess').addFilter(_DropEnsurepipNoise())

patched = fe.install_hook(STORE, rep['envs'], log=print)
print('hook installed in', patched)
assert 'swegemma.harness.verification' in patched, patched
ARM = next(p.parent for p in Path('/kaggle/input').rglob('agent.yaml') if 'swe_scaffold_v2_b1' in p.read_text(encoding='utf-8'))
(WORK / 'no_wheels').mkdir(exist_ok=True); (WORK / 'gold').mkdir(exist_ok=True)
SEL = WORK / 'tasks_selected.jsonl'
with open(SEL, 'w', encoding='utf-8') as f:
    for t in TASKS: f.write(json.dumps(t) + '\\n')
reg = ModelRegistry()
for m in discover_declared_models(ARM): reg.register(m, LiteLlm(model='openai/' + m, api_base='http://127.0.0.1:8000/v1', api_key='EMPTY'))
config = EvalConfig(tasks_path=SEL, snapshots_dir=FRESH / 'snapshots', results_dir=WORK / 'gold', submission_dir=ARM, models=reg, sandbox='subprocess',
    graph_dir=str(WORK / 'no_graphs'), embeddings_dir=str(WORK / 'no_embeddings'), wheels_dir=WORK / 'no_wheels')
evaluator = Evaluator(config)
tasks = load_tasks(config.tasks_path)
print('tasks loaded by swegemma:', len(tasks), flush=True)
results = {}
sem = asyncio.Semaphore(4)
BUDGET_S = __BUDGET_H__ * 3600; TG = time.time()

async def one(task, patch):
    t0 = time.perf_counter()
    try:
        snap, base, patch_path = resolve_task_snapshot_paths(config.snapshots_dir, task.instance_id, task.repo)
        res = await verify_task(evaluator.docker, config, task, snap, base_snapshot_path=base, patch_path=patch_path, agent_patch=patch, agent_error=None,
                                trace=SessionTrace(), start_time=t0)
        return {'resolved': bool(res.resolved), 'error': res.error, 'exit': res.test_exit_code, 'tail': (res.test_output or '')[-1500:] if not res.resolved else '', 'sec': round(time.perf_counter() - t0, 1)}
    except Exception as e:
        return {'resolved': False, 'error': f'{type(e).__name__}: {e}'[:400], 'exit': -1, 'tail': traceback.format_exc()[-800:], 'sec': round(time.perf_counter() - t0, 1)}

async def check(task):
    async with sem:
        if time.time() - TG > BUDGET_S:
            results[task.instance_id] = {'repo': task.repo, 'gold': {'resolved': False, 'error': 'skipped_budget'}}; return
        g = await one(task, task.patch)
        rec = {'repo': task.repo, 'gold': g}
        if g['resolved']:
            rec['empty'] = await one(task, '')
        results[task.instance_id] = rec
        print(f"[{len(results)}/{len(tasks)}] {task.instance_id}: gold {'PASS' if g['resolved'] else 'FAIL'} ({g['sec']}s)" + (f" | empty {'RESOLVED(!)' if rec['empty']['resolved'] else 'fails (good)'}" if 'empty' in rec else '') + f" | hook {fe.install_hook.stats}", flush=True)
        if len(results) % 20 == 0: (WORK / 'gold_results.json').write_text(json.dumps(results, indent=1), encoding='utf-8')

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
print('\\n===== G25a: harness gold check on FreshSWE-Py (subprocess sandbox + per-task pinned env) =====')
print(f"gold passes {sum(r['gold']['resolved'] for r in results.values())}/{len(results)} | empty patch also resolved {sum(1 for r in results.values() if 'empty' in r and r['empty']['resolved'])} | USABLE {len(good)} tasks, {len({results[i]['repo'] for i in good})} repos | hook {fe.install_hook.stats} | gold stage {(time.time() - TG) / 60:.0f} min")
per = {}
for i, r in results.items():
    a = per.setdefault(r['repo'], [0, 0]); a[0] += 1; a[1] += i in good
print('per repo usable/checked:', {k: f'{v[1]}/{v[0]}' for k, v in sorted(per.items(), key=lambda kv: -kv[1][0])})
errs = {}
for i, r in results.items():
    if not r['gold']['resolved']: errs.setdefault((r['gold']['error'] or 'tests failed')[:90], []).append(i)
print('\\nFAIL reasons:')
for k, v in sorted(errs.items(), key=lambda kv: -len(kv[1]))[:20]: print(f'  {len(v):3d}  {k}   e.g. {v[:3]}')
fails = [i for i, r in results.items() if not r['gold']['resolved'] and r['gold']['error'] is None][:6]
for i in fails: print('\\n---', i, '\\n', results[i]['gold']['tail'][-900:])
# package the store as one file (thousands of small files are slow as kernel output), then drop the directory
subprocess.run(f'tar -C {WORK} -cf - fresh_store | pigz -p 4 > {WORK}/fresh_store.tar.gz || tar -C {WORK} -czf {WORK}/fresh_store.tar.gz fresh_store', shell=True, check=True)
print('tar MB', round((WORK / 'fresh_store.tar.gz').stat().st_size / 1e6))
import shutil; shutil.rmtree(STORE); shutil.rmtree(WORK / 'gold', ignore_errors=True)
print('total minutes', round((time.time() - T0) / 60))'''


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


arg = sys.argv[1] if len(sys.argv) > 1 else "0"
ids = arg.split(",") if not arg.isdigit() else []
limit = int(arg) if arg.isdigit() else 0
budget_h = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
cells = [code(e1.INSTALL), code("FRESH_SRC = " + json.dumps(FRESH_SRC)), code(BUILD.replace("__LIMIT__", str(limit)).replace("__IDS__", json.dumps(ids))),
         code(GOLD.replace("__BUDGET_H__", str(budget_h)))]
for c in cells:
    compile("".join(c["source"]), "cell", "exec")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "g25a.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g25a-freshswe-env", "title": "Gemma 4 G25a FreshSWE Env", "code_file": "g25a.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "keywords": [],
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g2-arm", "moshegolderson/freshswe-py"],
    "kernel_sources": [], "competition_sources": [], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT, "| limit", limit, "| gold budget h", budget_h)
