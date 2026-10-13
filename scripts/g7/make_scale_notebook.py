"""Generate kaggle_assets/g8_scale_<shard>/ : one shard of the CPU, internet-enabled task-validation job (same worker as the G7 spike, scripts/g7/spike_worker.py).

Usage: python scripts/g7/make_scale_notebook.py <candidates.jsonl> <shard_index> <n_shards> [budget_hours=9]
Tasks are dealt round-robin to the shards (so each shard has the same mix); results are saved incrementally to /kaggle/working/scale_results_<shard>.json; no new task is
started after the time budget (Kaggle CPU sessions end after 12 h); the freeze list is kept only for tasks that are valid / p2p_partial.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
cand, shard, n_shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
budget_h = float(sys.argv[4]) if len(sys.argv) > 4 else 9.0
tasks = [json.loads(l) for l in open(cand, encoding="utf-8")]
mine = [t for i, t in enumerate(tasks) if i % n_shards == shard]
OUT = ROOT / "kaggle_assets" / f"g8_scale_{shard}"
OUT.mkdir(parents=True, exist_ok=True)
WORKER = (HERE / "spike_worker.py").read_text(encoding="utf-8").replace("\r\n", "\n")

RUN = '''import json, os, time, subprocess, sys, collections, concurrent.futures as cf
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'virtualenv'], check=True)   # Kaggle's `python -m venv` is broken (ensurepip)
exec(WORKER_SRC)   # defines run_task(task, work_root)
WORK = Path('/tmp/scale'); WORK.mkdir(exist_ok=True)
OUTDIR = Path('/kaggle/working'); OUTDIR.mkdir(exist_ok=True)
RES_PATH = OUTDIR / f'scale_results_{SHARD}.json'
BUDGET_S = BUDGET_H * 3600
t0 = time.time(); results = []
def _one(t):
    if time.time() - t0 > BUDGET_S: return {'id': t['instance_id'], 'repo': t['repo'], 'verdict': 'skipped_budget', 'reason': ''}
    try: r = run_task(t, WORK)
    except Exception as e: r = {'id': t['instance_id'], 'repo': t['repo'], 'verdict': 'harness_error', 'reason': repr(e)[:300]}
    if r.get('verdict') not in ('valid', 'p2p_partial'): r.pop('freeze', None)
    return r
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(_one, t) for t in TASKS]
    for k, f in enumerate(cf.as_completed(futs), 1):
        r = f.result(); results.append(r)
        if k % 10 == 0 or k == len(futs):
            c = collections.Counter(x['verdict'] for x in results)
            print(f"[{time.time() - t0:7.0f}s] {k}/{len(futs)} done | " + ' '.join(f'{a}={b}' for a, b in c.most_common()), flush=True)
            RES_PATH.write_text(json.dumps(results))
RES_PATH.write_text(json.dumps(results))
print('\\nverdicts:', dict(collections.Counter(r['verdict'] for r in results)), '| wall', round(time.time() - t0), 's')
fails = collections.Counter((r['verdict'], (r.get('reason') or '')[:60]) for r in results if r['verdict'] not in ('valid',))
for (v, why), n in fails.most_common(25): print(f'  {n:3d} {v:17s} {why!r}')'''


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


# the candidate list lives in a small private dataset (a 7 MB notebook is rejected by Kaggle with HTTP 400)
DS = ROOT / "kaggle_assets" / "g8_candidates_dataset"
DS.mkdir(parents=True, exist_ok=True)
(DS / "candidates.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8", newline="\n")
(DS / "dataset-metadata.json").write_text(json.dumps({"title": "gemma4-g8-candidates", "id": "noobhunterjay/gemma4-g8-candidates",
                                                      "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
head = (f"import json\nfrom pathlib import Path\nSHARD = {shard}\nN_SHARDS = {n_shards}\nBUDGET_H = {budget_h}\nWORKER_SRC = {json.dumps(WORKER)}\n"
        "_p = next(Path('/kaggle/input').rglob('candidates.jsonl'))\n"
        "_all = [json.loads(l) for l in open(_p, encoding='utf-8')]\n"
        "TASKS = [t for i, t in enumerate(_all) if i % N_SHARDS == SHARD]\nprint(len(_all), 'candidates;', len(TASKS), 'in this shard')")
nb = {"cells": [code("import os, sys\nprint(sys.version, '| PYTHONPATH =', os.environ.get('PYTHONPATH'))"), code(head), code(RUN)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "scale.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": f"noobhunterjay/gemma-4-g8-scale-{shard}", "title": f"Gemma 4 G8 Scale {shard}", "code_file": "scale.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "keywords": [], "dataset_sources": ["noobhunterjay/gemma4-g8-candidates"], "kernel_sources": [],
    "competition_sources": [], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT, "|", len(mine), "of", len(tasks), "tasks")
