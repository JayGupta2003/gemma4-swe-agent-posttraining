"""Generate kaggle_assets/g7_spike/ : CPU, INTERNET-ENABLED notebook that measures how many SWE-rebench tasks can be turned into validated tasks on Kaggle's
Python 3.12 image (clone at base_commit, venv, install, apply test_patch -> FAIL_TO_PASS must fail, apply gold patch -> FAIL_TO_PASS and PASS_TO_PASS must pass).

The notebook only measures yield and cost; no GPU. The worker function (`run_task`) is the same code as in scripts/g7/spike_worker.py, which is dry-run locally first.

Usage: python scripts/g7/make_spike_notebook.py <spike_tasks.jsonl>
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "kaggle_assets" / "g7_spike"
OUT.mkdir(parents=True, exist_ok=True)
tasks = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
WORKER = (HERE / "spike_worker.py").read_text(encoding="utf-8").replace("\r\n", "\n")

RUN = '''import json, os, time, concurrent.futures as cf
from pathlib import Path
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'virtualenv'], check=True)   # Kaggle's `python -m venv` is broken (ensurepip)
exec(WORKER_SRC)   # defines run_task(task, work_root)
WORK = Path('/tmp/spike'); WORK.mkdir(exist_ok=True)
OUTDIR = Path('/kaggle/working'); OUTDIR.mkdir(exist_ok=True)
t0 = time.time(); results = []
def _one(t):
    try: return run_task(t, WORK)
    except Exception as e: return {'id': t['instance_id'], 'verdict': 'harness_error', 'reason': repr(e)[:300]}
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(_one, t): t['instance_id'] for t in TASKS}
    for f in cf.as_completed(futs):
        r = f.result(); results.append(r)
        print(f"[{time.time() - t0:6.0f}s] {r['id']:55s} {r['verdict']:14s} {r.get('reason', '')[:100]}", flush=True)
        (OUTDIR / 'spike_results.json').write_text(json.dumps(results, indent=1))
import collections
print('\\nverdicts:', dict(collections.Counter(r['verdict'] for r in results)), '| wall', round(time.time() - t0), 's')
print('valid tasks:', [r['id'] for r in results if r['verdict'] == 'valid'])'''

SETUP = "import sys, subprocess\nprint(sys.version)\nprint(subprocess.run(['git', '--version'], capture_output=True, text=True).stdout)\nprint(subprocess.run(['pip', '--version'], capture_output=True, text=True).stdout)"


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


nb = {"cells": [code(SETUP), code("import json\nWORKER_SRC = " + json.dumps(WORKER) + "\nTASKS = json.loads(" + repr(json.dumps(tasks)) + ")"), code(RUN)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "spike.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g7-task-spike", "title": "Gemma 4 G7 Task Spike", "code_file": "spike.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "keywords": [], "dataset_sources": [], "kernel_sources": [],
    "competition_sources": [], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT, "|", len(tasks), "tasks")
