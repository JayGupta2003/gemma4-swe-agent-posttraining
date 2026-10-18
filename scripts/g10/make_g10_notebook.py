"""Generate kaggle_assets/g10/ : rollouts of the 31B (B1 arm, unchanged prompt/sampling/5-minute cap, capture proxy with exact token ids) over the 171-task pool built in G9.

Derived from the G2 notebook (kaggle_assets/g2/g2.ipynb) by exact-string edits, each asserted:
  - tasks, snapshots and wheels come from the g9-build kernel output (pool/), task ids from the g9-gold kernel output (rollout_pool.json)
  - before the Evaluator runs, the shared wheel set is unpacked into a side directory appended to sys.path (scripts/g9/taskdeps.py: pytest/plugins excluded, [pytest11] entry
    points stripped), because swegemma's subprocess sandbox never injects wheels and the sandbox venv inherits the notebook's sys.path
  - no code graphs/embeddings exist for these tasks: empty directories (get_code_neighbors/subgraph return errors; <1% of calls)
  - 4-task smoke with the capture gate, then round p1 over all pool tasks; a second round only if the time budget allows (it will not at the default budget)
Cost: ~15 min boot + 6 min smoke + ~3.3 h round = ~3.7 h wall = ~7.4 quota h on L4x4. 300-minute watchdog.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g2" / "g2.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g10"
OUT.mkdir(parents=True, exist_ok=True)
TASKDEPS_SRC = (ROOT / "scripts" / "g9" / "taskdeps.py").read_text(encoding="utf-8").replace("\r\n", "\n")


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
assert [c["cell_type"] for c in cells] == ["markdown", "code", "markdown", "code", "markdown", "code", "markdown", "code"]

# ---------------------------------------------------------------- SETUP (cell 3)
s = txt[3]
s = sub(s, "G2 = WORKING_DIR / 'g2'", "G2 = WORKING_DIR / 'g10'")
# fail in the first seconds, not after the model load, if the machine is not the expected L4x4
s = sub(s, "RESULTS = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "RESULTS = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "import torch\n"
        "_gpus = [(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]\n"
        "print('GPUs:', _gpus)\n"
        "assert len(_gpus) == 4 and all(c[0] >= 8 for _, c in _gpus), f'expected 4x L4 (compute capability >= 8), got {_gpus}'\n")
s = sub(s, "'g2_summary.json'", "'g10_summary.json'")
s = sub(s, "BUDGET_MIN = 235", "BUDGET_MIN = 300")
s = sub(s, "DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')\n",
        "_tj = [p for p in Path('/kaggle/input').rglob('tasks.jsonl') if 'competitions' not in p.parts and (p.parent / 'snapshots').is_dir()]   # NOT the competition's own file\n"
        "assert len(_tj) == 1, _tj\n"
        "POOL = _tj[0].parent\nDATA_DIR = POOL\n"
        "assert (POOL / 'snapshots').is_dir() and (POOL / 'wheels').is_dir(), POOL\n"
        "# side directory with the task dependencies (see scripts/g9/taskdeps.py): must exist on sys.path BEFORE any sandbox is created\n"
        "exec(__TASKDEPS__)\n"
        "_rep = build_taskdeps(POOL / 'wheels', Path('/tmp/taskdeps/site-packages'))\n"
        "print('taskdeps: rc', _rep['rc'], '| installed', _rep['installed'], '| denied', len(_rep['denied']), '| pytest11 stripped', len(_rep['pytest_entry_points_stripped']))\n"
        "assert _rep['rc'] == 0, _rep['stderr_tail']\n"
        "sys.path.append('/tmp/taskdeps/site-packages')\n"
        "(WORKING_DIR / 'no_graphs').mkdir(exist_ok=True); (WORKING_DIR / 'no_embeddings').mkdir(exist_ok=True)\n")
i0 = s.index("ALL_IDS = [")
i1 = s.index("\n", i0)
s = s[:i0] + "ALL_IDS = json.loads(next(Path('/kaggle/input').rglob('rollout_pool.json')).read_text())" + s[i1:]
s = sub(s, "SMOKE_IDS = random.Random(5).sample([i for i in ALL_IDS if i.startswith(('fastapi_', 'rich_'))], 4)", "SMOKE_IDS = random.Random(5).sample(ALL_IDS, 4)")
# the harness adds the graph-tool paragraph to the task prompt ONLY when a graph json and an embeddings npz (> 100 bytes) exist for the task; the hidden test tasks have them,
# so write tiny stub files to keep the prompt identical to test time (the graph tools then just return errors if the agent uses them: < 1% of calls)
s = sub(s, "by_id = {t.instance_id: t for t in load_tasks(TASKS_PATH)}\n",
        "by_id = {t.instance_id: t for t in load_tasks(TASKS_PATH)}\n"
        "import numpy as _np\n"
        "for _t in by_id.values():\n"
        "    _short = _t.repo.split('/')[-1]\n"
        "    (WORKING_DIR / 'no_graphs' / f'{_short}_{_t.base_commit}.json').write_text(json.dumps({'directed': True, 'multigraph': False, 'graph': {}, 'nodes': [], 'links': [], 'note': 'stub graph for rollouts, no code graph exists for this task'}))\n"
        "    _np.savez(WORKING_DIR / 'no_embeddings' / f'{_short}_{_t.base_commit}.npz', ids=_np.array(['stub'] * 8), vectors=_np.zeros((8, 256), dtype='float32'))\n"
        "print('stub graphs/embeddings written for', len(by_id), 'tasks')\n")
txt[3] = s.replace("__TASKDEPS__", json.dumps(TASKDEPS_SRC))

# ---------------------------------------------------------------- RUN (cell 7)
s = txt[7]
s = sub(s, "graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings')", "graph_dir=str(WORKING_DIR / 'no_graphs'), embeddings_dir=str(WORKING_DIR / 'no_embeddings')")
s = sub(s, "est_round_min = 100", "est_round_min = 215")
s = sub(s, "run_round(f'r{k}', ids)", "run_round(f'p{k}', ids)")
s = sub(s, "for tag in ('smoke', 'r1', 'r2'):", "for tag in ('smoke', 'p1', 'p2'):")
s = sub(s, "G2 SUMMARY", "G10 SUMMARY")
txt[7] = s

txt[0] = ("# G10: rollouts of the 31B (B1 arm) over the G9 pool with exact request/response capture\n\n"
          "Smoke round + capture gate, then round `p1` over the pool. Output: `g10/capture.jsonl.gz`, `g10/p1/task_results.jsonl`, traces, `g10/g10_summary.json`.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")

(OUT / "g10.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g2" / "kernel-metadata.json").read_text(encoding="utf-8"))
# competition_sources MUST stay attached: the L4x4 machine is only granted to notebooks attached to the competition (G10 v1 dropped it and got a T4x2:
# "Bfloat16 is only supported on GPUs with compute capability >= 8.0. Your Tesla T4 ..."). Consequence: the competition's own tasks.jsonl is mounted too.
assert meta["competition_sources"] == ["gemma-4-developer-agent"] and meta["machine_shape"] == "NvidiaL4"
meta.update({"id": "noobhunterjay/gemma-4-g10-rollouts", "title": "Gemma 4 G10 Rollouts", "code_file": "g10.ipynb",
             "kernel_sources": ["noobhunterjay/gemma-4-g9-build", "noobhunterjay/gemma-4-g9-gold"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
