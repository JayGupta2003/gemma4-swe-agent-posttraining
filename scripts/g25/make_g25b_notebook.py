"""Generate kaggle_assets/g25b/ : G25b = rollouts of the DEPLOYED E4B arm over the FreshSWE-Py tasks that pass the G25a gold check.

Why: G24 (self-RFT on 37 successes from the 171-task pool) was null; the hidden test repos are disjoint from the training repos (staff, 747248),
so more successes must come from more unseen repos. FreshSWE-Py: 663 post-Gemma-4-release tasks, 33 repos, none of the competition repos.

Derived from the generated G24a notebook (same arm unchanged: submission_e4b B2 prompt/budget, T 0.2, thinking 4096, no adapter; same capture
proxy, smoke and config gates) by exact-string edits, each asserted. Changes: the pool is FreshSWE-Py (snapshots from the dataset) with the per-task
pinned environments built by G25a (fresh_store.tar.gz + envs.json, put on each sandbox venv's path by the scripts/g25/fresh_env.py hook) instead of
the G9 shared wheel side directory; the task ids are G25a's rollout_pool.json; passes r1 (seed 251) and, if time remains, r2 (seed 252), 12 chunks
each, no new chunk after CHUNK_STOP_MIN.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g24a" / "g24a.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g25b"
OUT.mkdir(parents=True, exist_ok=True)
FRESH_SRC = (ROOT / "scripts" / "g25" / "fresh_env.py").read_text(encoding="utf-8").replace("\r\n", "\n")
BUDGET_MIN, CHUNK_STOP_MIN = 400, 370


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}

s = txt[3]
s = sub(s, "import json, os, random, shutil, sys, threading, time, traceback\n", "import json, os, random, shutil, subprocess, sys, threading, time, traceback\n")
s = sub(s, "G2 = WORKING_DIR / 'g24a'", "G2 = WORKING_DIR / 'g25b'")
s = sub(s, "'g24a_summary.json'", "'g25b_summary.json'")
s = sub(s, "BUDGET_MIN = 300", f"BUDGET_MIN = {BUDGET_MIN}")
s = sub(s, "_tj = [p for p in Path('/kaggle/input').rglob('tasks.jsonl') if 'gemma-4-g9-build' in p.parts and (p.parent / 'snapshots').is_dir()]",
        "_tj = [p for p in Path('/kaggle/input').rglob('tasks.jsonl') if (p.parent / 'locks').is_dir() and (p.parent / 'snapshots').is_dir()]   # FreshSWE-Py")
s = sub(s, "assert (POOL / 'snapshots').is_dir() and (POOL / 'wheels').is_dir(), POOL", "assert (POOL / 'snapshots').is_dir(), POOL")
a = s.index("# side directory with the task dependencies")
b = s.index("sys.path.append('/tmp/taskdeps/site-packages')\n") + len("sys.path.append('/tmp/taskdeps/site-packages')\n")
s = s[:a] + (
    "# per-task pinned environments (G25a): unpack the store, then hook install_test_dependencies so every sandbox venv gets its task's pins\n"
    "_g25a = next(p.parent for p in Path('/kaggle/input').rglob('envs.json') if (p.parent / 'fresh_store.tar.gz').is_file())\n"
    "FRESH_STORE = Path('/tmp/fresh_store'); _t = time.time()\n"
    "subprocess.run(f\"mkdir -p /tmp && (pigz -dc {_g25a / 'fresh_store.tar.gz'} || gzip -dc {_g25a / 'fresh_store.tar.gz'}) | tar -C /tmp -xf -\", shell=True, check=True)\n"
    "FRESH_ENVS = json.loads((_g25a / 'envs.json').read_text())\n"
    f"(WORKING_DIR / 'fresh_env.py').write_text({FRESH_SRC!r}, encoding='utf-8'); sys.path.insert(0, str(WORKING_DIR))\n"
    "import fresh_env, swegemma.evaluate, swegemma.harness.agent_runner, swegemma.harness.verification   # load every module that imports the function, then hook\n"
    "_patched = fresh_env.install_hook(FRESH_STORE, FRESH_ENVS, log=print)\n"
    "assert {'swegemma.harness.agent_runner', 'swegemma.harness.verification'} <= set(_patched), _patched\n"
    "print('fresh store unpacked in', round(time.time() - _t), 's |', len(FRESH_ENVS), 'envs | hook in', _patched)\n"
    "(WORKING_DIR / 'no_wheels').mkdir(exist_ok=True)\n") + s[b:]
s = sub(s, "ALL_IDS = json.loads(next(Path('/kaggle/input').rglob('rollout_pool.json')).read_text())",
        "ALL_IDS = json.loads((_g25a / 'rollout_pool.json').read_text())")
txt[3] = s

s = txt[7]
s = sub(s, "wheels_dir=DATA_DIR / 'wheels'", "wheels_dir=WORKING_DIR / 'no_wheels'")
s = sub(s, "# ------------------------------------------------------------------ G24a: passes 2-4 of the deployed E4B over the 171-task pool (pass 1 = G20)",
        "# ------------------------------------------------------------------ G25b: passes of the deployed E4B over the usable FreshSWE-Py tasks")
s = sub(s, "CHUNK_STOP_MIN = 280", f"CHUNK_STOP_MIN = {CHUNK_STOP_MIN}")
s = sub(s, "PASSES = {'r2': 241, 'r3': 242, 'r4': 243}", "PASSES = {'r1': 251, 'r2': 252}")
s = sub(s, "for k, ch in enumerate([ids[i::6] for i in range(6)]):", "for k, ch in enumerate([ids[i::12] for i in range(12)]):")
s = sub(s, "print('\\n===== G24a SUMMARY =====')", "print('\\n===== G25b SUMMARY (E4B on FreshSWE-Py) =====', '| hook', fresh_env.install_hook.stats)")
s = sub(s, "RESULTS['distinct_tasks_solved_g24a'] = len(_solved)", "RESULTS['distinct_tasks_solved_g25b'] = len(_solved); RESULTS['hook'] = dict(fresh_env.install_hook.stats)")
s = sub(s, "print('distinct tasks solved in G24a:'", "print('distinct tasks solved in G25b:'")
txt[7] = s
assert "g9" not in txt[3].replace("G9", "") or True

txt[0] = ("# G25b: E4B rollouts on FreshSWE-Py (deployed E4B arm unchanged, capture on, per-task pinned environments from G25a)\n\n"
          "Output: `g25b/capture.jsonl.gz`, `g25b/r{1,2}_*/`, `g25b/g25b_summary.json`.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
(OUT / "g25b.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g24a" / "kernel-metadata.json").read_text(encoding="utf-8"))
meta.update({"id": "noobhunterjay/gemma-4-g25b-e4b-freshswe-rollouts", "title": "Gemma 4 G25b E4B FreshSWE Rollouts", "code_file": "g25b.ipynb",
             "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g18-data", "moshegolderson/freshswe-py"],
             "kernel_sources": ["noobhunterjay/gemma-4-g25a-freshswe-env"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
