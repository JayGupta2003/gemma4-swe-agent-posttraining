"""Generate kaggle_assets/g26a/ : G26a = self-RFT iteration 2, step 1: rollouts of the G25c ADAPTER (E4B + LoRA) over the FreshSWE-Py tasks.

Why: G25c (E4B LoRA on 152 own successes, pool + FreshSWE-Py) gave e1 12 / e2 14 of 75 vs base mean 9 (WEAK by the pre-registered rule, but
2-round distinct 19 vs base 10-15). Expert iteration: sample the improved policy on the same task distribution, keep its new test-verified
successes, retrain from the base on the union (G26b, next quota week).

Derived from the generated G25c notebook (kaggle_assets/g25c/g25c.ipynb: same E4B arm, vLLM + LoRA serving, capture proxy, adapter-effect gate,
smoke gate) by exact-string edits, each asserted:
  - training cell -> copy the G25c scorer-format eval arm from the G25c kernel output (sha256 of the adapter checked against the archived copy,
    which is also the one in submission_e4b_g25c), the packaging cell -> wiring check only;
  - setup: the FreshSWE-Py pool + the G25a per-task environment store and install_test_dependencies hook (as G25b), stub graphs/embeddings;
  - evaluation: run_round(..., pool=True) points the harness at the FreshSWE tasks/snapshots; the smoke round stays on 4 competition tasks
    (gate: requests really use the adapter); then chunks of 50 FreshSWE tasks, tasks NOT solved in G25b first (seed 261), then the solved ones
    (seed 262), no new chunk after CHUNK_STOP_MIN. No 75-task evaluation round (that is G26b's job after retraining).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g25c" / "g25c.ipynb").read_text(encoding="utf-8"))
G25B_NB = json.loads((ROOT / "kaggle_assets" / "g25b" / "g25b.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g26a"
OUT.mkdir(parents=True, exist_ok=True)
FRESH_SRC = (ROOT / "scripts" / "g25" / "fresh_env.py").read_text(encoding="utf-8").replace("\r\n", "\n")
ADAPTER_SHA = "8886613e6ba0cb2f13d5144e723dd79445b727f1b4b00f0ec766cf821e3f4299"   # data_extra/g25c_run1/eval_arm/adapters/g3_lora/adapter_model.safetensors
BUDGET_MIN, CHUNK_STOP_MIN, CHUNK = 400, 362, 50

G25B_SUMMARY = json.loads((ROOT / "data_extra" / "g25b_run1" / "g25b" / "g25b_summary.json").read_text(encoding="utf-8"))
SOLVED_G25B = sorted({i for k, v in G25B_SUMMARY.items() if isinstance(v, dict) and k[:3] in ("r1_", "r2_") for i in v.get("resolved_ids", [])})
assert len(SOLVED_G25B) == G25B_SUMMARY["distinct_tasks_solved_g25b"] == 123, len(SOLVED_G25B)


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:80]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
SETUP, TRAIN, PACKAGE, EVAL = 3, 7, 9, 13
NOT_G25C = "'gemma-4-g25c' not in str(p)"   # the G25c kernel output is mounted too: never pick its copies of the arm / data

s = txt[SETUP]
s = sub(s, "G3 = WORK / 'g25c'", "G3 = WORK / 'g26a'")
s = sub(s, "'g25c_summary.json'", "'g26a_summary.json'")
s = sub(s, "BUDGET_MIN = 475", f"BUDGET_MIN = {BUDGET_MIN}")
s = sub(s, "DATA_DIR = next(p.parent for p in Path('/kaggle/input').rglob('tasks.jsonl') if (p.parent / 'sample_submission').is_dir() and (p.parent / 'snapshots').is_dir())",
        f"DATA_DIR = next(p.parent for p in Path('/kaggle/input').rglob('tasks.jsonl') if (p.parent / 'sample_submission').is_dir() and (p.parent / 'snapshots').is_dir() and {NOT_G25C})")
s = sub(s, "_sft = sorted(Path('/kaggle/input').rglob('sft_sequences.jsonl'))", f"_sft = sorted(p for p in Path('/kaggle/input').rglob('sft_sequences.jsonl') if {NOT_G25C})")
s = sub(s, "if 'swe_scaffold_v5_e4b_base' in p.read_text(encoding='utf-8') and 'adapter:' not in p.read_text(encoding='utf-8')]",
        f"if 'swe_scaffold_v5_e4b_base' in p.read_text(encoding='utf-8') and 'adapter:' not in p.read_text(encoding='utf-8') and {NOT_G25C}]")
s = sub(s, "bf = sorted(p.parent for p in Path('/kaggle/input').rglob('config.json') if 'unquantized' in str(p).lower())",
        f"bf = sorted(p.parent for p in Path('/kaggle/input').rglob('config.json') if 'unquantized' in str(p).lower() and {NOT_G25C})")
# FreshSWE-Py pool + per-task environments (the G25b block, verbatim where possible)
b = "".join(G25B_NB["cells"][3]["source"])
blk = b[b.index("_tj = [p for p in Path('/kaggle/input').rglob('tasks.jsonl') if (p.parent / 'locks').is_dir()"):b.index("TASKS_PATH = DATA_DIR / 'tasks.jsonl'\n")]
blk = blk.replace("POOL = _tj[0].parent\nDATA_DIR = POOL\n", "FRESH = _tj[0].parent; FRESH_TASKS = FRESH / 'tasks.jsonl'\n")
blk = blk.replace("assert (POOL / 'snapshots').is_dir(), POOL\n", "assert (FRESH / 'snapshots').is_dir(), FRESH\n")
assert "POOL" not in blk and "DATA_DIR" not in blk and "fresh_env.install_hook" in blk, blk[:400]
stub = (
    "FRESH_IDS = json.loads((_g25a / 'rollout_pool.json').read_text())\n"
    "fresh_by_id = {t.instance_id: t for t in load_tasks(FRESH_TASKS)}; assert not [i for i in FRESH_IDS if i not in fresh_by_id]\n"
    "import numpy as _np\n"
    "for _t in fresh_by_id.values():\n"
    "    _short = _t.repo.split('/')[-1]\n"
    "    (WORKING_DIR / 'no_graphs' / f'{_short}_{_t.base_commit}.json').write_text(json.dumps({'directed': True, 'multigraph': False, 'graph': {}, 'nodes': [], 'links': [], 'note': 'stub graph for rollouts, no code graph exists for this task'}))\n"
    "    _np.savez(WORKING_DIR / 'no_embeddings' / f'{_short}_{_t.base_commit}.npz', ids=_np.array(['stub'] * 8), vectors=_np.zeros((8, 256), dtype='float32'))\n"
    f"SOLVED_G25B = set({json.dumps(SOLVED_G25B)})\n"
    "assert SOLVED_G25B <= set(FRESH_IDS)\n"
    "print('FreshSWE:', len(FRESH_IDS), 'usable tasks |', len(SOLVED_G25B), 'solved by the base in G25b | stub graphs for', len(fresh_by_id))\n")
s = sub(s, "SMOKE_IDS = random.Random(5).sample(", blk + "(WORKING_DIR / 'no_graphs').mkdir(exist_ok=True); (WORKING_DIR / 'no_embeddings').mkdir(exist_ok=True)\n" + stub
        + "SMOKE_IDS = random.Random(5).sample(")
s = sub(s, f"(WORKING_DIR / 'fresh_env.py').write_text({FRESH_SRC!r}, encoding='utf-8')", "(WORKING_DIR / 'fresh_env.py').write_text(FRESH_SRC, encoding='utf-8')")   # G25b embedded the same fresh_env.py
s = "FRESH_SRC = " + json.dumps(FRESH_SRC) + "\n" + s
txt[SETUP] = s

txt[TRAIN - 1] = "## Load the G25c adapter arm (no training)"
txt[TRAIN] = (
    "import hashlib\n"
    "t0 = time.time()\n"
    "_ea = sorted(p.parent.parent.parent for p in Path('/kaggle/input').rglob('adapter_model.safetensors') if 'gemma-4-g25c' in str(p) and p.parent.parent.parent.name == 'eval_arm')\n"
    "assert len(_ea) == 1, _ea\n"
    "EVAL_ARM = WORK / 'eval_arm'; shutil.rmtree(EVAL_ARM, ignore_errors=True); shutil.copytree(_ea[0], EVAL_ARM)\n"
    "_sha = hashlib.sha256((EVAL_ARM / 'adapters' / 'g3_lora' / 'adapter_model.safetensors').read_bytes()).hexdigest()\n"
    f"assert _sha == '{ADAPTER_SHA}', ('not the G25c adapter', _sha)\n"
    "for _f in ('prompts/system.md', 'prompts/analyzer.md', 'configs/sampling.yaml', 'eval_config.yaml'):\n"
    "    assert (EVAL_ARM / _f).read_bytes() == (B1 / _f).read_bytes(), f'eval arm differs from the deployed E4B arm in {_f}'\n"
    "R['train'] = {'source': 'G25c adapter (kernel noobhunterjay/gemma-4-g25c-e4b-rft-train v1, eval_arm)', 'path': str(_ea[0]), 'sha256': _sha}\n"
    "R['train_wall_s'] = round(time.time() - t0); save_results(); print(R['train'])\n")
txt[PACKAGE - 1] = "## Wiring check of the loaded adapter arm"
txt[PACKAGE] = "R['wiring_real'] = W.check_wiring(EVAL_ARM); print('WIRING OK with the G25c adapter:', R['wiring_real']); save_results()\n"

s = txt[EVAL]
s = sub(s, "def run_round(tag, ids):\n    proxy.round_tag = tag; rd = G3 / tag\n    cfg = EvalConfig(tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / 'snapshots', results_dir=rd,",
        "def run_round(tag, ids, pool=False):\n    proxy.round_tag = tag; rd = G3 / tag\n"
        "    _ds = dict(tasks_path=FRESH_TASKS, snapshots_dir=FRESH / 'snapshots', graph_dir=str(WORKING_DIR / 'no_graphs'), embeddings_dir=str(WORKING_DIR / 'no_embeddings'), wheels_dir=WORKING_DIR / 'no_wheels') if pool else \\\n"
        "          dict(tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / 'snapshots', graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'), wheels_dir=DATA_DIR / 'wheels')\n"
        "    cfg = EvalConfig(**_ds, results_dir=rd,")
s = sub(s, "        graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'), wheels_dir=DATA_DIR / 'wheels', verbose=False, **budget_of(EVAL_ARM))",
        "        verbose=False, **budget_of(EVAL_ARM))")
s = sub(s, "'resolved_on_trained_tasks': len(ok & set(TRAINED)), 'resolved_on_unseen_tasks': len(ok & set(UNSEEN)),",
        "'resolved_on_trained_tasks': len(ok & set(TRAINED)), 'resolved_new_vs_g25b': sorted(ok - SOLVED_G25B) if pool else None,")
a = s.index("# ---- evaluation round over all 75 tasks with the adapter")
s = s[:a] + (
    "# ---- G26a: adapter rollouts over the FreshSWE-Py tasks, G25b-unsolved first, chunks of 50, no new chunk after CHUNK_STOP_MIN\n"
    f"CHUNK_STOP_MIN = {CHUNK_STOP_MIN}\n"
    "_un = sorted(set(FRESH_IDS) - SOLVED_G25B); random.Random(261).shuffle(_un)\n"
    "_so = sorted(SOLVED_G25B); random.Random(262).shuffle(_so)\n"
    f"ORDER = _un + _so; CHUNKS = [ORDER[i:i + {CHUNK}] for i in range(0, len(ORDER), {CHUNK})]\n"
    "print('rollout order:', len(_un), 'G25b-unsolved then', len(_so), 'G25b-solved |', len(CHUNKS), 'chunks', flush=True)\n"
    "for k, ch in enumerate(CHUNKS):\n"
    "    elapsed_min = (time.time() - T_START) / 60\n"
    "    if elapsed_min > CHUNK_STOP_MIN:\n"
    "        print(f'not starting a1_{k}: elapsed {elapsed_min:.0f} min > {CHUNK_STOP_MIN}'); R['stopped_before_chunk'] = k; break\n"
    "    run_round(f'a1_{k}', ch, pool=True)\n"
    "_rs = [R[t] for t in R if t.startswith('a1_') and isinstance(R[t], dict) and 'n' in R[t]]\n"
    "_ok = set().union(*[set(r['resolved_ids']) for r in _rs]) if _rs else set()\n"
    "_done = set().union(*[set(c) for c in CHUNKS[:len(_rs)]]) if _rs else set()\n"
    "R['a1_total'] = {'n': sum(r['n'] for r in _rs), 'resolved': len(_ok), 'patch_written': sum(r['patch_written'] for r in _rs), 'time_cap_hit': sum(r['time_cap_hit'] for r in _rs),\n"
    "                 'chunks': len(_rs), 'minutes': round((time.time() - T_START) / 60, 1),\n"
    "                 'on_g25b_unsolved': f\"{len(_ok - SOLVED_G25B)}/{len(_done - SOLVED_G25B)}\", 'on_g25b_solved': f\"{len(_ok & SOLVED_G25B)}/{len(_done & SOLVED_G25B)}\",\n"
    "                 'distinct_solved_union_with_g25b': len(_ok | SOLVED_G25B)}\n"
    "R['hook'] = dict(fresh_env.install_hook.stats); save_results()\n"
    "print('\\n===== G26a SUMMARY (G25c adapter rollouts on FreshSWE-Py) =====')\n"
    "print('adapter effect:', R.get('adapter_effect'), '| smoke:', {k: v for k, v in R.get('smoke', {}).items() if k not in ('resolved_ids', 'proxy')})\n"
    "print('a1:', R['a1_total'], '| hook', R['hook'])\n"
    "print('REFERENCE: base E4B (G25b r1) solved 119/617 = 19.3%; 123 distinct over r1 + 2 chunks of r2')\n"
    "R['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); R['capture'] = proxy.stats(); save_results(); _wd.cancel()\n")
txt[EVAL] = s

txt[0] = ("# G26a: self-RFT iteration 2, step 1 - rollouts of the G25c adapter (E4B + LoRA) on the FreshSWE-Py tasks\n\n"
          "Load the G25c adapter arm -> wiring check -> vLLM + LoRA + capture proxy -> adapter-effect + smoke gates -> chunks of 50 FreshSWE tasks (G25b-unsolved first).")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")
assert "make_adapters.main" not in "".join(txt.values()) and "train_g3.py')], env" not in "".join(txt.values())
(OUT / "g26a.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g25c" / "kernel-metadata.json").read_text(encoding="utf-8"))
meta.update({"id": "noobhunterjay/gemma-4-g26a-e4b-adapter-rollouts", "title": "Gemma 4 G26a E4B Adapter Rollouts", "code_file": "g26a.ipynb",
             "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g25-data", "moshegolderson/freshswe-py"],
             "kernel_sources": ["noobhunterjay/gemma-4-g25a-freshswe-env", "noobhunterjay/gemma-4-g25c-e4b-rft-train"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT, "| chunks of", CHUNK, "| watchdog", BUDGET_MIN, "min | G25b solved", len(SOLVED_G25B))
