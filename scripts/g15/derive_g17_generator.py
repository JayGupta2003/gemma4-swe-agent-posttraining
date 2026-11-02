"""Generate kaggle_assets/g17/ : G17 = RPO (DPO + NLL on the chosen trajectory) LoRA training of the 31B on the budgeted preference pairs (scripts/g15/select_pairs.py: 28 pairs, 27 pool tasks,
0.6M tokens, from the three pool rounds p1-p3), 1 epoch, then the SAME 75-task competition evaluation as G3/G11 (B1 arm, 5-minute cap, adapter, no call cap) so the result is comparable to
the base rounds (25 and 28 resolved, 44 and 39 capped) and to G3 (28) / G11 (22).

Derived from the G3 generator (scripts/g3/make_g3_notebook.py: wiring check, staged checkpoint, supervised training, adapter export, vLLM with the adapter, smoke gate, evaluation) by exact-string
edits, each asserted. Cost model (measured SFT throughput 81 tok/s, ~3 forward-backward units per pair token incl. the cached reference pass): ~6.1 h training + ~0.4 h staging/boot/smoke + ~1.8 h
evaluation = ~8.3 h wall = ~17 quota h; the trainer stops itself after RPO_MAX_SECONDS (21,600 s = 6 h of training, reference pass not counted) and the whole notebook after 690 minutes (Kaggle's limit is 12 h).
Pre-registered decision rule (docs/PREFERENCE_RUN.md section 5): interesting only if resolved >= 34, or capped <= 35 with resolved >= 28, with patches written not lower than the base and calls
per task not higher; anything inside 22-30 is a null. Behaviour checks (E4B lesson): patches written, capped count and calls per task are read next to the resolved count.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = (ROOT / "scripts" / "g3" / "make_g3_notebook.py").read_text(encoding="utf-8")


def sub(t, old, new, n=1):
    assert t.count(old) == n, (old[:80], t.count(old))
    return t.replace(old, new)


g = src
g = sub(g, '"""Generate kaggle_assets/g3/ : G3 = LoRA SFT on the G2 sequences (1 epoch, all 30 solved tasks, no hold-out) + evaluation round with the adapter.',
        '"""Generate kaggle_assets/g17/ : G17 = RPO LoRA training of the 31B on 28 budgeted preference pairs (1 epoch) + the 75-task evaluation round with the adapter. Derived from the G3 generator.\n[G3 docstring follows] G3 = LoRA SFT on the G2 sequences (1 epoch, all 30 solved tasks, no hold-out) + evaluation round with the adapter.')
g = sub(g, "HERE = Path(__file__).resolve().parent\n", "HERE = Path(__file__).resolve().parent.parent / 'g3'   # the shared trainer/wiring sources live in scripts/g3\n")
g = sub(g, 'OUT = ROOT / "kaggle_assets" / "g3"', 'OUT = ROOT / "kaggle_assets" / "g17"')
g = sub(g, '"train_g3.py": HERE / "train_g3.py",', '"train_rpo.py": HERE.parent / "g15" / "train_rpo.py",')
# SETUP
g = sub(g, "G3 = WORK / 'g3'", "G3 = WORK / 'g17'")
g = sub(g, "'g3_summary.json'", "'g17_summary.json'")
g = sub(g, "BUDGET_MIN = 340", "BUDGET_MIN = 690")
g = sub(g, "DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent');",
        "DATA_DIR = next(p.parent for p in Path('/kaggle/input').rglob('tasks.jsonl') if (p.parent / 'sample_submission').is_dir() and (p.parent / 'snapshots').is_dir());")
g = sub(g, "SFT = next(Path('/kaggle/input').rglob('sft_sequences.jsonl'))", "_pp = sorted(Path('/kaggle/input').rglob('pairs.jsonl')); assert len(_pp) == 1, _pp; SFT = _pp[0]")
# TRAIN
g = sub(g, "env = dict(os.environ, G3_MODEL=str(STAGED), G3_DATA=str(SFT), G3_OUT=str(G3 / 'train'), G3_ACC='2', G3_LR='2e-4', G3_EPOCHS='1', G3_WARMUP='3',\n"
           "           G3_MAXMEM=json.dumps({'0': '16GiB', '1': '16GiB', '2': '16GiB', '3': '16GiB'}), G3_CKPT_EVERY='10', TOKENIZERS_PARALLELISM='false')",
        "env = dict(os.environ, RPO_MODEL=str(STAGED), RPO_DATA=str(SFT), RPO_OUT=str(G3 / 'train'), RPO_ACC='1', RPO_LR='1e-4', RPO_EPOCHS='1', RPO_WARMUP='2',\n"
        "           RPO_BETA='0.1', RPO_ALPHA='1.0', RPO_NORM='0', RPO_MAX_SECONDS='21600',\n"
        "           RPO_MAXMEM=json.dumps({'0': '16GiB', '1': '16GiB', '2': '16GiB', '3': '16GiB'}), RPO_CKPT_EVERY='8', TOKENIZERS_PARALLELISM='false')")
g = sub(g, "[sys.executable, str(CODE / 'train_g3.py')], env, G3 / 'train.log',\n                                  [('[g3] loaded in', 600), ('[g3] step 1/', 1800)], max_seconds=4.0 * 3600)",
        "[sys.executable, str(CODE / 'train_rpo.py')], env, G3 / 'train.log',\n                                  [('[rpo] loaded in', 900), ('[rpo] step 1/', 6000)], max_seconds=7.5 * 3600)")
g = sub(g, "('stage', 'error', 'optimizer_steps_planned', 'tokens_per_second', 'peak_gib_per_gpu', 'oom_events', 'leak_gib_at_end', 'train_seconds', 'trainable_params_m', 'data')}",
        "('stage', 'error', 'optimizer_steps_planned', 'tokens_per_second', 'peak_gib_per_gpu', 'oom_events', 'leak_gib_at_end', 'train_seconds', 'ref_seconds', 'stopped_early', 'trainable_params_m', 'data')}")
g = sub(g, "R['train']['losses'] = [s['loss'] for s in M.get('steps', [])]; R['train']['grad_norms'] = [s['grad_norm'] for s in M.get('steps', [])]",
        "R['train']['losses'] = [s['dpo_loss'] for s in M.get('steps', [])]; R['train']['grad_norms'] = [s['grad_norm'] for s in M.get('steps', [])]\n"
        "R['train']['margins'] = [s['margin'] for s in M.get('steps', [])]; R['train']['acc'] = [s['acc'] for s in M.get('steps', [])]\n"
        "R['train']['nll_chosen'] = [s['nll_chosen'] for s in M.get('steps', [])]; R['train']['d_chosen'] = [s['d_chosen'] for s in M.get('steps', [])]; R['train']['d_rejected'] = [s['d_rejected'] for s in M.get('steps', [])]")
g = sub(g, "if M.get('stage') != 'done' or len(M.get('steps', [])) < 0.8 * planned:", "if M.get('stage') != 'done' or (len(M.get('steps', [])) < 0.8 * planned and not M.get('stopped_early')) or len(M.get('steps', [])) < 0.4 * planned:")
g = sub(g, "print('losses:', R['train']['losses']); print('grad norms:', R['train']['grad_norms'])",
        "print('dpo losses:', R['train']['losses']); print('margins:', R['train']['margins']); print('chosen logp change:', R['train']['d_chosen']); print('rejected logp change:', R['train']['d_rejected']); print('grad norms:', R['train']['grad_norms'])")
g = sub(g, "('losses', 'grad_norms')", "('losses', 'grad_norms', 'margins', 'acc', 'nll_chosen', 'd_chosen', 'd_rejected')", 2)
g = sub(g, "G3 SUMMARY", "G17 SUMMARY")
# notebook files / metadata
g = sub(g, '(OUT / "g3.ipynb")', '(OUT / "g17.ipynb")')
g = sub(g, '"id": "noobhunterjay/gemma-4-g3-train", "title": "Gemma 4 G3 Train", "code_file": "g3.ipynb"', '"id": "noobhunterjay/gemma-4-g17-rpo-train", "title": "Gemma 4 G17 RPO Train", "code_file": "g17.ipynb"')
g = sub(g, '"noobhunterjay/gemma4-g3-sft"', '"noobhunterjay/gemma4-g17-pairs"')
g = sub(g, '"# G3: LoRA SFT on G2 trajectories (all 30 solved tasks, 1 epoch) + evaluation round with the adapter', '"# G17: RPO LoRA on 28 budgeted preference pairs (1 epoch) + 75-task evaluation round with the adapter')
(ROOT / "scripts" / "g15" / "make_g17_notebook.py").write_text(g, encoding="utf-8", newline="\n")
print("derived generator written (scripts/g15/make_g17_notebook.py)")
