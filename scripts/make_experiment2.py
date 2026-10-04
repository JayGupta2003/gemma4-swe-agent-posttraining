"""Generate Experiment 2 (prompt-ladder A/B) artifacts. NOTHING here launches a run.

  kaggle_assets/exp2_arms_dataset/<arm>/...   four complete submission dirs derived from submission/
  kaggle_assets/exp2_arms/exp2_arms.ipynb     notebook: one vLLM boot, all arms on the same discovery tasks
  kaggle_assets/exp2_arms/kernel-metadata.json
  docs/exp2_task_sets.json                    discovery set (48 fresh tasks) + confirmation set (27 Exp1 tasks)

Arms form a LADDER: each rung adds exactly one idea to the previous one, so adjacent arms are single-factor.
  B0_control      current submission/ (5 min cap, temp 0.2, thinking 4096)
  B1_edit_early   Exp1's A1 prompt (replication of the best-looking Exp1 arm on FRESH tasks)
  B2_no_repro     B1 + do not write a reproduction script before the first edit
  B3_complete     B2 + one completeness grep for sibling occurrences before submit_patch

Why these: Exp1 traces showed stuck attempts spend 34% of calls on repro scripts (resolved attempts 21%) and
only 3% on edits (resolved 11%); near-miss analysis showed 26% of failures touch fewer files than the reference
fix (resolved: 4%). See docs/STRATEGY.md "Experiment 2".
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_experiment1 as e1  # noqa: E402  (reuse the notebook cells and the Exp1 prompt text)

ROOT = e1.ROOT
SRC = e1.SRC
ARMS_DIR = ROOT / "kaggle_assets" / "exp2_arms_dataset"
NB_DIR = ROOT / "kaggle_assets" / "exp2_arms"
TASK_SETS = ROOT / "docs" / "exp2_task_sets.json"

# --- B2: the repro-script bullet in system.md section 3 (keeps the /workspace-footgun warning) -------------
OLD_REPRO = (
    "- If you need a scratch/reproduction script to check your understanding of the bug, write it to `/tmp/repro.py` "
    "(via `run_command`), never under `/workspace` — the file tools (`read_file`/`write_file`/`edit_file`) only accept paths "
    "inside `/workspace` and will fail on `/tmp` paths. Anything left inside `/workspace` when you call `submit_patch` — "
    "including scratch scripts — becomes part of your submitted patch.\n"
)
NEW_REPRO = (
    "- **Do NOT write a reproduction script before your first edit.** Attempts that get stuck spend a third of their tool calls "
    "on repro scripts and almost none on editing. Read the issue and the function it names, make your best-hypothesis edit, "
    "and only then verify with the existing targeted test. If no existing test covers the behaviour, you may write a short "
    "scratch script AFTER your first edit, to `/tmp/repro.py` (via `run_command`), never under `/workspace` — the file tools "
    "(`read_file`/`write_file`/`edit_file`) only accept paths inside `/workspace` and will fail on `/tmp` paths. Anything left "
    "inside `/workspace` when you call `submit_patch` — including scratch scripts — becomes part of your submitted patch.\n"
)
OLD_STEP2_TAIL = "Do not keep exploring once you have found the function the issue is about.\n"
NEW_STEP2_TAIL = ("Do not keep exploring once you have found the function the issue is about, "
                  "and do not write a reproduction script first.\n")

# --- B3: completeness check (the near-miss finding: failed fixes touch fewer places than the reference fix) ----
OLD_STEP4 = "4. Call `submit_patch` as soon as the targeted test passes, or as soon as time is running low.\n"
NEW_STEP4 = (
    "4. Run the completeness check from section 4 (ONE grep for other places that need the same change), then call `submit_patch`. "
    "If time is running low, skip the check and submit immediately.\n"
)
OLD_S4_1 = "  1. Double check you have not left any scratch files"
NEW_S4_1 = (
    "  1. **Completeness check (ONE `grep -rn` tool call):** search the repo (source and `docs_src/`, excluding tests) for the function, "
    "class, argument or pattern you just changed, to find sibling occurrences: sync/async twins, other branches or classes that handle "
    "the same case, other callers that pass the old value, near-duplicate helpers. Reference fixes very often touch more than one place. "
    "If a sibling has the same bug, apply the same minimal fix there too; if not, move on without further exploration.\n"
    "  2. Double check you have not left any scratch files"
)
# renumber the rest of that list (reverse order so no replacement collides with another)
RENUMBER = [("  4. Output a short summary", "  5. Output a short summary"), ("  3. Verify `patch_size > 0`", "  4. Verify `patch_size > 0`"),
            ("  2. Call `submit_patch` immediately", "  3. Call `submit_patch` immediately")]


def rep(text: str, old: str, new: str, what: str) -> str:
    assert text.count(old) == 1, f"{what}: expected exactly one match, found {text.count(old)}"
    return text.replace(old, new, 1)


def build_arm(name: str, agent_name: str, edit_early: bool, no_repro: bool, complete: bool) -> Path:
    d = ARMS_DIR / name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(SRC, d)
    agent = (d / "agent.yaml").read_text(encoding="utf-8")
    agent = rep(agent, "name: swe_scaffold_v1\n", f"name: {agent_name}\n", "agent name")
    (d / "agent.yaml").write_text(agent, encoding="utf-8")
    sysmd = (d / "prompts" / "system.md").read_text(encoding="utf-8")
    if edit_early:
        core = e1.NEW_CORE
        if no_repro:
            core = rep(core, OLD_STEP2_TAIL, NEW_STEP2_TAIL, "step2 tail")
        if complete:
            core = rep(core, OLD_STEP4, NEW_STEP4, "step4")
        sysmd = rep(sysmd, e1.OLD_CORE, core, "core block")
    if no_repro:
        sysmd = rep(sysmd, OLD_REPRO, NEW_REPRO, "repro bullet")
    if complete:
        sysmd = rep(sysmd, OLD_S4_1, NEW_S4_1, "section 4 step 1")
        for old, new in RENUMBER:
            sysmd = rep(sysmd, old, new, f"renumber {old[:12]}")
    (d / "prompts" / "system.md").write_text(sysmd, encoding="utf-8")
    # Python's write_text emits CRLF on Windows; normalize everything to LF so arms differ ONLY by the intended change.
    for f in d.rglob("*"):
        if f.is_file():
            b = f.read_bytes()
            if b"\r\n" in b:
                f.write_bytes(b.replace(b"\r\n", b"\n"))
    return d


def task_sets() -> tuple[list[str], list[str]]:
    """discovery = every Kaggle-env-gold-passing task NOT in the Exp1 dev set (fresh, no sampling, no cherry-picking);
    confirmation = the Exp1 dev tasks whose gold passes in Kaggle's env (27; 3 requests tasks are unwinnable there)."""
    gold = json.loads((ROOT / "docs" / "kaggle_env_gold_results.json").read_text(encoding="utf-8"))
    ok = sorted(i for i, v in gold.items() if v["resolved"])
    exp1_dev = e1.pick_dev_tasks()  # same deterministic function that produced the Exp1 set (seed 7)
    discovery = [i for i in ok if i not in exp1_dev]
    confirmation = [i for i in exp1_dev if i in ok]
    assert not set(discovery) & set(confirmation)
    return discovery, confirmation


def main() -> None:
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    arms = [
        build_arm("B0_control", "swe_exp2_b0_control", False, False, False),
        build_arm("B1_edit_early", "swe_exp2_b1_edit_early", True, False, False),
        build_arm("B2_no_repro", "swe_exp2_b2_no_repro", True, True, False),
        build_arm("B3_complete", "swe_exp2_b3_complete", True, True, True),
    ]
    (ARMS_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "gemma4-exp2-arms", "id": "noobhunterjay/gemma4-exp2-arms",
        "licenses": [{"name": "CC0-1.0"}]}, indent=2), encoding="utf-8")

    discovery, confirmation = task_sets()
    TASK_SETS.write_text(json.dumps({"discovery": discovery, "confirmation": confirmation}, indent=1), encoding="utf-8")

    # Reuse Exp1's notebook cells with exactly three string substitutions; assert each really happened.
    setup = e1.SETUP.replace("__DEV_IDS__", json.dumps(discovery))
    assert "'swe_exp_'" in setup
    setup = setup.replace("'swe_exp_'", "'swe_exp2_'")
    run = e1.RUN
    assert "WORKING_DIR / 'exp1'" in run
    run = run.replace("WORKING_DIR / 'exp1'", "WORKING_DIR / 'exp2'")

    NB_DIR.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [
        e1.md("# Experiment 2: prompt ladder on the real gemma-4-31b-it-qat-w4a16-ct\n\n"
              f"Four arms (control; edit-early; +no repro script before first edit; +completeness grep) on the SAME {len(discovery)} fresh tasks "
              "(every Kaggle-env-gold-passing public task not used in Experiment 1), one vLLM boot, temperature 0.2, 5-minute cap. "
              "Hypotheses, metrics and the pre-registered decision rule are in docs/STRATEGY.md ('Experiment 2')."),
        e1.code(e1.INSTALL), e1.md("## Arms and discovery tasks"), e1.code(setup),
        e1.md("## vLLM (real 31B)"), e1.code(e1.VLLM), e1.md("## Run all arms (same tasks, shuffled arm order)"), e1.code(run)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}
    (NB_DIR / "exp2_arms.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (NB_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": "noobhunterjay/gemma-4-exp2-arms", "title": "Gemma 4 Exp2 Arms",
        "code_file": "exp2_arms.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-exp2-arms"],
        "kernel_sources": [], "competition_sources": ["gemma-4-developer-agent"],
        "model_sources": ["google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaL4"}, indent=2), encoding="utf-8")
    by = lambda ids: {r: sum(i.startswith(r) for i in ids) for r in ("fastapi", "rich", "requests")}
    print("arms:", [a.name for a in arms])
    print(f"discovery {len(discovery)}:", by(discovery))
    print(f"confirmation {len(confirmation)}:", by(confirmation))


if __name__ == "__main__":
    main()
