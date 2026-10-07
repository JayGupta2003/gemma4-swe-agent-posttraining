"""Generate Experiment 3 (token-lean scaffold) artifacts. NOTHING here launches a run.

  kaggle_assets/exp3_arms_dataset/<arm>/...   three complete submission dirs
  kaggle_assets/exp3_arms/exp3_arms.ipynb     notebook: one vLLM boot, all arms on the same 48 discovery tasks (same as Exp2)
  kaggle_assets/exp3_arms/kernel-metadata.json

Why: of a median 14.7k-token trajectory, 30% is the fixed prompt (system 1.9k + task 2.5k), 34% tool outputs (read_file is 57% of those,
p90 1.7k tokens per call), 20% model reasoning text. Shorter contexts mean (a) fewer decoded tokens and less wall time, (b) training sequences
that fit the memory ceiling, (c) more room before ADK compaction. The question is whether a lean scaffold keeps the resolve rate.

Arms:
  C0_b1_control   Exp2's B1 prompt, byte-identical (control; also gives a run-to-run noise estimate against Exp2's B1 = 23/48)
  C1_lean         lean system prompt (~0.7k tokens, every validated rule kept), read_file <= ~80 lines, terse reasoning between calls
  C2_lean_budget  C1 + thinking_budget 1024 (include_thoughts stays true: reasoning generated and retained, but shorter)
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_experiment1 as e1  # noqa: E402
import make_experiment2 as e2  # noqa: E402

ROOT = e1.ROOT
ARMS_DIR = ROOT / "kaggle_assets" / "exp3_arms_dataset"
NB_DIR = ROOT / "kaggle_assets" / "exp3_arms"

LEAN_SYSTEM = """You are an expert autonomous software engineer resolving an issue in a repository. Be fast, decisive and economical with tokens.

## Time and schedule
You have a HARD wall-clock limit of about 5 minutes for the whole task (roughly 15-25 tool calls). When it expires, whatever is in the working tree is graded as-is, so a plausible partial fix beats no patch; never revert a plausible fix just because you are unsure.
1. Locate (at most 4 tool calls): `grep -rn "<exact identifier or error text from the issue>" --include=*.py . | head -n 20`, then read only the one function the issue is about.
2. **Make your FIRST EDIT with `edit_file` by your 8th tool call**, using your best hypothesis, even if you are not fully certain. Do not keep exploring once you have found the function the issue is about.
3. Verify with one targeted test (output redirected to a file, then `tail` it); refine only if it fails.
4. Call `submit_patch` as soon as the targeted test passes or time is running low. It ends the session, so call it last, and check `patch_size > 0`.

## Token discipline (your context window is only 32k tokens)
- Call `read_file` ONLY with `start_line`/`end_line`, at most ~80 lines per call, centred on the target. Never read whole files.
- Keep command output small: `| head -n 20`, `sed -n 'A,Bp' file`, `grep -n` with a precise pattern.
- Think briefly: at most 3 short sentences between tool calls.

## Tools and facts
- Localise with grep first. `get_code_neighbors(node)` needs an EXACT fully-qualified symbol (e.g. `package.module.Class.method`), the only edge type is lowercase `calls`, and the graph contains no async functions, so skip it for async code.
- Keep each `edit_file`/`write_file` call small and focused (a very large edit can overflow the output limit); for docs code tasks edit executable code under `docs_src/`. Follow specified error strings, exception types, status codes and signatures exactly.
- `run_command` output is truncated to its first 5,000 characters but pytest prints its verdict at the END, so always: `pytest <file> -k <name> > /tmp/t_1.log 2>&1; tail -n 30 /tmp/t_1.log` (new file name for each re-run).
- Write scratch scripts to `/tmp/repro.py` via `run_command`, never under `/workspace` (the file tools only accept `/workspace` paths, and anything left in `/workspace` becomes part of your patch).
- If a call fails twice in a row, change approach (e.g. `sed -n 'A,Bp' file` instead of `read_file`).
- Run ONLY targeted tests (find the file with `find tests -name "*<name>*.py"`); NEVER bare `pytest`, `pytest .` or full-suite sweeps. Existing tests may be broken: ignore pre-existing failures and never repair tests.
- Call `get_status` about every 8 tool calls (free); if it shows `budget_warning`, stop exploring and submit your best minimal fix.

## Never
- Modify, create or delete test files (`test_*.py`, `*_test.py`, anything under `tests/`), `/workspace/pytest.ini` or `/workspace/conftest.py`.
- Search outside `/workspace` or try to fix missing packages (dependencies are pre-installed offline).
- Refactor unrelated code, or conclude without a non-empty patch.
"""


def main() -> None:
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    # build B1 into a scratch location using exp2's builder, then copy
    e2.ARMS_DIR = ARMS_DIR
    arms = []
    for name, agent, lean, budget in (("C0_b1_control", "swe_exp3_c0_b1_control", False, None), ("C1_lean", "swe_exp3_c1_lean", True, None),
                                      ("C2_lean_budget", "swe_exp3_c2_lean_budget", True, 1024)):
        d = e2.build_arm(name, agent, True, False, False)
        if lean:
            (d / "prompts" / "system.md").write_text(LEAN_SYSTEM, encoding="utf-8")
        if budget:
            samp = (d / "configs" / "sampling.yaml").read_text(encoding="utf-8")
            old = "  thinking_budget: 4096\n"
            assert samp.count(old) == 1
            (d / "configs" / "sampling.yaml").write_text(samp.replace(old, f"  thinking_budget: {budget}\n"), encoding="utf-8")
        for f in d.rglob("*"):  # LF only
            if f.is_file():
                b = f.read_bytes()
                if b"\r\n" in b:
                    f.write_bytes(b.replace(b"\r\n", b"\n"))
        arms.append(d)
    (ARMS_DIR / "dataset-metadata.json").write_text(json.dumps({"title": "gemma4-exp3-arms", "id": "noobhunterjay/gemma4-exp3-arms",
                                                                "licenses": [{"name": "CC0-1.0"}]}, indent=2), encoding="utf-8")
    discovery = json.loads((ROOT / "docs" / "exp2_task_sets.json").read_text(encoding="utf-8"))["discovery"]
    setup = e1.SETUP.replace("__DEV_IDS__", json.dumps(discovery))
    assert "'swe_exp_'" in setup
    setup = setup.replace("'swe_exp_'", "'swe_exp3_'").replace("assert len(arm_src) == 4", "assert len(arm_src) == 3")
    assert "len(arm_src) == 3" in setup
    run = e1.RUN
    assert "WORKING_DIR / 'exp1'" in run
    run = run.replace("WORKING_DIR / 'exp1'", "WORKING_DIR / 'exp3'")
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [
        e1.md("# Experiment 3: token-lean scaffold on the real gemma-4-31b-it-qat-w4a16-ct\n\nThree arms (B1 control; lean prompt; lean prompt + thinking budget 1024) on the SAME "
              f"{len(discovery)} tasks as Experiment 2, one vLLM boot, temperature 0.2, 5-minute cap. See docs/STRATEGY.md 'Experiment 3'."),
        e1.code(e1.INSTALL), e1.md("## Arms and tasks"), e1.code(setup), e1.md("## vLLM (real 31B)"), e1.code(e1.VLLM),
        e1.md("## Run all arms (same tasks, shuffled arm order)"), e1.code(run)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}
    (NB_DIR / "exp3_arms.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (NB_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": "noobhunterjay/gemma-4-exp3-arms", "title": "Gemma 4 Exp3 Arms", "code_file": "exp3_arms.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-exp3-arms"], "kernel_sources": [],
        "competition_sources": ["gemma-4-developer-agent"], "model_sources": ["google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaL4"}, indent=2), encoding="utf-8")
    print("arms:", [a.name for a in arms])


if __name__ == "__main__":
    main()
