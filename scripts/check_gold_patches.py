"""Verify a dataset's reference (gold) patches actually resolve under the local sandbox
harness. Community reports claim gold patches fail offline for many fastapi/requests
tasks (missing test deps?) -- this script quantifies that directly instead of trusting
a task pool blindly for SFT/RL data or agent-scaffold validation. Works against both
the official 129-task competition pool and external pools (e.g. SWE-Pallets-X) that
follow the same tasks.jsonl/snapshots/graphs/embeddings layout.

Usage:
    python scripts/check_gold_patches.py                                    # official 129, concurrency=1
    python scripts/check_gold_patches.py fastapi_15661                      # one task
    python scripts/check_gold_patches.py --concurrency 6                    # official 129, 6 in parallel
    python scripts/check_gold_patches.py --data-dir data_extra/swe-pallets-x --concurrency 6
    python scripts/check_gold_patches.py --data-dir data_extra/swe-pallets-x flask_de8429f
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

# swegemma's resolve_sandbox_setup_script() looks for the competition's own
# sandbox/setup.py relative to the dataset root; external datasets (e.g.
# SWE-Pallets-X) don't ship their own copy, so point it at the real one from
# the official competition dataset explicitly.
os.environ.setdefault(
    "KAGGLE_SANDBOX_DIR", str(Path(__file__).resolve().parent.parent / "data" / "sandbox")
)

from adk_eval_core.tracing import SessionTrace
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import EvalConfig, Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks, setup_gemma_model_registry

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_subdir(data_dir: Path, name: str) -> Path:
    """Some external datasets (e.g. SWE-Pallets-X) double-nest their tar.gz extractions
    (snapshots/snapshots/*.tgz instead of snapshots/*.tgz) -- handle both layouts."""
    nested = data_dir / name / name
    flat = data_dir / name
    return nested if nested.is_dir() else flat


async def check_one(
    evaluator: Evaluator, config: EvalConfig, task, sem: asyncio.Semaphore, results_dir: Path
) -> dict:
    async with sem:
        snapshot_path, base_snapshot_path, patch_path = resolve_task_snapshot_paths(
            config.snapshots_dir, task.instance_id, task.repo
        )
        start = time.perf_counter()
        try:
            res = await verify_task(
                evaluator.docker,
                config,
                task,
                snapshot_path,
                base_snapshot_path=base_snapshot_path,
                patch_path=patch_path,
                agent_patch=task.patch,
                agent_error=None,
                trace=SessionTrace(),
                start_time=start,
            )
            resolved, error = res.resolved, res.error
            test_output, test_exit_code = res.test_output, res.test_exit_code
        except Exception as e:  # noqa: BLE001 - want to keep going across the whole pool
            resolved, error = False, f"{type(e).__name__}: {e}"
            test_output, test_exit_code = "", -1

        if test_output:
            log_path = results_dir / "test_outputs" / f"{task.instance_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(test_output, encoding="utf-8")

        result = {
            "instance_id": task.instance_id,
            "repo": task.repo,
            "resolved": resolved,
            "error": error,
            "test_exit_code": test_exit_code,
        }
        status = "PASS" if resolved else "FAIL"
        print(f"{task.instance_id} ({task.repo}): {status}" + (f" -- {error}" if error else ""))
        return result


async def main(data_dir: Path, task_ids: list[str] | None, concurrency: int) -> None:
    results_dir = PROJECT_ROOT / "submission_drafts" / "gold_patch_check" / data_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)
    config = EvalConfig(
        tasks_path=data_dir / "tasks.jsonl",
        snapshots_dir=_resolve_subdir(data_dir, "snapshots"),
        results_dir=results_dir,
        submission_dir=PROJECT_ROOT / "data" / "sample_submission",
        # resolve_wheels_dir()/resolve_sandbox_setup_script() both default to paths relative
        # to tasks_path/snapshots_dir; external datasets don't ship their own copy of either,
        # so point explicitly at the real competition's wheels cache (KAGGLE_SANDBOX_DIR above
        # handles the setup.py half of this).
        wheels_dir=PROJECT_ROOT / "data" / "wheels",
        models=setup_gemma_model_registry(),
        image="swebench-sandbox:latest",
        sandbox="docker",
        graph_dir=str(_resolve_subdir(data_dir, "graphs")),
        embeddings_dir=str(_resolve_subdir(data_dir, "embeddings")),
    )
    evaluator = Evaluator(config)
    tasks = load_tasks(config.tasks_path)
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.instance_id in wanted]

    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*(check_one(evaluator, config, t, sem, results_dir) for t in tasks))
    results = list(results)

    n_resolved = sum(r["resolved"] for r in results)
    print(f"\n{n_resolved}/{len(results)} gold patches resolved locally")

    by_repo: dict[str, list[bool]] = {}
    for r in results:
        by_repo.setdefault(r["repo"], []).append(r["resolved"])
    for repo, flags in sorted(by_repo.items()):
        print(f"  {repo}: {sum(flags)}/{len(flags)}")

    out_path = results_dir / "gold_patch_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_ids", nargs="*", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Dataset root containing tasks.jsonl, snapshots/, graphs/, embeddings/ (default: data/, the official 129-task pool)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.data_dir, args.task_ids or None, args.concurrency))
