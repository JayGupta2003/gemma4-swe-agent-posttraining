"""Verify the competition's reference (gold) patches actually resolve under the local
sandbox harness. Community reports claim gold patches fail offline for many fastapi/
requests tasks (missing test deps?) -- this script quantifies that directly instead of
trusting the local 129-task pool blindly for SFT/RL data or agent-scaffold validation.

Usage:
    python scripts/check_gold_patches.py                       # all 129 tasks, concurrency=1
    python scripts/check_gold_patches.py fastapi_15661          # one task
    python scripts/check_gold_patches.py --concurrency 6        # all tasks, 6 in parallel
    python scripts/check_gold_patches.py --concurrency 6 fastapi_15661 rich_2725
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from adk_eval_core.tracing import SessionTrace
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import EvalConfig, Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks, setup_gemma_model_registry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "submission_drafts" / "gold_patch_check"


async def check_one(evaluator: Evaluator, config: EvalConfig, task, sem: asyncio.Semaphore) -> dict:
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
        except Exception as e:  # noqa: BLE001 - want to keep going across all 129 tasks
            resolved, error = False, f"{type(e).__name__}: {e}"

        result = {"instance_id": task.instance_id, "repo": task.repo, "resolved": resolved, "error": error}
        status = "PASS" if resolved else "FAIL"
        print(f"{task.instance_id} ({task.repo}): {status}" + (f" -- {error}" if error else ""))
        return result


async def main(task_ids: list[str] | None, concurrency: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = EvalConfig(
        tasks_path=DATA / "tasks.jsonl",
        snapshots_dir=DATA / "snapshots",
        results_dir=RESULTS_DIR,
        submission_dir=DATA / "sample_submission",
        models=setup_gemma_model_registry(),
        image="swebench-sandbox:latest",
        sandbox="docker",
    )
    evaluator = Evaluator(config)
    tasks = load_tasks(config.tasks_path)
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.instance_id in wanted]

    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*(check_one(evaluator, config, t, sem) for t in tasks))
    results = list(results)

    n_resolved = sum(r["resolved"] for r in results)
    print(f"\n{n_resolved}/{len(results)} gold patches resolved locally")

    by_repo: dict[str, list[bool]] = {}
    for r in results:
        by_repo.setdefault(r["repo"], []).append(r["resolved"])
    for repo, flags in sorted(by_repo.items()):
        print(f"  {repo}: {sum(flags)}/{len(flags)}")

    out_path = RESULTS_DIR / "gold_patch_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_ids", nargs="*", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(main(args.task_ids or None, args.concurrency))
