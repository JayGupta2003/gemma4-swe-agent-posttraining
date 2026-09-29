"""Rollout collection: run the Workstream 1 agent scaffold against verified-resolvable
tasks and keep only trajectories whose resulting patch actually passes (rejection
sampling) -- raw material for SFT training data (Workstream 2/3).

This is deliberately built on the exact two-phase pipeline the real Kaggle scorer
uses (Phase 1: agent runs in a sandbox and produces a patch; Phase 2: that patch is
verified in a clean container with anti-tampering resets) rather than a bespoke
mini-harness, so a trajectory that "passes" here would also pass under the real
scorer. Concretely this calls the same `run_agent_sandbox` + `verify_task` functions
`swegemma.evaluate.Evaluator.evaluate_task` calls internally.

Only tasks already confirmed locally resolvable (see check_gold_patches.py /
check_gold_patches_era.py results) are attempted -- there's no point spending agent
budget on a task whose environment can't even run the KNOWN-CORRECT gold patch.

For the external SWE-Pallets-X pool this reuses the era-aware, self-exclusive wheel
infrastructure from era_wheels.py, using the extra_pkgs_pinned already discovered by
check_gold_patches_era.py (no rediscovery needed) so each task gets its own
correctly-pinned per-task wheels dir instead of one version-blind shared cache.

IMPORTANT caveat (see docs/STRATEGY.md): local runs via --models-yaml
scripts/lmstudio_models.yaml hit a much smaller Gemma-4 E4B variant through LM
Studio, not the real gemma-4-31b-it-qat-w4a16-ct checkpoint the competition
requires. That's fine for validating this pipeline end-to-end (does the agent
loop run, does it produce syntactically valid patches, does verification work),
but trajectories collected that way are NOT valid production SFT data -- they
are not "the 31B's own self-generated trajectories". Real training data collection
must run against the actual 31B checkpoint (Kaggle L4x4 notebook), by pointing
--models-yaml at nothing (falls back to setup_gemma_model_registry()'s real
endpoint) inside that environment.

Usage:
    # local smoke test via LM Studio (fast iteration, E4B stand-in -- pipeline
    # validation only, not real training data)
    python scripts/collect_rollouts.py --models-yaml scripts/lmstudio_models.yaml \
        --max-tasks 2 --samples-per-task 1

    # official pool, real 31B registry (run this inside the Kaggle notebook)
    python scripts/collect_rollouts.py --data-dir data --samples-per-task 4 \
        --submission-dir submission_drafts/rollout_submission

    # SWE-Pallets-X pool (era-aware wheels), one repo at a time
    python scripts/collect_rollouts.py --data-dir data_extra/swe-pallets-x \
        --repo click --samples-per-task 4 --submission-dir submission_drafts/rollout_submission
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("KAGGLE_SANDBOX_DIR", str(PROJECT_ROOT / "data" / "sandbox"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# swegemma caches unpacked wheels host-side keyed by a FIXED name regardless of which
# wheels_dir built them -- must be cleared before every attempt whose wheels_dir may
# differ from the previous one (see era_wheels.py / check_gold_patches_era.py).
_SWEGEMMA_SP_CACHE_DIR = Path(tempfile.gettempdir()) / "swegemma_sp_cache_v8"


def _clear_swegemma_sp_cache() -> None:
    shutil.rmtree(_SWEGEMMA_SP_CACHE_DIR, ignore_errors=True)


from swegemma.config import EvalConfig
from swegemma.context import SwegemmaContext
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import Evaluator
from swegemma.harness.agent_runner import run_agent_sandbox
from swegemma.harness.verification import save_trace_artifact, verify_task
from swegemma.models import Task, load_tasks, setup_gemma_model_registry

from era_wheels import ensure_pytest_ini_addopts_preserved, ensure_setuptools_scm_stub, prepare_task_wheels_dir
from check_gold_patches_era import _build_safe_base_dir

OFFICIAL_RESULTS = PROJECT_ROOT / "submission_drafts" / "gold_patch_check" / "gold_patch_results.json"


def _era_results_path(data_dir: Path) -> Path:
    return PROJECT_ROOT / "submission_drafts" / "gold_patch_check" / f"{data_dir.name}_era" / "gold_patch_results.json"


def _resolve_subdir(data_dir: Path, name: str) -> Path:
    nested = data_dir / name / name
    return nested if nested.is_dir() else data_dir / name


def load_resolvable_tasks(data_dir: Path) -> dict[str, list[str]]:
    """instance_id -> known extra_pkgs_pinned (empty list for the official pool),
    restricted to tasks confirmed `resolved: true` by the gold-patch verifiers."""
    is_official = data_dir.resolve() == (PROJECT_ROOT / "data").resolve()
    results_path = OFFICIAL_RESULTS if is_official else _era_results_path(data_dir)
    if not results_path.exists():
        raise SystemExit(
            f"No gold-patch verification results found at {results_path}. "
            "Run scripts/check_gold_patches.py (official pool) or "
            "scripts/check_gold_patches_era.py --data-dir <dir> (external pool) first --"
            " rollout collection only attempts tasks already confirmed locally resolvable."
        )
    records = json.loads(results_path.read_text(encoding="utf-8"))
    return {
        r["instance_id"]: r.get("extra_pkgs_pinned", [])
        for r in records
        if r.get("resolved")
    }


async def _run_one_attempt(
    config: EvalConfig,
    task: Task,
    snapshot_path: Path,
    base_snapshot_path: Path | None,
    patch_path: Path | None,
    task_index: int,
    total_tasks: int,
):
    """Phase 1 (agent produces a patch) + Phase 2 (clean-container verification),
    mirroring Evaluator.evaluate_task's body but with explicit (possibly era-patched)
    snapshot paths, since Evaluator.evaluate_task always re-resolves them itself from
    a single shared snapshots_dir."""
    evaluator = Evaluator(config)
    start = time.perf_counter()
    context = SwegemmaContext(
        docker_manager=evaluator.docker,
        container_id="",
        problem_statement=task.problem_statement,
        hints_text=getattr(task, "hints_text", "") or "",
        task=task,
        repo=task.repo,
        graph_dir=config.graph_dir,
        embeddings_dir=config.embeddings_dir,
        budget=config.budget,
        harness=config.harness,
    )
    agent_patch, agent_error, trace = await run_agent_sandbox(
        evaluator.docker,
        config,
        task,
        snapshot_path,
        base_snapshot_path=base_snapshot_path,
        patch_path=patch_path,
        task_index=task_index,
        total_tasks=total_tasks,
        context=context,
    )

    if agent_error and not agent_patch:
        json_path = save_trace_artifact(trace, config.results_dir, task.instance_id)
        return {
            "instance_id": task.instance_id,
            "repo": task.repo,
            "resolved": False,
            "agent_patch": "",
            "error": agent_error,
            "test_exit_code": -1,
            "duration_seconds": time.perf_counter() - start,
            "tool_calls": context.tool_calls_used,
            "total_llm_calls": context.llm_calls_used,
            "trace_path": json_path,
        }

    res = await verify_task(
        evaluator.docker,
        config,
        task,
        snapshot_path,
        base_snapshot_path=base_snapshot_path,
        patch_path=patch_path,
        agent_patch=agent_patch,
        agent_error=agent_error,
        trace=trace,
        start_time=start,
    )
    return {
        "instance_id": task.instance_id,
        "repo": task.repo,
        "resolved": res.resolved,
        "agent_patch": res.agent_patch,
        "error": res.error,
        "test_exit_code": res.test_exit_code,
        "duration_seconds": res.duration_seconds,
        "tool_calls": context.tool_calls_used,
        "total_llm_calls": context.llm_calls_used,
        "trace_path": res.trace_json_path,
    }


async def collect_for_task(
    task: Task,
    known_extra: list[str],
    *,
    is_official: bool,
    base_config_kwargs: dict,
    safe_base_dir: Path | None,
    era_work_dir: Path,
    samples_per_task: int,
    results_dir: Path,
    task_index: int,
    total_tasks: int,
    sem: asyncio.Semaphore,
) -> list[dict]:
    async with sem:
        real_snapshots_dir = base_config_kwargs["snapshots_dir"]
        snapshot_path, base_snapshot_path, patch_path = resolve_task_snapshot_paths(
            real_snapshots_dir, task.instance_id, task.repo
        )

        if is_official:
            wheels_dir = base_config_kwargs.get("wheels_dir")
        else:
            snapshot_path = ensure_setuptools_scm_stub(task.instance_id, snapshot_path)
            snapshot_path = ensure_pytest_ini_addopts_preserved(task.instance_id, snapshot_path)
            if base_snapshot_path is not None:
                base_snapshot_path = ensure_setuptools_scm_stub(task.instance_id, base_snapshot_path)
                base_snapshot_path = ensure_pytest_ini_addopts_preserved(task.instance_id, base_snapshot_path)
            wheels_dir = prepare_task_wheels_dir(task, known_extra, safe_base_dir, era_work_dir)

        records = []
        for sample_idx in range(samples_per_task):
            config = EvalConfig(wheels_dir=wheels_dir, **{k: v for k, v in base_config_kwargs.items() if k != "wheels_dir"})
            _clear_swegemma_sp_cache()
            try:
                record = await _run_one_attempt(
                    config, task, snapshot_path, base_snapshot_path, patch_path, task_index, total_tasks
                )
            except Exception as e:  # noqa: BLE001 - keep going across the whole pool
                record = {
                    "instance_id": task.instance_id,
                    "repo": task.repo,
                    "resolved": False,
                    "agent_patch": "",
                    "error": f"{type(e).__name__}: {e}",
                    "test_exit_code": -1,
                    "duration_seconds": 0.0,
                    "tool_calls": 0,
                    "total_llm_calls": 0,
                    "trace_path": None,
                }
            record["sample_idx"] = sample_idx
            records.append(record)

            status = "KEPT (resolved)" if record["resolved"] else "discarded (failed)"
            print(
                f"[{task_index}/{total_tasks}] {task.instance_id} sample {sample_idx}: {status}"
                + (f" -- {record['error']}" if record["error"] else "")
            )

            with open(results_dir / "rollouts.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

        return records


async def main(args: argparse.Namespace) -> None:
    data_dir: Path = args.data_dir
    is_official = data_dir.resolve() == (PROJECT_ROOT / "data").resolve()

    resolvable = load_resolvable_tasks(data_dir)
    if args.task_ids:
        wanted = set(args.task_ids)
        missing = wanted - resolvable.keys()
        if missing:
            print(f"WARNING: these task_ids are not in the confirmed-resolvable set, skipping: {sorted(missing)}")
        resolvable = {k: v for k, v in resolvable.items() if k in wanted}

    all_tasks = load_tasks(data_dir / "tasks.jsonl")
    tasks = [t for t in all_tasks if t.instance_id in resolvable]
    if args.repo:
        tasks = [t for t in tasks if args.repo.lower() in t.repo.lower()]
    tasks.sort(key=lambda t: (t.repo, t.instance_id))
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    if not tasks:
        raise SystemExit("No matching resolvable tasks found for the given filters.")

    run_name = args.run_name or f"{data_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    results_dir = PROJECT_ROOT / "submission_drafts" / "rollouts" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "traces").mkdir(exist_ok=True)

    safe_base_dir = None
    era_work_dir = PROJECT_ROOT / "data_extra" / "era_wheel_cache"
    if is_official:
        wheels_dir = PROJECT_ROOT / "data" / "wheels"
    else:
        safe_base_dir = _build_safe_base_dir(PROJECT_ROOT / "data" / "wheels")
        wheels_dir = None  # computed per-task

    base_config_kwargs = dict(
        tasks_path=data_dir / "tasks.jsonl",
        snapshots_dir=_resolve_subdir(data_dir, "snapshots"),
        results_dir=results_dir,
        submission_dir=args.submission_dir,
        models=setup_gemma_model_registry(models_yaml_path=args.models_yaml),
        image="swebench-sandbox:latest",
        sandbox="docker",
        graph_dir=str(_resolve_subdir(data_dir, "graphs")),
        embeddings_dir=str(_resolve_subdir(data_dir, "embeddings")),
        wheels_dir=wheels_dir,
        timeout_seconds=args.timeout_seconds,
        max_time_minutes=args.max_time_minutes,
        max_tool_calls=args.max_tool_calls,
        max_turns=args.max_turns,
    )

    print(
        f"Collecting rollouts: {len(tasks)} tasks x {args.samples_per_task} samples "
        f"= {len(tasks) * args.samples_per_task} attempts, submission_dir={args.submission_dir}, "
        f"models_yaml={args.models_yaml or '(real registry)'}\n"
        f"Results -> {results_dir}"
    )

    sem = asyncio.Semaphore(1)  # sp_cache is not safe for concurrent wheels_dir switches
    all_records: list[dict] = []
    for i, task in enumerate(tasks, start=1):
        recs = await collect_for_task(
            task,
            resolvable[task.instance_id],
            is_official=is_official,
            base_config_kwargs=base_config_kwargs,
            safe_base_dir=safe_base_dir,
            era_work_dir=era_work_dir,
            samples_per_task=args.samples_per_task,
            results_dir=results_dir,
            task_index=i,
            total_tasks=len(tasks),
            sem=sem,
        )
        all_records.extend(recs)

    n_kept = sum(r["resolved"] for r in all_records)
    print(f"\n{n_kept}/{len(all_records)} attempts resolved (kept as SFT-candidate trajectories)")
    by_repo: dict[str, list[bool]] = {}
    for r in all_records:
        by_repo.setdefault(r["repo"], []).append(r["resolved"])
    for repo, flags in sorted(by_repo.items()):
        print(f"  {repo}: {sum(flags)}/{len(flags)}")
    print(f"\nWrote {results_dir / 'rollouts.jsonl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--submission-dir", type=Path, default=PROJECT_ROOT / "submission")
    parser.add_argument("--models-yaml", type=Path, default=None)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--repo", type=str, default=None, help="Substring filter on task.repo")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-time-minutes", type=float, default=10)
    parser.add_argument("--max-tool-calls", type=int, default=40)
    parser.add_argument("--max-turns", type=int, default=40)
    args = parser.parse_args()
    asyncio.run(main(args))
