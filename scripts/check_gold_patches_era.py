"""Era-aware gold-patch verification for third-party task pools (SWE-Pallets-X etc).

Builds a per-task wheels directory (see era_wheels.py) that never shadows a task's
own editable-installed repo, and iteratively discovers + era-pins whatever external
dependencies that specific task's tests actually import -- rather than relying on
one shared, version-blind wheels cache.

Usage:
    python scripts/check_gold_patches_era.py --data-dir data_extra/swe-pallets-x
    python scripts/check_gold_patches_era.py --data-dir data_extra/swe-pallets-x flask_de8429f
    python scripts/check_gold_patches_era.py --data-dir data_extra/swe-pallets-x --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

# swegemma caches unpacked wheels host-side keyed by a FIXED name ("sp_base",
# "sp_whl_<pkg>") regardless of which wheels_dir built them -- since we deliberately
# use a different per-task wheels_dir every time, this cache must be cleared before
# each attempt or later tasks silently reuse an earlier task's stale wheel set.
_SWEGEMMA_SP_CACHE_DIR = Path(tempfile.gettempdir()) / "swegemma_sp_cache_v8"


def _clear_swegemma_sp_cache() -> None:
    shutil.rmtree(_SWEGEMMA_SP_CACHE_DIR, ignore_errors=True)

from adk_eval_core.tracing import SessionTrace
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import EvalConfig, Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks, setup_gemma_model_registry

from era_wheels import (
    SELF_TESTABLE_PACKAGES,
    ensure_pytest_ini_addopts_preserved,
    ensure_setuptools_scm_stub,
    prepare_task_wheels_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("KAGGLE_SANDBOX_DIR", str(PROJECT_ROOT / "data" / "sandbox"))

SAFE_BASE_DIR = PROJECT_ROOT / "data_extra" / "era_wheel_cache" / "_safe_base"
MAX_ROUNDS = 8

_MISSING_MODULE_RE = re.compile(r"No module named '([A-Za-z0-9_\.]+)'")
_MODULE_TO_PKG = {
    "ephemeral_port_reserve": "ephemeral-port-reserve",
}


def _build_safe_base_dir(real_wheels_dir: Path) -> Path:
    """Real data/wheels/ minus anything matching a self-testable repo's package name."""
    SAFE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    if (SAFE_BASE_DIR / ".ready").exists():
        return SAFE_BASE_DIR
    for whl in real_wheels_dir.glob("*.whl"):
        pkg_name = re.sub(r"[-_.]+", "-", whl.name.split("-", 1)[0]).lower()
        if pkg_name in SELF_TESTABLE_PACKAGES:
            continue
        dest = SAFE_BASE_DIR / whl.name
        if not dest.exists():
            dest.write_bytes(whl.read_bytes())
    (SAFE_BASE_DIR / ".ready").write_text("ok", encoding="utf-8")
    return SAFE_BASE_DIR


async def _verify_once(task, base_config: EvalConfig, wheels_dir: Path) -> tuple:
    config = EvalConfig(
        tasks_path=base_config.tasks_path,
        snapshots_dir=base_config.snapshots_dir,
        results_dir=base_config.results_dir,
        submission_dir=base_config.submission_dir,
        models=base_config.models,
        image=base_config.image,
        sandbox=base_config.sandbox,
        graph_dir=base_config.graph_dir,
        embeddings_dir=base_config.embeddings_dir,
        wheels_dir=wheels_dir,
    )
    evaluator = Evaluator(config)
    snapshot_path, base_snapshot_path, patch_path = resolve_task_snapshot_paths(
        config.snapshots_dir, task.instance_id, task.repo
    )
    snapshot_path = ensure_setuptools_scm_stub(task.instance_id, snapshot_path)
    snapshot_path = ensure_pytest_ini_addopts_preserved(task.instance_id, snapshot_path)
    if base_snapshot_path is not None:
        base_snapshot_path = ensure_setuptools_scm_stub(task.instance_id, base_snapshot_path)
        base_snapshot_path = ensure_pytest_ini_addopts_preserved(task.instance_id, base_snapshot_path)
    start = time.perf_counter()
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
    return res.resolved, res.error, res.test_output, res.test_exit_code


async def check_one_era_aware(
    task, base_config: EvalConfig, safe_base_dir: Path, work_dir: Path, sem: asyncio.Semaphore,
    repo_dep_cache: dict[str, list[str]],
) -> dict:
    async with sem:
        repo_short = task.repo.rsplit("/", 1)[-1].lower()
        known_extra: list[str] = list(repo_dep_cache.get(repo_short, []))
        resolved, error, test_output, test_exit_code = False, None, "", -1
        for round_idx in range(MAX_ROUNDS):
            wheels_dir = prepare_task_wheels_dir(task, known_extra, safe_base_dir, work_dir)
            _clear_swegemma_sp_cache()
            try:
                resolved, error, test_output, test_exit_code = await _verify_once(task, base_config, wheels_dir)
            except Exception as e:  # noqa: BLE001
                resolved, error, test_output, test_exit_code = False, f"{type(e).__name__}: {e}", "", -1

            if resolved:
                break

            missing = set()
            for m in _MISSING_MODULE_RE.findall(test_output or ""):
                top = m.split(".")[0]
                missing.add(_MODULE_TO_PKG.get(top, top))
            new_missing = [m for m in missing if m not in known_extra]
            if not new_missing:
                break  # not a missing-module problem (or nothing more we can auto-fix)
            known_extra.extend(new_missing)

        if resolved and known_extra:
            repo_dep_cache[repo_short] = known_extra

        if test_output:
            log_path = base_config.results_dir / "test_outputs" / f"{task.instance_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(test_output, encoding="utf-8")

        result = {
            "instance_id": task.instance_id,
            "repo": task.repo,
            "resolved": resolved,
            "error": error,
            "test_exit_code": test_exit_code,
            "extra_pkgs_pinned": known_extra,
        }
        status = "PASS" if resolved else "FAIL"
        extra_note = f" (pinned: {known_extra})" if known_extra else ""
        print(f"{task.instance_id} ({task.repo}): {status}{extra_note}" + (f" -- {error}" if error else ""))
        return result


async def main(data_dir: Path, task_ids: list[str] | None, concurrency: int) -> None:
    results_dir = PROJECT_ROOT / "submission_drafts" / "gold_patch_check" / f"{data_dir.name}_era"
    results_dir.mkdir(parents=True, exist_ok=True)
    work_dir = PROJECT_ROOT / "data_extra" / "era_wheel_cache"

    def _resolve_subdir(name: str) -> Path:
        nested = data_dir / name / name
        return nested if nested.is_dir() else data_dir / name

    real_wheels_dir = PROJECT_ROOT / "data" / "wheels"
    safe_base_dir = _build_safe_base_dir(real_wheels_dir)

    base_config = EvalConfig(
        tasks_path=data_dir / "tasks.jsonl",
        snapshots_dir=_resolve_subdir("snapshots"),
        results_dir=results_dir,
        submission_dir=PROJECT_ROOT / "data" / "sample_submission",
        models=setup_gemma_model_registry(),
        image="swebench-sandbox:latest",
        sandbox="docker",
        graph_dir=str(_resolve_subdir("graphs")),
        embeddings_dir=str(_resolve_subdir("embeddings")),
    )

    tasks = load_tasks(base_config.tasks_path)
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.instance_id in wanted]

    if concurrency != 1:
        print(
            "Note: forcing concurrency=1 -- swegemma's host-side wheel cache is keyed by a "
            "fixed name regardless of wheels_dir, so concurrent tasks with different per-task "
            "wheel needs would corrupt each other's cache.",
        )
    sem = asyncio.Semaphore(1)
    # Pre-seeded from manual investigation, so the first task of each repo doesn't have to
    # re-discover these one ModuleNotFoundError at a time; still auto-extends per-task if a
    # task needs something beyond this.
    repo_dep_cache: dict[str, list[str]] = {
        "flask": ["werkzeug", "markupsafe", "click", "jinja2", "itsdangerous"],
        "werkzeug": ["markupsafe"],
        "jinja2": ["markupsafe"],
    }
    results = list(await asyncio.gather(
        *(check_one_era_aware(t, base_config, safe_base_dir, work_dir, sem, repo_dep_cache) for t in tasks)
    ))

    n_resolved = sum(r["resolved"] for r in results)
    print(f"\n{n_resolved}/{len(results)} gold patches resolved locally (era-aware)")

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
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.data_dir, args.task_ids or None, args.concurrency))
