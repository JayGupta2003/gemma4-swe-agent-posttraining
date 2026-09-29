"""Isolate the read_file/edit_file 404 bug seen during real agent runs: every
copy_from() call for fastapi/sse.py failed with a Docker Engine API 404, for the
entire session, even with a freshly created container. This bypasses the LLM
agent loop entirely and drives the harness's own container setup + copy_from
directly, to find exactly which step breaks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.harness.container_setup import (
    extract_snapshot,
    install_editable_package,
    install_test_dependencies,
    setup_baseline_commit,
    setup_container_wheels,
    setup_git_exclude,
    setup_workspace_test_config,
)
from swegemma.evaluate import EvalConfig
from swegemma.models import load_tasks, setup_gemma_model_registry
from swegemma.sandbox import ContainerConfig, ContainerManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"


async def main() -> None:
    tasks = {t.instance_id: t for t in load_tasks(DATA / "tasks.jsonl")}
    task = tasks["fastapi_15588"]
    snapshot_path, base_snapshot_path, patch_path = resolve_task_snapshot_paths(
        DATA / "snapshots", task.instance_id, task.repo
    )

    manager = ContainerManager(ContainerConfig(image="swebench-sandbox:latest", timeout_seconds=120))
    container_id = manager.start()
    print("container:", container_id)

    try:
        print("extracting snapshot...")
        extract_snapshot(manager, container_id, snapshot_path, base_snapshot_path=base_snapshot_path, patch_path=patch_path)
        setup_git_exclude(manager, container_id)
        eval_config = EvalConfig(
            tasks_path=DATA / "tasks.jsonl",
            snapshots_dir=DATA / "snapshots",
            results_dir=PROJECT_ROOT / "submission_drafts" / "debug_copy_from",
            submission_dir=PROJECT_ROOT / "submission",
            models=setup_gemma_model_registry(),
            image="swebench-sandbox:latest",
            sandbox="docker",
        )
        setup_container_wheels(manager, container_id, eval_config)
        install_editable_package(manager, container_id)
        install_test_dependencies(manager, container_id, task.repo, config=eval_config)
        setup_workspace_test_config(manager, container_id, task.repo)
        setup_baseline_commit(manager, container_id)

        print("\n--- ls via run_command/exec ---")
        res = manager.exec(container_id, "ls -la /workspace/fastapi/sse.py")
        print("exit_code:", res.exit_code, "stdout:", res.stdout, "stderr:", res.stderr)

        print("\n--- cat via exec (plain docker exec, no get_archive) ---")
        res2 = manager.exec(container_id, "cat /workspace/fastapi/sse.py | head -5")
        print("exit_code:", res2.exit_code, "stdout:", res2.stdout[:200], "stderr:", res2.stderr)

        print("\n--- readlink -f (what copy_from does first) ---")
        res3 = manager.exec(container_id, "readlink -f /workspace/fastapi/sse.py")
        print("exit_code:", res3.exit_code, "stdout:", repr(res3.stdout), "stderr:", res3.stderr)

        print("\n--- copy_from with realistic LLM-latency-like idle gaps (15s) between calls ---")
        import tempfile
        import time
        n_ok, n_fail = 0, 0
        n_calls = 12
        for i in range(n_calls):
            print(f"  [{i}] sleeping 15s to simulate LLM inference latency...")
            time.sleep(15)
            with tempfile.TemporaryDirectory() as tmpdir:
                local_file = Path(tmpdir) / "sse.py"
                try:
                    manager.copy_from(container_id, "/workspace/fastapi/sse.py", local_file)
                    ok = local_file.exists() and local_file.stat().st_size > 0
                    n_ok += 1 if ok else 0
                    n_fail += 0 if ok else 1
                    print(f"  [{i}] copy_from {'OK' if ok else 'returned but file missing/empty'}")
                except Exception as e:
                    n_fail += 1
                    print(f"  [{i}] copy_from FAILED: {type(e).__name__}: {e}")
        print(f"\nidle-gap test result: {n_ok} ok / {n_fail} failed out of {n_calls}")

    finally:
        manager.stop(container_id)


if __name__ == "__main__":
    asyncio.run(main())
