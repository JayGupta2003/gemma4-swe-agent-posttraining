"""Offline gate for experiment arms: every arm dir must compile with the harness's own compile_submission and
differ from the control only in the intended files. Run in the `gemma4agent` env:

    python scripts/check_arms_compile.py kaggle_assets/exp2_arms_dataset
"""

import sys
from pathlib import Path

from adk_submission import compile_submission
from swegemma.context import SwegemmaContext
from swegemma.models import Task, resolve_swegemma_adapter, setup_gemma_model_registry

root = Path(sys.argv[1])
models = setup_gemma_model_registry()
task = Task(instance_id="d", repo="d/d", base_commit="0" * 40, problem_statement="d")
ctx = SwegemmaContext(docker_manager=None, container_id="", problem_statement="d", hints_text="", task=task,
                      repo="d/d", graph_dir="/tmp", embeddings_dir="/tmp")
tools = ctx.create_tools()
bad = 0
for d in sorted(p for p in root.iterdir() if (p / "agent.yaml").is_file()):
    try:
        agent = compile_submission(submission_dir=d, tool_registry=tools, model_registry=models,
                                   adapter_resolver_fn=lambda b, i: resolve_swegemma_adapter(b, i, models))
        crlf = [str(f.relative_to(d)) for f in d.rglob("*") if f.is_file() and b"\r\n" in f.read_bytes()]
        print(f"OK   {d.name}: agent={agent.name} sub_agents={[a.name for a in agent.sub_agents]} CRLF files={crlf}")
        bad += bool(crlf)
    except Exception as e:  # noqa: BLE001
        bad += 1
        print(f"FAIL {d.name}: {type(e).__name__}: {str(e)[:300]}")
sys.exit(1 if bad else 0)
