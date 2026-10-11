"""Build the adapter evaluation arm and verify that a LoRA adapter really flows through the harness's compile path, BEFORE any GPU time is spent on training.

make_eval_arm: copy the B1 arm, add `adapter: <name>` right after the `model:` line of agent.yaml and sub_agents/code_analyzer.yaml, drop adapters/<name>/ in.
check_wiring:  compile the arm exactly like swegemma 0.2.11 does (ModelRegistry with the base model and the adapter alias, discover_adapters manifest,
               resolve_swegemma_adapter) and assert that the root agent's model resolves to the adapter alias (LiteLlm 'openai/<name>').
"""

import re
import shutil
from pathlib import Path

ADAPTER = "g3_lora"
BASE = "gemma-4-31b-it-qat-w4a16-ct"


def make_eval_arm(b1_dir: Path, adapter_src: Path, dst: Path, name: str = ADAPTER) -> Path:
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(b1_dir, dst)
    for rel in ("agent.yaml", "sub_agents/code_analyzer.yaml"):
        p = dst / rel
        s = p.read_text(encoding="utf-8")
        pat = re.compile(rf"^(model: {re.escape(BASE)}\s*\n)", re.M)
        assert len(pat.findall(s)) == 1, f"{rel}: expected exactly one `model: {BASE}` line"
        assert "adapter:" not in s, f"{rel} already declares an adapter"
        p.write_text(pat.sub(r"\1" + f"adapter: {name}\n", s, count=1), encoding="utf-8", newline="\n")
    (dst / "adapters" / name).mkdir(parents=True)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(adapter_src / f, dst / "adapters" / name / f)
    return dst


def check_wiring(arm: Path, name: str = ADAPTER) -> dict:
    from adk_submission import ModelRegistry, compile_submission, discover_adapters, discover_declared_models
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, build_submission_limits
    from swegemma.context import SwegemmaContext
    from swegemma.harness.agent_runner import resolve_swegemma_adapter
    from swegemma.models import Task

    task = Task(instance_id="d", repo="d/d", base_commit="0" * 40, problem_statement="d")
    ctx = SwegemmaContext(docker_manager=None, container_id="", problem_statement="d", hints_text="", task=task, repo="d/d", graph_dir="/tmp", embeddings_dir="/tmp")
    tools = ctx.create_tools()
    declared = discover_declared_models(arm)
    assert declared == {BASE}, declared
    reg = ModelRegistry()
    for m in declared:   # base model and adapter alias, as VllmServer.create_model_registry registers them when the server has an adapter manifest
        reg.register(m, LiteLlm(model="openai/" + m, api_base="http://127.0.0.1:8000/v1", api_key="EMPTY"))
    manifest = discover_adapters(str(arm), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    assert set(manifest.adapters) == {name}, manifest.adapters
    reg.register(name, LiteLlm(model="openai/" + name, api_base="http://127.0.0.1:8000/v1", api_key="EMPTY"))
    limits, gen = build_submission_limits()
    agent = compile_submission(submission_dir=arm, tool_registry=tools, model_registry=reg, limits=limits, generation_constraints=gen, adapter_manifest=manifest,
                               adapter_resolver_fn=lambda b, i: resolve_swegemma_adapter(b, i, reg))
    model = getattr(agent, "model", None)
    model_id = getattr(model, "model", str(model))
    assert model_id == f"openai/{name}", f"root agent resolves to {model_id!r}, expected 'openai/{name}'"
    return {"agent": agent.name, "root_model": model_id, "adapters": sorted(manifest.adapters)}
