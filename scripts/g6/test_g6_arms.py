"""Preflight for the G6 config screen (no GPU): build each arm from the B1 arm, compile it with the CURRENT wheelhouse sources (adk_submission 0.2.13 /
swegemma 0.2.11), then push one request through the compiled root agent's LiteLlm with a stub client and check that temperature, top_p, top_k, the output cap and
the thinking budget really arrive in the request that would go to vLLM. Run with PYTHONPATH pointing at the extracted new wheels first.

Usage: python scripts/g6/test_g6_arms.py <b1_arm_dir> <work_dir>
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g6_arms as G  # noqa: E402

from adk_submission import ModelRegistry, compile_submission, discover_declared_models  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.genai import types  # noqa: E402
from swegemma.config import build_submission_limits  # noqa: E402
from swegemma.context import SwegemmaContext  # noqa: E402
from swegemma.harness.agent_runner import resolve_swegemma_adapter  # noqa: E402
from swegemma.models import Task  # noqa: E402

b1, work = Path(sys.argv[1]), Path(sys.argv[2])
M31, M12 = "gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct"
task = Task(instance_id="d", repo="d/d", base_commit="0" * 40, problem_statement="d")
ctx = SwegemmaContext(docker_manager=None, container_id="", problem_statement="d", hints_text="", task=task, repo="d/d", graph_dir="/tmp", embeddings_dir="/tmp")
tools = ctx.create_tools()
limits, gen = build_submission_limits()


class StubClient:
    def __init__(self):
        self.calls = []

    async def acompletion(self, **kw):
        self.calls.append(kw)
        import litellm
        return litellm.ModelResponse(choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}])


def capture_request(arm: Path):
    declared = discover_declared_models(arm)
    reg = ModelRegistry()
    for m in declared:
        reg.register(m, LiteLlm(model="openai/" + m, api_base="http://127.0.0.1:8000/v1", api_key="EMPTY"))
    agent = compile_submission(submission_dir=arm, tool_registry=tools, model_registry=reg, limits=limits, generation_constraints=gen,
                               adapter_resolver_fn=lambda b, i: resolve_swegemma_adapter(b, i, reg))
    model = agent.model
    stub = StubClient()
    model.llm_client = stub
    req = LlmRequest(model=model.model, config=agent.generate_content_config.model_copy(deep=True),
                     contents=[types.Content(role="user", parts=[types.Part(text="hi")])])

    async def go():
        async for _ in model.generate_content_async(req, stream=False):
            pass
    asyncio.run(go())
    assert stub.calls, "the stub client was never called"
    return stub.calls[-1]


ok = True
# control: the B1 arm itself must show the CURRENT settings (4096 / 8192 / 0.2), proving the check can detect a difference
ctl = capture_request(b1)
cur = dict(temperature=0.2, top_p=0.95, top_k=None, max_output_tokens=8192, thinking_budget=4096)
body = dict(ctl); body.update(ctl.get("extra_body") or {})
print("control request fields:", {k: body.get(k) for k in ("temperature", "top_p", "top_k", "max_completion_tokens", "max_tokens", "thinking_token_budget")}, "| chat_template_kwargs", body.get("chat_template_kwargs"))
mm = G.request_matches(body, cur)
print("  control vs current B1 settings:", "OK" if not mm else mm)
ok &= not mm
for name, kw in G.ARMS.items():
    for model_name in (M31, M12):
        arm = G.make_arm(b1, work / f"arm_{name}_{model_name.split('-')[2]}", **kw)
        for rel in ("agent.yaml", "sub_agents/code_analyzer.yaml"):
            p = arm / rel
            p.write_text(p.read_text().replace(M31, model_name), encoding="utf-8", newline="\n")
        req = capture_request(arm)
        body = dict(req); body.update(req.get("extra_body") or {})
        mm = G.request_matches(body, kw)
        print(f"[{name} / {model_name}] fields:", {k: body.get(k) for k in ("temperature", "top_p", "top_k", "max_completion_tokens", "thinking_token_budget")}, "->", "OK" if not mm else mm)
        ok &= not mm
print("\nPREFLIGHT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
