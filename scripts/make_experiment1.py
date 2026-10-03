"""Generate Experiment 1 (scaffold-efficiency A/B) artifacts:

  kaggle_assets/exp1_arms_dataset/<arm>/...   four complete submission dirs derived from submission/
  kaggle_assets/exp1_arms/exp1_arms.ipynb     notebook: one vLLM boot, all arms on the same dev tasks
  kaggle_assets/exp1_arms/kernel-metadata.json

Arms (single-factor vs. control, plus the combination):
  A0_control      current submission/ (5 min cap, temp 0.2, thinking 4096)
  A1_edit_early   prompt: hard schedule, first real edit by tool call 8, partial fix beats none
  A2_think_off    sampling: thinking_budget 0 + include_thoughts false (96% of wall time is model latency)
  A3_edit_think   A1 + A2

See docs/STRATEGY.md "Experiment 1" for hypotheses, metrics and the decision rule.
"""

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "submission"
ARMS_DIR = ROOT / "kaggle_assets" / "exp1_arms_dataset"
NB_DIR = ROOT / "kaggle_assets" / "exp1_arms"

OLD_CORE = (
    "## Core Objective: Fast, Minimal, and Precise Fixes\n"
    "Aim to understand, resolve, and submit the fix in the minimum number of tool calls (under 8-10 turns). "
    "Move directly from the problem statement to the relevant files, apply the solution, verify with a targeted test, and submit.\n"
)
NEW_CORE = (
    "## Core Objective: Fast, Minimal, and Precise Fixes\n"
    "You have a HARD wall-clock limit of about 5 minutes for the whole task (roughly 15-25 tool calls). "
    "When it expires, whatever is in the working tree is graded as-is, so a plausible partial fix beats no patch, "
    "and never revert a plausible fix just because you are unsure. Follow this schedule:\n"
    "1. Locate (at most 4 tool calls): grep for exact identifiers or error text from the issue, then read the one function the issue is about.\n"
    "2. **Make your FIRST EDIT with `edit_file` by your 8th tool call**, using your best hypothesis, even if you are not fully certain. "
    "Do not keep exploring once you have found the function the issue is about.\n"
    "3. Verify with one targeted test (output redirected to a file, then `tail` it) and refine only if it fails.\n"
    "4. Call `submit_patch` as soon as the targeted test passes, or as soon as time is running low.\n"
)

# thinking_budget: 0 is rejected by this harness version's schema (must be >= 1) despite what the README says;
# the community's working configs disable thinking with include_thoughts: false alone (it wins over any budget).
THINK_OFF = "thinking_config:\n  include_thoughts: false\n"


def build_arm(name: str, agent_name: str, edit_early: bool, think_off: bool) -> Path:
    d = ARMS_DIR / name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(SRC, d)
    agent = (d / "agent.yaml").read_text(encoding="utf-8")
    assert agent.startswith("name: swe_scaffold_v1\n")
    (d / "agent.yaml").write_text(agent.replace("name: swe_scaffold_v1\n", f"name: {agent_name}\n", 1), encoding="utf-8")
    if edit_early:
        sysmd = (d / "prompts" / "system.md").read_text(encoding="utf-8")
        assert OLD_CORE in sysmd, "Core Objective block not found in system.md"
        (d / "prompts" / "system.md").write_text(sysmd.replace(OLD_CORE, NEW_CORE, 1), encoding="utf-8")
    if think_off:
        samp = (d / "configs" / "sampling.yaml").read_text(encoding="utf-8")
        old = "thinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n"
        assert old in samp, "thinking_config block not found"
        (d / "configs" / "sampling.yaml").write_text(samp.replace(old, THINK_OFF, 1), encoding="utf-8")
    # Python's write_text emits CRLF on Windows while copied files keep their original bytes; normalize every
    # file to LF so arms differ ONLY by the intended change (stray \r in a prompt would be a confound).
    for f in d.rglob("*"):
        if f.is_file():
            b = f.read_bytes()
            if b"\r\n" in b:
                f.write_bytes(b.replace(b"\r\n", b"\n"))
    return d


def pick_dev_tasks() -> list[str]:
    res = json.loads((ROOT / "submission_drafts" / "gold_patch_check" / "gold_patch_results.json").read_text(encoding="utf-8"))
    ids = sorted(r["instance_id"] for r in res if r.get("resolved"))
    fastapi = [i for i in ids if i.startswith("fastapi_")]
    requests_ = [i for i in ids if i.startswith("requests_")]
    rich = [i for i in ids if i.startswith("rich_")]
    rich_pick = sorted(random.Random(7).sample(rich, 17))  # fixed seed, no cherry-picking
    return fastapi + requests_ + rich_pick


INSTALL = '''import glob, importlib, os, shutil, subprocess, sys
from pathlib import Path
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'] = '1'
os.environ['VLLM_ENGINE_READY_TIMEOUT_S'] = '1200'
os.environ['VLLM_NO_USAGE_STATS'] = '1'
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
for pat in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth', '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for pth in glob.glob(pat):
        try: os.unlink(pth)
        except OSError: pass
tmp = Path('/tmp/wheelhouse'); tmp.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower(): continue
    name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    t = tmp / name
    if not t.exists(): os.symlink(w, t)
wheels = sorted(str(w) for w in tmp.glob('*.whl'))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()
print('installed', len(wheels), 'wheels')'''

SETUP = '''import json, random
from swegemma.models import load_tasks

DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WORKING_DIR = Path('/kaggle/working'); WORKING_DIR.mkdir(parents=True, exist_ok=True)

# Locate the arms dataset by content (agent.yaml name prefix), not by an assumed mount path.
arm_src = sorted({p.parent for p in Path('/kaggle/input').rglob('agent.yaml') if 'swe_exp_' in p.read_text(encoding='utf-8')})
assert len(arm_src) == 4, f'expected 4 arms, found {len(arm_src)}: {arm_src}'
ARMS = {}
for p in arm_src:
    dst = WORKING_DIR / 'arms' / p.name
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(p, dst)
    for must in ('agent.yaml', 'prompts/system.md', 'prompts/analyzer.md', 'configs/sampling.yaml', 'sub_agents/code_analyzer.yaml', 'eval_config.yaml'):
        assert (dst / must).is_file(), f'arm {p.name} missing {must}'
    ARMS[p.name] = dst
print('arms:', list(ARMS))
for n, d in ARMS.items():
    print(n, '| sampling:', (d / 'configs/sampling.yaml').read_text().replace(chr(10), ' ')[-110:])

DEV_IDS = __DEV_IDS__
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
by_id = {t.instance_id: t for t in load_tasks(TASKS_PATH)}
missing = [i for i in DEV_IDS if i not in by_id]
assert not missing, f'dev ids missing from dataset: {missing}'
print(f'{len(DEV_IDS)} dev tasks:', {r: sum(1 for i in DEV_IDS if i.startswith(r)) for r in ('fastapi', 'requests', 'rich')})'''

VLLM = '''import litellm, torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
from swegemma.models.discovery import validate_single_declared_model
litellm.drop_params = True
TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
first_arm = ARMS[sorted(ARMS)[0]]
declared_model = validate_single_declared_model(first_arm)
adapters = discover_adapters(str(first_arm), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
vllm_cfg = VllmConfig(model=str(MODEL_PATH), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
    max_model_len=32768, dtype='bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'auto',
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=True, max_loras=8, max_lora_rank=128,
    tensor_parallel_size=tp_size, startup_timeout=60 * 20)
server_instance = VllmServer(vllm_cfg, adapter_manifest=adapters)
server_instance.start()
print(f'vLLM server started on {server_instance.base_url} (tp={tp_size})')
models = server_instance.create_model_registry(aliases=[declared_model, TARGET_MODEL_NAME], model_prefix='openai/', api_key='EMPTY')'''

RUN = '''import asyncio, concurrent.futures, time, traceback, yaml
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator

def run_sync(coro_fn):
    try: loop = asyncio.get_running_loop()
    except RuntimeError: loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro_fn())).result()
    return asyncio.run(coro_fn())

def budget_of(arm_dir):
    cfg = yaml.safe_load((arm_dir / 'eval_config.yaml').read_text(encoding='utf-8')).get('evaluation', {})
    out = {}
    for k, cast in (('timeout_seconds', int), ('max_tool_calls', int), ('max_time_minutes', float)):
        if k in cfg: out[k] = cast(cfg[k])
    t = cfg.get('max_turns', cfg.get('max_llm_calls'))
    if t is not None: out['max_turns'] = int(t)
    return out

limits, gen_constraints = build_submission_limits()
EXP_DIR = WORKING_DIR / 'exp1'; EXP_DIR.mkdir(exist_ok=True)
order = sorted(ARMS); random.Random(11).shuffle(order)   # fixed shuffle: arm order is not systematically confounded with time
print('arm order this session:', order)
summary = {}
for arm in order:
    results_dir = EXP_DIR / arm
    cfg = EvalConfig(tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / 'snapshots', results_dir=results_dir,
        submission_dir=ARMS[arm], models=models, sandbox='subprocess', task_ids=DEV_IDS, concurrency=4, display_mode='quiet',
        limits=limits, generation_constraints=gen_constraints, adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
        graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'), wheels_dir=DATA_DIR / 'wheels',
        verbose=False, **budget_of(ARMS[arm]))
    print(f'\\n===== arm {arm} | budget {budget_of(ARMS[arm])} =====')
    t0 = time.perf_counter()
    try:
        res = run_sync(Evaluator(cfg).run)
    except Exception:
        traceback.print_exc(); summary[arm] = {'error': 'run raised'}; continue
    dur = time.perf_counter() - t0
    rows = [json.loads(l) for l in open(results_dir / 'task_results.jsonl', encoding='utf-8')]
    n = len(rows)
    summary[arm] = {
        'n': n, 'resolved': sum(r['resolved'] for r in rows),
        'patch_written': sum(1 for r in rows if r['agent_patch_size'] > 0),
        'time_cap_hit': sum(1 for r in rows if 'session timeout' in (r['error'] or '')),
        'mean_task_s': round(sum(r['duration_seconds'] for r in rows) / n, 1),
        'wall_s': round(dur, 1),
        'resolved_ids': sorted(r['instance_id'] for r in rows if r['resolved']),
    }
    print(json.dumps({k: v for k, v in summary[arm].items() if k != 'resolved_ids'}))
    (EXP_DIR / 'summary_all.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')   # incremental: survives a dead session

print('\\n===== SUMMARY (same %d dev tasks per arm) =====' % len(DEV_IDS))
print(f"{'arm':16s} {'resolved':>8s} {'patch':>6s} {'capHit':>7s} {'mean_s':>7s}")
for arm in sorted(summary):
    s = summary[arm]
    if 'error' in s: print(arm, s); continue
    print(f"{arm:16s} {s['resolved']:>8d} {s['patch_written']:>6d} {s['time_cap_hit']:>7d} {s['mean_task_s']:>7.1f}")'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


def main() -> None:
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    arms = [
        build_arm("A0_control", "swe_exp_a0_control", False, False),
        build_arm("A1_edit_early", "swe_exp_a1_edit_early", True, False),
        build_arm("A2_think_off", "swe_exp_a2_think_off", False, True),
        build_arm("A3_edit_think", "swe_exp_a3_edit_think", True, True),
    ]
    (ARMS_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "gemma4-exp1-arms", "id": "noobhunterjay/gemma4-exp1-arms",
        "licenses": [{"name": "CC0-1.0"}]}, indent=2), encoding="utf-8")
    dev = pick_dev_tasks()
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [
        md("# Experiment 1: scaffold-efficiency A/B on the real gemma-4-31b-it-qat-w4a16-ct\n\n"
           "Four arms (control; edit-early prompt; thinking off; both) on the SAME 30 dev tasks, one vLLM boot, "
           "temperature 0.2, 5-minute cap (the submission's settings). Hypotheses, metrics and the decision rule are in docs/STRATEGY.md ('Experiment 1'). "
           "Motivation: in 104 prior attempts 63 hit the time cap (46 with NO patch), 96% of wall time was model latency, and the median first real edit came at tool call 10."),
        code(INSTALL), md("## Arms and dev tasks"), code(SETUP.replace("__DEV_IDS__", json.dumps(dev))),
        md("## vLLM (real 31B)"), code(VLLM), md("## Run all arms (same tasks, shuffled arm order)"), code(RUN)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}
    (NB_DIR / "exp1_arms.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (NB_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": "noobhunterjay/gemma-4-exp1-arms", "title": "Gemma 4 Exp1 Arms",
        "code_file": "exp1_arms.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-exp1-arms"],
        "kernel_sources": [], "competition_sources": ["gemma-4-developer-agent"],
        "model_sources": ["google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaL4"}, indent=2), encoding="utf-8")
    print("arms:", [a.name for a in arms])
    print(f"{len(dev)} dev tasks:", dev)


if __name__ == "__main__":
    main()
