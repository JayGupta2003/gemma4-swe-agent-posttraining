"""CPU-only (no GPU quota) compatibility check for the smaller Gemma 4 variants announced 2026-10-12 (12B, E4B, E2B w4a16-ct), against the CURRENT wheelhouse.

Real serving needs a GPU, so this notebook answers the questions that can be answered without one:
  1. which package versions the wheelhouse has now, and whether swegemma still restricts the allowed model names (would a 12B agent.yaml even compile?)
  2. each model's config + checkpoint tensor names (architecture, quantisation, vision/audio tensors) read from the safetensors headers only
  3. what vLLM's Gemma4 loader does with those tensor names (source of the load_weights/skip logic; registry entries)
  4. a best-effort dry run of vLLM's own weight loading on the 'meta' device, with and without the hf-overrides staff documented, plus the vision_embedder skip
     that users report is still missing. This tier may fail on a CPU box; that is reported, not hidden.

Usage: python scripts/make_serve_check_notebook.py   -> kaggle_assets/serve_check/ (push with `kaggle kernels push -p kaggle_assets/serve_check`)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_experiment1 as e1  # noqa: E402  (INSTALL cell)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle_assets" / "serve_check"
OUT.mkdir(parents=True, exist_ok=True)

FACTS = r'''import importlib.metadata as md, inspect, re
for p in ('vllm', 'swegemma', 'adk_submission', 'google-adk', 'transformers', 'compressed-tensors', 'torch'):
    try: print(f'{p:20s}', md.version(p))
    except Exception as e: print(f'{p:20s} (not found: {e})')
print('wheelhouse wheels:', sorted(w.name for w in WHEELHOUSE_DIR.glob('*.whl') if re.match(r'(vllm|swegemma|adk_submission|compressed|transformers)', w.name)))
import swegemma.config as cfg
src = inspect.getsource(cfg)
print('\nallowed-model references in swegemma.config:')
for ln in src.splitlines():
    if re.search(r'ALLOWED_MODEL|gemma-4-(12b|e4b|e2b|31b)', ln): print('  ', ln.strip()[:200])
print('ALLOWED_MODEL_NAMES =', getattr(cfg, 'ALLOWED_MODEL_NAMES', '(attribute not present)'))
from adk_submission import VllmConfig
print('\nVllmConfig fields:', [f for f in getattr(VllmConfig, '__dataclass_fields__', getattr(VllmConfig, 'model_fields', {}))])'''

MODELS = r'''import glob, json, struct
from pathlib import Path
roots = {'12b': 'gemma-4-12b-it-qat-w4a16-ct', 'e4b': 'gemma-4-e4b-it-qat-w4a16-ct', 'e2b': 'gemma-4-e2b-it-qat-w4a16-ct', '31b': 'gemma-4-31b-it-qat-w4a16-ct'}
MODEL_DIRS = {}
for k, name in roots.items():
    hits = sorted(glob.glob(f'/kaggle/input/models/google/gemma-4/*/{name}/*'))
    print(k, name, '->', hits)
    if hits: MODEL_DIRS[k] = Path(hits[-1])

def header(f):
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        return json.loads(fh.read(n))

TENSORS = {}
for k, d in MODEL_DIRS.items():
    print('\n=====', k, d)
    print('files:', sorted(p.name for p in d.iterdir()))
    c = json.loads((d / 'config.json').read_text())
    print('architectures', c.get('architectures'), '| model_type', c.get('model_type'))
    tc = c.get('text_config', c)
    print('text: hidden', tc.get('hidden_size'), 'layers', tc.get('num_hidden_layers'), 'heads', tc.get('num_attention_heads'), 'kv', tc.get('num_key_value_heads'), 'vocab', tc.get('vocab_size'),
          'sliding', tc.get('sliding_window'), 'head_dim', tc.get('head_dim'), 'global_head_dim', tc.get('global_head_dim'))
    print('top-level config keys:', sorted(c))
    q = c.get('quantization_config') or {}
    print('quantization:', {k2: (v if k2 != 'config_groups' else '...') for k2, v in q.items()} if q else None)
    names = {}
    for f in sorted(d.glob('*.safetensors')):
        for n, v in header(f).items():
            if n != '__metadata__': names[n] = (v['dtype'], v['shape'])
    TENSORS[k] = names
    pref = {}
    for n in names:
        parts = n.split('.')
        key = '.'.join(parts[:2] if parts[0] != 'model' else parts[:3])
        pref[key] = pref.get(key, 0) + 1
    print(f'{len(names)} tensors; by prefix:', dict(sorted(pref.items())))
    print('non-text examples:', [n for n in names if re.search(r'vision|audio|embedder|multi_modal', n)][:8])'''

COMPILE = r'''import shutil, re
from adk_submission import ModelRegistry, compile_submission, discover_declared_models
from google.adk.models.lite_llm import LiteLlm
from swegemma.context import SwegemmaContext
from swegemma.models import Task
from swegemma.harness.agent_runner import resolve_swegemma_adapter
from swegemma.config import build_submission_limits
arm_src = next(Path('/kaggle/input/datasets/noobhunterjay/gemma4-g2-arm').rglob('agent.yaml')).parent
print('B1 arm source:', arm_src)
task = Task(instance_id='d', repo='d/d', base_commit='0' * 40, problem_statement='d')
ctx = SwegemmaContext(docker_manager=None, container_id='', problem_statement='d', hints_text='', task=task, repo='d/d', graph_dir='/tmp', embeddings_dir='/tmp')
tools = ctx.create_tools()
limits, gen = build_submission_limits()
for k, name in roots.items():
    d = Path(f'/tmp/arm_{k}'); shutil.rmtree(d, ignore_errors=True); shutil.copytree(arm_src, d)
    for rel in ('agent.yaml', 'sub_agents/code_analyzer.yaml'):
        p = d / rel; p.write_text(p.read_text().replace('gemma-4-31b-it-qat-w4a16-ct', name))
    try:
        declared = discover_declared_models(d)
        reg = ModelRegistry()
        for m in declared: reg.register(m, LiteLlm(model='openai/' + m, api_base='http://127.0.0.1:8000/v1', api_key='EMPTY'))
        agent = compile_submission(submission_dir=d, tool_registry=tools, model_registry=reg, limits=limits, generation_constraints=gen,
                                   adapter_resolver_fn=lambda b, i: resolve_swegemma_adapter(b, i, reg))
        print(f'[{k}] compile OK: declared {declared} -> {getattr(agent, "name", agent)}')
    except Exception as e:
        print(f'[{k}] compile FAILED: {type(e).__name__}: {str(e)[:400]}')'''

SOURCE = r'''import vllm, os, re
from vllm.model_executor.models import registry
root = Path(vllm.__file__).parent / 'model_executor' / 'models'
print('gemma4 files:', sorted(p.name for p in root.glob('*gemma4*')) + sorted(p.name for p in root.glob('*gemma_4*')))
reg = getattr(registry, '_VLLM_MODELS', None) or getattr(registry, '_TEXT_GENERATION_MODELS', {})
for key in sorted(reg):
    if 'emma4' in key: print('registry:', key, '->', reg[key])
for f in sorted(root.glob('*gemma4*.py')):
    s = f.read_text()
    print('\n=====', f.name, len(s.splitlines()), 'lines')
    for m in re.finditer(r'^class (\w+)', s, re.M): print('  class', m.group(1))
    i = s.find('def load_weights')
    while i >= 0:
        j = s.find('\n    def ', i + 10); j = len(s) if j < 0 else j
        print('--- load_weights (first 3500 chars) ---'); print(s[i:i + 3500]); break
    for kw in ('vision_embedder', 'patch_dense', 'skip_substrs', 'ignore_unexpected'):
        print(f'  mentions of {kw}:', len(re.findall(kw, s)))'''

DRYRUN = r'''import signal, traceback, torch
class TO(Exception): pass
def _h(*a): raise TO()
signal.signal(signal.SIGALRM, _h)
def attempt(label, overrides, patch_skip):
    print(f'\n--- dry run [{label}] hf_overrides={overrides} patch_skip={patch_skip}')
    signal.alarm(600)
    try:
        from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
        import vllm.model_executor.models.gemma4 as g4
        mc = ModelConfig(model=str(MODEL_DIRS['12b']), hf_overrides=overrides or {}, dtype='bfloat16', max_model_len=32768, trust_remote_code=False)
        print('  ModelConfig OK; architectures:', mc.architectures, '| hf model_type:', getattr(mc.hf_config, 'model_type', None))
        vc = VllmConfig(model_config=mc)
        from vllm.model_executor.models.registry import ModelRegistry as VR
        cls, arch = VR.resolve_model_cls(mc.architectures, model_config=mc)
        print('  resolved class:', cls.__name__, 'arch', arch)
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        init_distributed_environment(world_size=1, rank=0, distributed_init_method='tcp://127.0.0.1:29555', local_rank=0, backend='gloo'); initialize_model_parallel(1, 1)
        with set_current_vllm_config(vc), torch.device('meta'):
            model = cls(vllm_config=vc, prefix='')
        params = dict(model.named_parameters())
        print('  model built on meta:', len(params), 'params; first:', list(params)[:3])
        ck = TENSORS['12b']
        weights = ((n, torch.empty(shape, dtype=torch.bfloat16, device='meta')) for n, (dt, shape) in ck.items())
        loaded = model.load_weights(weights)
        print('  load_weights OK; loaded', len(loaded), 'params')
    except TO: print('  TIMEOUT')
    except BaseException as e:
        print(f'  FAILED: {type(e).__name__}: {str(e)[:300]}')
        print(''.join(traceback.format_exc().splitlines(True)[-6:]))
    finally: signal.alarm(0)
OV = {'architectures': ['Gemma4ForCausalLM'], 'model_type': 'gemma4'}
attempt('no overrides', None, False)
attempt('staff-documented overrides', OV, False)'''


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


cells = [code(e1.INSTALL), code(FACTS), code(MODELS), code(COMPILE), code(SOURCE), code(DRYRUN)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "serve_check.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-serve-check", "title": "Gemma 4 Serve Check", "code_file": "serve_check.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-g2-arm"], "kernel_sources": [],
    "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": ["google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2", "google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2",
                      "google/gemma-4/Other/gemma-4-e2b-it-qat-w4a16-ct/2", "google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT)
