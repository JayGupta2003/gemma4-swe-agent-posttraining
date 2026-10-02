"""Generates kaggle_assets/zip_test/zip_test.ipynb (+ kernel-metadata.json).

Tests the actual submission.zip ARTIFACT on Linux: extract each zip the way the scorer
would, then run adk_submission.compile_submission on the result. Every earlier notebook
run used a *directory copy* of submission/, so the zip artifact itself was never tested.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle_assets" / "zip_test"
OUT.mkdir(parents=True, exist_ok=True)

INSTALL = '''import glob, importlib, os, subprocess, sys
from pathlib import Path
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['OTEL_SDK_DISABLED'] = 'true'
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

TEST = '''import base64, os, shutil, subprocess, traceback, zipfile
from pathlib import Path

DATA = next(p for p in Path('/kaggle/input').rglob('bad_backslash.zip.b64.txt')).parent
print('test data dir:', DATA)

from adk_submission import compile_submission
from swegemma.context import SwegemmaContext
from swegemma.models import Task, resolve_swegemma_adapter, setup_gemma_model_registry

models = setup_gemma_model_registry()
task = Task(instance_id='d', repo='d/d', base_commit='0'*40, problem_statement='d')
ctx = SwegemmaContext(docker_manager=None, container_id='', problem_statement='d', hints_text='', task=task,
                      repo='d/d', graph_dir='/tmp', embeddings_dir='/tmp')
tools = ctx.create_tools()

def try_compile(label, d):
    try:
        agent = compile_submission(submission_dir=Path(d), tool_registry=tools, model_registry=models,
                                   adapter_resolver_fn=lambda b, i: resolve_swegemma_adapter(b, i, models))
        print(f'  [{label}] COMPILE OK: {agent.name}')
        return True
    except Exception as e:
        print(f'  [{label}] COMPILE FAILED: {type(e).__name__}: {str(e)[:300]}')
        return False

results = {}
for variant in ('bad_backslash', 'good_posix'):
    print(f'\\n===== {variant} =====')
    zpath = Path(f'/tmp/{variant}.zip')
    zpath.write_bytes(base64.b64decode((DATA / f'{variant}.zip.b64.txt').read_text()))
    # A: python zipfile.extractall on Linux
    dA = Path(f'/tmp/extract_py_{variant}'); shutil.rmtree(dA, ignore_errors=True); dA.mkdir()
    with zipfile.ZipFile(zpath) as z:
        print('  stored names (Linux zipfile):', z.namelist())
        z.extractall(dA)
    print('  extracted tree (python):', sorted(str(p.relative_to(dA)) for p in dA.rglob('*') if p.is_file()))
    okA = try_compile('python-extract', dA)
    # B: unzip CLI on Linux
    dB = Path(f'/tmp/extract_unzip_{variant}'); shutil.rmtree(dB, ignore_errors=True); dB.mkdir()
    r = subprocess.run(['unzip', '-o', str(zpath), '-d', str(dB)], capture_output=True, text=True)
    print('  unzip rc:', r.returncode, (r.stderr or '')[:200].strip())
    print('  extracted tree (unzip):', sorted(str(p.relative_to(dB)) for p in dB.rglob('*') if p.is_file()))
    okB = try_compile('unzip-extract', dB)
    results[variant] = (okA, okB)

print('\\n===== SUMMARY =====')
for k, v in results.items():
    print(k, 'python-extract ok =', v[0], '| unzip-extract ok =', v[1])'''

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}

nb = {
    "cells": [
        md("# Zip artifact test (Linux)\n\nSubmissions 56717963, 56752702 and 56780281 were built with PowerShell `Compress-Archive` and all crashed with a generic 'unhandled error'; the one zip built with forward slashes (56682704) was the only one that ran. Hypothesis: Compress-Archive stores backslash entry names, which on Linux extract as flat files named like `configs\\\\sampling.yaml`, so `!include configs/sampling.yaml` cannot resolve. This notebook extracts a backslash zip and a POSIX zip on Linux and tries to compile each."),
        code(INSTALL),
        code(TEST),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(OUT / "zip_test.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": "noobhunterjay/gemma-4-zip-artifact-test",
    "title": "Gemma 4 Zip Artifact Test",
    "code_file": "zip_test.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "noobhunterjay/gemma4-zip-test"],
    "kernel_sources": [],
    "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
