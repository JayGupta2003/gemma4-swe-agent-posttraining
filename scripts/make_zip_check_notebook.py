"""CPU-only Linux check of an EXACT submission.zip artifact before a submission slot is spent.

Usage: python scripts/make_zip_check_notebook.py <zip_path> <tag>
Creates kaggle_assets/zip_check_<tag>/ (notebook + metadata) and kaggle_assets/zip_check_<tag>_dataset/ (the zip as base64 text, because Kaggle
auto-extracts uploaded .zip files, plus per-file SHA-256 of the expected contents). The notebook, on Linux: decodes the zip, checks its sha256, extracts
with Python's zipfile AND the unzip CLI, compares every extracted file's SHA-256 with the expected one, and runs adk_submission.compile_submission.
"""

import base64
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_experiment1 as e1  # noqa: E402  (INSTALL cell)

ROOT = Path(__file__).resolve().parent.parent
zip_path, tag = Path(sys.argv[1]).resolve(), sys.argv[2]
OUT, DS = ROOT / "kaggle_assets" / f"zip_check_{tag}", ROOT / "kaggle_assets" / f"zip_check_{tag}_dataset"
OUT.mkdir(parents=True, exist_ok=True)
DS.mkdir(parents=True, exist_ok=True)

raw = zip_path.read_bytes()
with zipfile.ZipFile(zip_path) as z:
    expected = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
(DS / f"{tag}.zip.b64.txt").write_text(base64.b64encode(raw).decode("ascii"), encoding="utf-8", newline="\n")
(DS / "expected.json").write_text(json.dumps({"zip_sha256": hashlib.sha256(raw).hexdigest(), "files": expected}, indent=1), encoding="utf-8", newline="\n")
(DS / "dataset-metadata.json").write_text(json.dumps({"title": f"gemma4-zipcheck-{tag}", "id": f"noobhunterjay/gemma4-zipcheck-{tag}",
                                                      "licenses": [{"name": "CC0-1.0"}]}, indent=2), encoding="utf-8", newline="\n")

TEST = r'''import base64, hashlib, json, shutil, subprocess, zipfile
from pathlib import Path
DATA = next(Path('/kaggle/input').rglob('expected.json')).parent
exp = json.loads((DATA / 'expected.json').read_text())
raw = base64.b64decode(next(DATA.glob('*.zip.b64.txt')).read_text())
zp = Path('/tmp/check.zip'); zp.write_bytes(raw)
print('zip sha256 matches local build:', hashlib.sha256(raw).hexdigest() == exp['zip_sha256'], '|', len(raw), 'bytes')
from adk_submission import ModelRegistry, compile_submission, discover_declared_models
from swegemma.context import SwegemmaContext
from swegemma.models import Task
from swegemma.harness.agent_runner import resolve_swegemma_adapter
from google.adk.models.lite_llm import LiteLlm
def build_models(d):   # same recipe as swegemma 0.2.11's own CLI: one LiteLlm per declared model (the endpoint is irrelevant for compilation)
    reg = ModelRegistry()
    for m in discover_declared_models(d):
        reg.register(m.strip(), LiteLlm(model='openai/' + m.strip(), api_base='http://127.0.0.1:8000/v1', api_key='EMPTY'))
    return reg
task = Task(instance_id='d', repo='d/d', base_commit='0'*40, problem_statement='d')
ctx = SwegemmaContext(docker_manager=None, container_id='', problem_statement='d', hints_text='', task=task, repo='d/d', graph_dir='/tmp', embeddings_dir='/tmp')
tools = ctx.create_tools()
ok_all = True
def check(label, d):
    global ok_all
    files = {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(d).rglob('*') if p.is_file()}
    same = files == exp['files']
    print(f'  [{label}] extracted files:', sorted(files), '| identical to expected contents:', same)
    try:
        models = build_models(d)
        agent = compile_submission(submission_dir=Path(d), tool_registry=tools, model_registry=models, adapter_resolver_fn=lambda b, i: resolve_swegemma_adapter(b, i, models))
        print(f'  [{label}] COMPILE OK: {agent.name}'); ok = True
    except Exception as e:
        print(f'  [{label}] COMPILE FAILED: {type(e).__name__}: {str(e)[:300]}'); ok = False
    ok_all = ok_all and same and ok
dA = Path('/tmp/ex_py'); shutil.rmtree(dA, ignore_errors=True); dA.mkdir()
with zipfile.ZipFile(zp) as z:
    print('  stored names (Linux zipfile):', z.namelist()); z.extractall(dA)
check('python-extract', dA)
dB = Path('/tmp/ex_unzip'); shutil.rmtree(dB, ignore_errors=True); dB.mkdir()
r = subprocess.run(['unzip', '-o', str(zp), '-d', str(dB)], capture_output=True, text=True)
print('  unzip rc:', r.returncode, (r.stderr or '').strip()[:200])
check('unzip-extract', dB)
print('\nZIP CHECK RESULT:', 'PASS' if ok_all else 'FAIL')'''


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


nb = {"cells": [code(e1.INSTALL), code(TEST)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / f"zip_check_{tag}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": f"noobhunterjay/gemma-4-zipcheck-{tag}", "title": f"Gemma 4 Zip Check {tag}", "code_file": f"zip_check_{tag}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", f"noobhunterjay/gemma4-zipcheck-{tag}"], "kernel_sources": [],
    "competition_sources": ["gemma-4-developer-agent"], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT, "and", DS, "| zip sha256", hashlib.sha256(raw).hexdigest()[:16])
