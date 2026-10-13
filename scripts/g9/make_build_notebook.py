"""Generate kaggle_assets/g9_build/ : CPU + internet notebook that runs scripts/g9/build_pool.py over the G8 results (kernel outputs of g8-scale-0/1, mounted via
kernel_sources) and the candidates dataset. Output: /kaggle/working/pool/{tasks.jsonl,snapshots/,wheels/,manifest.json}.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kaggle_assets" / "g9_build"
OUT.mkdir(parents=True, exist_ok=True)
SRC = (Path(__file__).resolve().parent / "build_pool.py").read_text(encoding="utf-8").replace("\r\n", "\n")

RUN = '''import subprocess, sys, os, glob
from pathlib import Path
Path('/kaggle/working/build_pool.py').write_text(BUILD_SRC, encoding='utf-8')
res = sorted(glob.glob('/kaggle/input/**/scale_results_*.json', recursive=True))
cand = glob.glob('/kaggle/input/**/candidates.jsonl', recursive=True)
print('result files:', res, '| candidates:', cand)
assert len(res) == 2 and len(cand) == 1, (res, cand)
rc = subprocess.run([sys.executable, '-u', '/kaggle/working/build_pool.py', *res, '--candidates', cand[0], '--out', '/kaggle/working/pool', '--workers', '6']).returncode
print('build_pool exit code', rc)
assert rc == 0'''


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


nb = {"cells": [code("BUILD_SRC = " + json.dumps(SRC)), code(RUN)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "build.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g9-build", "title": "Gemma 4 G9 Build", "code_file": "build.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "keywords": [],
    "dataset_sources": ["noobhunterjay/gemma4-g8-candidates"],
    "kernel_sources": ["noobhunterjay/gemma-4-g8-scale-0", "noobhunterjay/gemma-4-g8-scale-1"],
    "competition_sources": [], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT)
