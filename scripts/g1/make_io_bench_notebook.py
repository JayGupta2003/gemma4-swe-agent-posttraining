"""CPU-only: why did loading the 62.5 GB bf16 checkpoint from /kaggle/input run at ~6 MB/s under transformers (mmap, 1188 tensors)?
Measures sequential large-block reads, parallel sequential reads, and a full-shard copy to local disk (/tmp)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "kaggle_assets" / "g1_io_bench"
OUT.mkdir(parents=True, exist_ok=True)
CELL = r'''import json, os, subprocess, time
from pathlib import Path
M = next(p.parent for p in Path('/kaggle/input').rglob('config.json') if 'unquantized' in str(p).lower())
s1, s2 = M / 'model-00001-of-00002.safetensors', M / 'model-00002-of-00002.safetensors'
rep = {'shard_gb': [round(s1.stat().st_size / 1e9, 1), round(s2.stat().st_size / 1e9, 1)]}
def sh(c, timeout=1200):
    t = time.time()
    try: out = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired: out = 'TIMEOUT'
    return out.strip(), round(time.time() - t, 1)
rep['df'] = sh('df -h /tmp /kaggle/working /kaggle/input | cat; free -g | cat; mount | grep -i "kaggle/input" | head -3')[0]
for name, off_mb in (('seq_2GB_at_0', 0), ('seq_2GB_at_20GB', 20000)):
    out, dt = sh(f'dd if={s1} of=/dev/null bs=16M skip={off_mb // 16} count=128 2>&1 | tail -1', 600)
    rep[name] = {'seconds': dt, 'dd': out}
    print(name, rep[name], flush=True)
t = time.time()
procs = [subprocess.Popen(f'dd if={s1} of=/dev/null bs=16M skip={(o * 1000) // 16} count=64 2>/dev/null', shell=True) for o in (5, 15, 30, 40)]
[p.wait() for p in procs]; dt = time.time() - t
rep['parallel4_seq_1GB_each'] = {'seconds': round(dt, 1), 'aggregate_MBps': round(4 * 64 * 16 * 1.048576 / dt, 1)}
print('parallel4', rep['parallel4_seq_1GB_each'], flush=True)
Path('/tmp/ckpt').mkdir(exist_ok=True)
out, dt = sh(f'cp {s2} /tmp/ckpt/ && ls -l /tmp/ckpt | tail -2', 1500)
rep['copy_shard2_to_tmp'] = {'seconds': dt, 'MBps': round(s2.stat().st_size / 1e6 / dt, 1) if dt else None, 'out': out}
print('copy shard2', rep['copy_shard2_to_tmp'], flush=True)
Path('/kaggle/working/io_bench.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))'''
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "g1_io_bench.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "noobhunterjay/gemma-4-g1-io-bench", "title": "Gemma 4 G1 IO Bench", "code_file": "g1_io_bench.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
    "model_sources": ["google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}, indent=2), encoding="utf-8")
print("wrote", OUT)
