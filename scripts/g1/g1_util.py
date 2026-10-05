"""Utilities for the G1 notebook, tested locally (test_g1_util.py) before use on paid GPU time.

stage_checkpoint: copy a checkpoint dir from the slow NFS mount (/kaggle/input: ~90 MB/s per stream, ~320 MB/s with 4 parallel streams)
                  to local disk with many parallel large-block reads; aborts early on poor throughput.
run_supervised:   run a subprocess and kill it if it misses progress milestones (instead of a long fixed timeout that burns GPU quota).
"""

import concurrent.futures as cf
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path


class StageAbort(RuntimeError):
    pass


def _copy_block(src: Path, dst: Path, off: int, n: int) -> int:
    with open(src, "rb", buffering=0) as fi, open(dst, "r+b", buffering=0) as fo:
        fi.seek(off)
        fo.seek(off)
        left = n
        while left:
            buf = fi.read(left)
            if not buf:
                raise IOError(f"short read from {src} at {off + n - left}")
            fo.write(buf)
            left -= len(buf)
    return n


def stage_checkpoint(src: Path, dst: Path, workers: int = 12, block: int = 64 << 20, min_mbps_after_s: float = 180, min_mbps: float = 30,
                     max_seconds: float = 1500, log=print) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    jobs, total = [], 0
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        size = f.stat().st_size
        total += size
        if size < block:
            shutil.copy(f, dst / f.name)
            continue
        with open(dst / f.name, "wb") as o:
            o.truncate(size)
        jobs += [(f, dst / f.name, off, min(block, size - off)) for off in range(0, size, block)]
    t0, done = time.time(), 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_copy_block, *j) for j in jobs]
        pending = set(futs)
        last_log = 0.0
        while pending:
            fin, pending = cf.wait(pending, timeout=5, return_when=cf.FIRST_COMPLETED)
            for fu in fin:
                done += fu.result()  # re-raises read errors
            el = time.time() - t0
            if el - last_log >= 20:
                last_log = el
                log(f"[stage] {done / 1e9:.1f}/{sum(j[3] for j in jobs) / 1e9:.1f} GB, {done / 1e6 / max(el, 1e-9):.0f} MB/s, {el:.0f}s")
            if el > min_mbps_after_s and done / 1e6 / el < min_mbps:
                ex.shutdown(wait=False, cancel_futures=True)
                raise StageAbort(f"throughput {done / 1e6 / el:.0f} MB/s < {min_mbps} after {el:.0f}s")
            if el > max_seconds:
                ex.shutdown(wait=False, cancel_futures=True)
                raise StageAbort(f"staging exceeded {max_seconds}s")
    el = time.time() - t0
    return {"bytes": total, "seconds": round(el, 1), "MBps": round(sum(j[3] for j in jobs) / 1e6 / max(el, 1e-9), 1), "workers": workers}


def run_supervised(cmd: list[str], env: dict, logfile: Path, milestones: list[tuple[str, float]], max_seconds: float, poll: float = 10) -> dict:
    """milestones: [(substring that must appear in the log, deadline in seconds since start)]. Missing a deadline kills the process."""
    t0 = time.time()
    with open(logfile, "w") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        reached: dict[str, float] = {}
        verdict = "finished"
        while p.poll() is None:
            time.sleep(poll)
            el = time.time() - t0
            txt = Path(logfile).read_text(errors="ignore")
            for pat, deadline in milestones:
                if pat not in reached and pat in txt:
                    reached[pat] = round(el, 1)
            missed = [pat for pat, deadline in milestones if pat not in reached and el > deadline]
            if missed:
                verdict = f"killed: milestone '{missed[0]}' not reached by {dict(milestones)[missed[0]]:.0f}s"
                break
            if el > max_seconds:
                verdict = f"killed: exceeded {max_seconds:.0f}s"
                break
        if p.poll() is None:
            p.kill()
            p.wait()
    return {"returncode": p.returncode, "verdict": verdict, "seconds": round(time.time() - t0, 1), "milestones_reached": reached}
