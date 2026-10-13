"""Build a harness-ready task pool from the G8 validation results (runs in an INTERNET-enabled CPU Kaggle notebook; dry-run locally with --limit).

Output layout (the competition's, so swegemma's Evaluator / verify_task can consume it unchanged):
  <out>/tasks.jsonl                    instance_id, repo, base_commit, problem_statement, hints_text, patch, test_patch, created_at, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit
  <out>/snapshots/<instance_id>.tgz    repo at base_commit rooted at '.', WITH a .git directory (HEAD -> refs/heads/main), like the competition's snapshots
  <out>/wheels/*.whl                   shared wheel set: highest version per package over the usable tasks' freeze lists, minus packages the Kaggle image already has at >= that version
  <out>/manifest.json                  per-task dependency status, dropped packages, sizes

Usage: python build_pool.py <results_dir_or_files...> --candidates <candidates.jsonl> --out <dir> [--limit N]
"""

import argparse
import concurrent.futures as cf
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from packaging.version import InvalidVersion, Version

MAX_SNAPSHOT_MB = 150
MAX_WHEEL_MB = 150


def run(cmd, cwd=None, timeout=600, env=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -999, "timeout"


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def make_snapshot(task, out_dir, work):
    iid, repo = task["instance_id"], task["repo"]
    d = Path(work) / iid
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    for cmd in (["git", "init", "-q", "-b", "main"], ["git", "remote", "add", "origin", f"https://github.com/{repo}.git"]):
        run(cmd, d, 120, env)
    rc, out = run(["git", "fetch", "-q", "--depth", "1", "origin", task["base_commit"]], d, 900, env)
    if rc != 0:
        rc, out = run(["git", "fetch", "-q", "origin", task["base_commit"]], d, 1500, env)
    if rc == 0:
        rc, out = run(["git", "checkout", "-q", "-B", "main", "FETCH_HEAD"], d, 300, env)
    if rc != 0:
        shutil.rmtree(d, ignore_errors=True)
        return {"ok": False, "reason": "git: " + out[-200:]}
    tgz = Path(out_dir) / "snapshots" / f"{iid}.tgz"
    with tarfile.open(tgz, "w:gz", compresslevel=3) as t:
        t.add(d, arcname=".")
    mb = tgz.stat().st_size / 1e6
    shutil.rmtree(d, ignore_errors=True)
    if mb > MAX_SNAPSHOT_MB:
        tgz.unlink()
        return {"ok": False, "reason": f"snapshot {mb:.0f} MB > {MAX_SNAPSHOT_MB}"}
    return {"ok": True, "mb": round(mb, 1)}


def host_versions():
    out = {}
    for dist in md.distributions():
        n = dist.metadata["Name"]
        if n:
            out[norm(n)] = dist.version
    return out


def parse_freeze(lines):
    deps = {}
    for l in lines:
        l = l.strip()
        if not l or l.startswith("-e") or " @ " in l or "==" not in l:
            continue
        n, v = l.split("==", 1)
        deps[norm(n)] = v.split(";")[0].strip()
    return deps


def vkey(v):
    try:
        return Version(v)
    except InvalidVersion:
        return Version("0")


def fetch_wheel(name, versions, wheels_dir):
    """Try the highest version first, then lower ones seen in freezes; wheel first, build from sdist as a fallback."""
    for v in versions[:3]:
        rc, out = run([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--only-binary=:all:", "-d", str(wheels_dir), f"{name}=={v}"], timeout=300)
        if rc == 0:
            return {"ok": True, "version": v, "how": "wheel"}
    for v in versions[:2]:
        rc, out = run([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "-w", str(wheels_dir), f"{name}=={v}"], timeout=420)
        if rc == 0:
            return {"ok": True, "version": v, "how": "built"}
    return {"ok": False, "reason": out[-160:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    (out / "wheels").mkdir(exist_ok=True)
    rows = []
    for p in a.results:
        rows += json.load(open(p, encoding="utf-8"))
    cand = {}
    for l in open(a.candidates, encoding="utf-8"):
        t = json.loads(l)
        cand[t["instance_id"]] = t
    usable = [r for r in rows if r["verdict"] in ("valid", "p2p_partial") and r["id"] in cand]
    if a.limit:
        usable = usable[: a.limit]
    print(f"{len(rows)} results, {len(usable)} usable", flush=True)
    t0 = time.time()
    manifest = {"tasks": {}, "wheels": {}, "oversize_wheels": [], "usable": len(usable)}

    # ---- 1. snapshots
    work = Path(os.environ.get("TMPDIR", "/tmp")) / "g9_snap"
    work.mkdir(parents=True, exist_ok=True)
    snap = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(make_snapshot, cand[r["id"]], out, work): r["id"] for r in usable}
        for k, f in enumerate(cf.as_completed(futs), 1):
            snap[futs[f]] = f.result()
            if k % 25 == 0 or k == len(futs):
                print(f"[{time.time() - t0:5.0f}s] snapshots {k}/{len(futs)} ok={sum(1 for v in snap.values() if v['ok'])}", flush=True)
    kept = [r for r in usable if snap[r["id"]]["ok"]]

    # ---- 2. shared wheel set
    host = host_versions()
    want = {}   # name -> sorted list of versions seen (high to low)
    task_deps = {}
    for r in kept:
        deps = parse_freeze(r.get("freeze", []))
        task_deps[r["id"]] = deps
        for n, v in deps.items():
            want.setdefault(n, set()).add(v)
    need = {}
    for n, vs in want.items():
        vs = sorted(vs, key=vkey, reverse=True)
        hv = host.get(n)
        if hv is not None and vkey(hv) >= vkey(vs[0]):
            continue   # the Kaggle image already has this package at least as new
        need[n] = vs
    print(f"{len(want)} distinct packages in freezes; {len(need)} not covered by the host image -> downloading", flush=True)
    res = {}
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(fetch_wheel, n, vs, out / "wheels"): n for n, vs in need.items()}
        for k, f in enumerate(cf.as_completed(futs), 1):
            res[futs[f]] = f.result()
            if k % 50 == 0 or k == len(futs):
                print(f"[{time.time() - t0:5.0f}s] wheels {k}/{len(futs)} ok={sum(1 for v in res.values() if v['ok'])}", flush=True)
    for w in list((out / "wheels").glob("*.whl")):
        if w.stat().st_size > MAX_WHEEL_MB * 1e6:
            manifest["oversize_wheels"].append(w.name)
            w.unlink()
    manifest["wheels"] = {n: v for n, v in res.items()}
    bad_pkgs = {n for n, v in res.items() if not v["ok"]} | {norm(re.split(r"[-_]\d", w)[0]) for w in manifest["oversize_wheels"]}

    # ---- 3. tasks.jsonl + manifest
    n_complete = 0
    with open(out / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for r in kept:
            t = cand[r["id"]]
            missing = sorted(n for n in task_deps[r["id"]] if n in bad_pkgs)
            manifest["tasks"][r["id"]] = {"verdict": r["verdict"], "snapshot_mb": snap[r["id"]]["mb"], "missing_deps": missing, "n_deps": len(task_deps[r["id"]])}
            n_complete += not missing
            fh.write(json.dumps({k: t[k] for k in ("instance_id", "repo", "base_commit", "problem_statement", "hints_text", "patch", "test_patch", "created_at",
                                                   "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit")}) + "\n")
    manifest["snapshots_ok"] = len(kept)
    manifest["tasks_with_complete_deps"] = n_complete
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    wsize = sum(w.stat().st_size for w in (out / "wheels").glob("*.whl")) / 1e9
    ssize = sum(s.stat().st_size for s in (out / "snapshots").glob("*.tgz")) / 1e9
    print(f"\nDONE in {time.time() - t0:.0f}s | tasks {len(kept)} (complete deps {n_complete}) | snapshots {ssize:.2f} GB | wheels {len(list((out / 'wheels').glob('*.whl')))} files {wsize:.2f} GB "
          f"| failed packages {len(bad_pkgs)} | oversize wheels {len(manifest['oversize_wheels'])}", flush=True)
    if bad_pkgs:
        print("packages without a usable wheel (first 30):", sorted(bad_pkgs)[:30])


if __name__ == "__main__":
    main()
