"""Per-task pinned environments for the FreshSWE-Py tasks (moshegolderson/freshswe-py) under swegemma's SUBPROCESS sandbox.

Why: G24 (E4B self-RFT on 37 pool tasks) was null; staff confirmed the hidden test repos are disjoint from the training repos, so more data must
come from MORE unseen repos. FreshSWE-Py gives 663 tasks from 33 repos (none of fastapi/rich/requests/httpx), created after Gemma 4's release,
validated under the scorer's conventions with one pinned environment per task (locks/<base_commit>.txt, built for CPython 3.13 in Docker).

Our Kaggle kernels run swegemma with sandbox='subprocess': every sandbox gets a fresh venv that sees the host through `_host_env.pth`, and
`install_test_dependencies` returns immediately (no sandbox/setup.py runs). So the task environment has to be put on the venv's path by us:

  1. build_store (CPU kernel, internet on, Python 3.12 like the GPU kernels): every distinct `name==version` pin of every lock is installed
     with `pip install --no-deps --target store/<name>-<version>/site-packages` (FreshSWE's wheelhouse first, then PyPI binary, then sdist);
     envs.json maps base_commit -> the store dirs of its lock. Packages the host/harness must own (pytest core, pip, setuptools, wheel, the
     harness's own pytest plugins) are skipped, as swegemma's container setup skips them.
  2. install_hook (any kernel running the harness): wraps `install_test_dependencies` in every swegemma module that imported it. For a
     subprocess sandbox it reads the task's base commit from the snapshot's root commit message (`base <sha>`, FreshSWE's convention) and writes
     `A_freshswe.pth` into the sandbox venv: the workspace and its top-level directories (the competition's sandbox/setup.py fast path), then the
     task's store dirs. `A_` sorts before `_host_env.pth`, so the task's pins come before any host copy. The hook runs for the agent sandbox and
     for every verification sandbox (both call install_test_dependencies after extracting the snapshot) and is idempotent.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# The host's pytest runs the tests, so pytest itself, its core dependencies, the harness's pytest-timeout and the installers stay the host's.
# Plugins (pytest-asyncio, pytest-xdist, ...) come from the task's lock: the environment is per task, and the Kaggle host lacks some of them
# (G25a re-check: strawberry's pytest-aiohttp needs pytest_asyncio).
SKIP = {"pytest", "pytest-timeout", "pluggy", "iniconfig", "exceptiongroup", "tomli", "pip", "setuptools", "wheel"}
PTH_NAME = "A_freshswe.pth"


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(text: str) -> list[tuple[str, str]]:
    """`name==version` pins from a lock/requirements file; extras, markers, hashes, comments and option lines are dropped."""
    pins = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-") or "==" not in line:
            continue
        name, ver = line.split("==", 1)
        name = norm(name.split("[", 1)[0].strip())
        ver = re.split(r"[;\s\\]", ver.strip(), maxsplit=1)[0]
        if name and ver:
            pins.append((name, ver))
    return pins


def store_dir_name(name: str, ver: str) -> str:
    return f"{norm(name)}-{ver}"


def _pip_target(spec: str, target: Path, extra: list[str], timeout: int) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "--no-compile", "--target", str(target), *extra, spec]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
        return p.returncode, (p.stdout + p.stderr)[-600:]
    except subprocess.TimeoutExpired:
        return -9, "timeout"


def install_pin(name: str, ver: str, store: Path, wheelhouse: Path | None) -> dict:
    """Install one pin into store/<name>-<ver>/site-packages (atomic rename). Returns {'source': wheelhouse|pypi|sdist|cached|None, 'log': ...}."""
    final = store / store_dir_name(name, ver) / "site-packages"
    if final.is_dir():
        return {"source": "cached"}
    spec = f"{name}=={ver}"
    tmp = store / ".tmp" / store_dir_name(name, ver)
    attempts = []
    if wheelhouse is not None:
        attempts.append(("wheelhouse", ["--only-binary=:all:", "--no-index", "--find-links", str(wheelhouse)], 300))
    attempts += [("pypi", ["--only-binary=:all:"], 300), ("sdist", [], 600)]
    log = ""
    for source, extra, timeout in attempts:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        rc, log = _pip_target(spec, tmp, extra, timeout)
        if rc == 0:
            for junk in ("bin",):
                shutil.rmtree(tmp / junk, ignore_errors=True)
            final.parent.mkdir(parents=True, exist_ok=True)
            tmp.rename(final)
            return {"source": source}
    shutil.rmtree(tmp, ignore_errors=True)
    return {"source": None, "log": log}


def build_store(tasks: list[dict], locks_dir: Path, store: Path, wheelhouse: Path | None, workers: int = 8, log=print) -> dict:
    """Install every distinct pin of the tasks' locks; return {'envs': {base_commit: [dir names]}, 'missing': {...}, 'stats': {...}}."""
    store.mkdir(parents=True, exist_ok=True)
    per_task, all_pins, no_lock = {}, set(), []
    for t in tasks:
        lock = locks_dir / f"{t['base_commit']}.txt"
        if not lock.exists():
            no_lock.append(t["instance_id"])
            continue
        pins = [(n, v) for n, v in parse_lock(lock.read_text(encoding="utf-8", errors="replace")) if n not in SKIP]
        per_task[t["base_commit"]] = pins
        all_pins |= set(pins)
    log(f"store: {len(per_task)} locks, {len(all_pins)} distinct pins, {len(no_lock)} tasks without a lock")
    results, t0 = {}, time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(install_pin, n, v, store, wheelhouse): (n, v) for n, v in sorted(all_pins)}
        for i, f in enumerate(cf.as_completed(futs), 1):
            results[futs[f]] = f.result()
            if i % 100 == 0 or i == len(futs):
                log(f"  pins {i}/{len(futs)} ({time.time() - t0:.0f} s), failed so far {sum(1 for r in results.values() if r['source'] is None)}")
    shutil.rmtree(store / ".tmp", ignore_errors=True)
    envs, missing = {}, {}
    for sha, pins in per_task.items():
        envs[sha] = [store_dir_name(n, v) for n, v in pins if results[(n, v)]["source"] is not None]
        miss = [f"{n}=={v}" for n, v in pins if results[(n, v)]["source"] is None]
        if miss:
            missing[sha] = miss
    sources = {}
    for r in results.values():
        sources[str(r["source"])] = sources.get(str(r["source"]), 0) + 1
    failed = {f"{n}=={v}": r.get("log", "")[-300:] for (n, v), r in results.items() if r["source"] is None}
    return {"envs": envs, "missing": missing, "failed_pins": failed, "no_lock": no_lock,
            "stats": {"locks": len(per_task), "pins": len(all_pins), "sources": sources, "seconds": round(time.time() - t0)}}


def base_commit_of(workspace: Path) -> str | None:
    """FreshSWE snapshots hold one root commit whose subject is `base <sha>`; later commits (the harness baseline) sit on top of it."""
    try:
        out = subprocess.run(["git", "-C", str(workspace), "log", "--format=%s"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines()[::-1]:
        parts = line.split()
        if len(parts) == 2 and parts[0] == "base" and len(parts[1]) == 40:
            return parts[1]
    return None


def pth_lines(workspace: Path, store: Path, dirs: list[str]) -> list[str]:
    ws = [str(workspace)] + [str(p) for p in sorted(workspace.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    return ws + [str(store / d / "site-packages") for d in dirs]


PRETEND_VERSION = "99.0.0"   # FreshSWE's image sets SETUPTOOLS_SCM_PRETEND_VERSION=99.0.0
SCM_TEMPLATE = ("# generated by fresh_env (stands in for the editable install's setuptools-scm / hatch-vcs version file)\n"
                "__version__ = version = '{v}'\n__version_tuple__ = version_tuple = ({t})\n__commit_id__ = commit_id = None\n")


def project_meta(ws: Path) -> dict:
    """What an editable install would provide: the distribution name and version, and the version file a VCS build hook writes."""
    import configparser  # noqa: PLC0415
    import tomllib  # noqa: PLC0415
    meta = {"name": None, "version": None, "version_files": []}
    pp = ws / "pyproject.toml"
    if pp.is_file():
        try:
            d = tomllib.loads(pp.read_text(encoding="utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            d = {}
        proj, tool = d.get("project", {}) or {}, d.get("tool", {}) or {}
        meta["name"] = proj.get("name") or (tool.get("poetry", {}) or {}).get("name")
        v = proj.get("version") or (tool.get("poetry", {}) or {}).get("version")
        meta["version"] = v if isinstance(v, str) and "version" not in (proj.get("dynamic") or []) else None
        vcs = (((tool.get("hatch", {}) or {}).get("build", {}) or {}).get("hooks", {}) or {}).get("vcs", {}) or {}
        scm = tool.get("setuptools_scm", tool.get("setuptools-scm", {})) or {}
        pdm = ((tool.get("pdm", {}) or {}).get("version", {}) or {})
        for f in (vcs.get("version-file"), vcs.get("version_file"), scm.get("version_file"), scm.get("write_to"), pdm.get("write_to")):
            if isinstance(f, str) and f.endswith(".py"):
                meta["version_files"].append(f)
    sc = ws / "setup.cfg"
    if not meta["name"] and sc.is_file():
        cp = configparser.ConfigParser()
        try:
            cp.read_string(sc.read_text(encoding="utf-8", errors="replace"))
            meta["name"] = cp.get("metadata", "name", fallback=None)
        except configparser.Error:
            pass
    return meta


def materialize_project(ws: Path, site_packages: list[Path]) -> str:
    """Emulate the parts of `pip install -e .` the tests rely on: the VCS version file (only if absent) and a minimal .dist-info
    (so importlib.metadata.version(<project>) works) in the sandbox venv. Returns a short status."""
    m = project_meta(ws)
    ver = m["version"] or PRETEND_VERSION
    wrote = []
    for rel in m["version_files"]:
        for cand in (ws / rel, ws / "src" / rel):
            if cand.parent.is_dir():
                if not cand.exists():
                    cand.write_text(SCM_TEMPLATE.format(v=PRETEND_VERSION, t=", ".join(PRETEND_VERSION.split("."))), encoding="utf-8")
                    wrote.append(rel)
                break
    if m["name"]:
        dist = re.sub(r"[-_.]+", "_", m["name"])
        for sp in site_packages:
            di = sp / f"{dist}-{ver}.dist-info"
            di.mkdir(exist_ok=True)
            (di / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {m['name']}\nVersion: {ver}\n", encoding="utf-8")
            (di / "INSTALLER").write_text("fresh_env\n", encoding="utf-8")
    return f"name={m['name']},ver={ver},vfiles={len(wrote)}"


def write_env_pth(paths: dict, store: Path, envs: dict) -> str:
    """Write A_freshswe.pth into the sandbox venv. Returns a short status string (for logging/tests)."""
    workspace, venv = Path(paths["workspace"]), Path(paths["venv"])
    sha = base_commit_of(workspace)
    if sha is None:
        return "no-base-commit"
    if sha not in envs:
        return f"no-env:{sha[:10]}"
    sps = list(venv.glob("lib/python*/site-packages"))
    materialize_project(workspace, sps)   # before pth_lines: may not create top-level dirs, only files inside existing ones
    lines = pth_lines(workspace, store, envs[sha])
    for sp in sps:
        (sp / PTH_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"ok:{sha[:10]}:{len(envs[sha])}:{len(sps)}"


def install_hook(store: Path, envs: dict, log=None) -> list[str]:
    """Wrap install_test_dependencies in every loaded swegemma module. Returns the patched module names."""
    import swegemma.harness.container_setup as cs  # noqa: PLC0415

    orig = getattr(cs.install_test_dependencies, "_fresh_orig", cs.install_test_dependencies)
    stats = install_hook.stats = {}

    def hooked(docker, container_id, *args, **kwargs):
        try:
            if type(docker).__name__ in {"SubprocessManager", "SubprocessSandbox"} and hasattr(docker, "sandboxes"):
                status = write_env_pth(docker.sandboxes[container_id], store, envs)
                key = status.split(":", 1)[0]
                stats[key] = stats.get(key, 0) + 1
                if log and key != "ok":
                    log(f"fresh_env hook: {status}")
        except Exception as e:  # never break the harness; the task then runs on the host environment and is reported by the gold check
            stats["error"] = stats.get("error", 0) + 1
            if log:
                log(f"fresh_env hook error: {type(e).__name__}: {e}")
        return orig(docker, container_id, *args, **kwargs)

    hooked._fresh_orig = orig
    patched = []
    for name, mod in list(sys.modules.items()):
        fn = getattr(mod, "install_test_dependencies", None) if name.startswith("swegemma") and mod is not None else None
        if fn is not None and (fn is orig or getattr(fn, "_fresh_orig", None) is orig):
            mod.install_test_dependencies = hooked
            patched.append(name)
    return patched


if __name__ == "__main__":
    print(json.dumps(parse_lock(Path(sys.argv[1]).read_text()), indent=0)[:2000])
