"""Validate one SWE-rebench task in a plain online environment (used by the G7 spike notebook; dry-run locally first).

run_task(task, work_root) -> dict with a verdict:
  valid             FAIL_TO_PASS all fail/error before the gold patch and all pass after; PASS_TO_PASS all pass after
  p2p_partial       FAIL_TO_PASS behave, but some PASS_TO_PASS tests do not pass after (usable with those tests dropped)
  f2p_passes_before at least one FAIL_TO_PASS test already passes with only the test patch (task does not test the fix)
  f2p_fails_after   the gold patch does not make FAIL_TO_PASS pass in this environment
  install_failed / clone_failed / patch_apply_failed / no_tests_ran / timeout
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STATUS = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(.+?)(?:\s+-\s+.*)?$", re.M)


def _run(cmd, cwd, timeout, env, shell=False, inp=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, shell=shell, capture_output=True, text=True, timeout=timeout, input=inp, errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired as e:
        return -999, ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + "\n[timeout]", time.time() - t0


def _tail(s, n=900):
    return s[-n:] if s else ""


def _statuses(out):
    return {m.group(2).strip(): m.group(1) for m in STATUS.finditer(out)}


def run_task(task, work_root, keep=False):
    iid, repo = task["instance_id"], task["repo"]
    d = Path(work_root) / iid
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    rdir, vdir = d / "repo", d / "venv"
    res = {"id": iid, "repo": repo, "python_requested": (task.get("install_config") or {}).get("python"), "times": {}}
    binname = "Scripts" if os.name == "nt" else "bin"
    env = dict(os.environ)
    env.update({"PIP_DISABLE_PIP_VERSION_CHECK": "1", "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1"})
    for k, v in ((task.get("install_config") or {}).get("env_vars") or {}).items():
        if v is not None:
            env[k] = str(v)
    try:
        # ---- clone at base_commit
        rdir.mkdir()
        for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", f"https://github.com/{repo}.git"]):
            _run(cmd, rdir, 120, env)
        rc, out, dt = _run(["git", "fetch", "-q", "--depth", "1", "origin", task["base_commit"]], rdir, 900, env)
        if rc != 0:
            rc, out, dt2 = _run(["git", "fetch", "-q", "origin", task["base_commit"]], rdir, 1500, env)
            dt += dt2
        if rc == 0:
            rc, out2, dt3 = _run(["git", "checkout", "-q", "FETCH_HEAD"], rdir, 300, env)
            out += out2
            dt += dt3
        res["times"]["clone"] = round(dt)
        if rc != 0:
            res.update(verdict="clone_failed", reason=_tail(out, 300))
            return res
        # ---- venv + install
        rc, out, dt = _run([sys.executable, "-m", "venv", str(vdir)], d, 300, env)
        if rc != 0:
            res.update(verdict="install_failed", reason="venv: " + _tail(out, 300))
            return res
        venv_bin = vdir / binname
        env["VIRTUAL_ENV"] = str(vdir)
        env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
        vpy = str(venv_bin / ("python.exe" if os.name == "nt" else "python"))
        rc, out, dt = _run([vpy, "-m", "pip", "install", "-q", "pytest", "pytest-timeout"], rdir, 600, env)
        ic = task.get("install_config") or {}
        steps = list(ic.get("pre_install") or []) + ([ic["install"]] if ic.get("install") else ["pip install -e ."])
        install_out = ""
        for s in steps:
            rc, o, dt2 = _run(s, rdir, 1500, env, shell=True)
            dt += dt2
            install_out += f"$ {s}\n{_tail(o, 600)}\n"
            if rc != 0:
                res["times"]["install"] = round(dt)
                res.update(verdict="install_failed", reason=f"`{s}` rc={rc}: {_tail(o, 400)}")
                return res
        res["times"]["install"] = round(dt)
        rc, out, _ = _run([vpy, "-m", "pip", "freeze"], rdir, 120, env)
        res["n_packages"] = len([l for l in out.splitlines() if l.strip()])
        res["freeze"] = out.splitlines()[:400]
        # ---- tests before (test patch only)
        for name, patch in (("test_patch", task["test_patch"]),):
            rc, o, _ = _run(["git", "apply", "--whitespace=nowarn", "-"], rdir, 120, env, inp=patch)
            if rc != 0:
                res.update(verdict="patch_apply_failed", reason=f"{name}: {_tail(o, 300)}")
                return res
        f2p, p2p = list(task["FAIL_TO_PASS"]), list(task["PASS_TO_PASS"])[:150]
        base = [vpy, "-m", "pytest", "-rA", "--tb=no", "-q", "-p", "no:cacheprovider", "--timeout=300"]
        rc, out_b, dt = _run(base + f2p, rdir, 1200, env)
        res["times"]["tests_before"] = round(dt)
        sb = _statuses(out_b)
        if rc == -999:
            res.update(verdict="timeout", reason="tests before the patch timed out")
            return res
        if not any(i in sb for i in f2p) and not sb:
            res.update(verdict="no_tests_ran", reason=_tail(out_b, 400))
            return res
        passed_before = [i for i in f2p if sb.get(i) == "PASSED"]
        # ---- gold patch, tests after
        rc, o, _ = _run(["git", "apply", "--whitespace=nowarn", "-"], rdir, 120, env, inp=task["patch"])
        if rc != 0:
            res.update(verdict="patch_apply_failed", reason="gold patch: " + _tail(o, 300))
            return res
        rc, out_a, dt = _run(base + f2p + p2p, rdir, 1800, env)
        res["times"]["tests_after"] = round(dt)
        sa = _statuses(out_a)
        if rc == -999:
            res.update(verdict="timeout", reason="tests after the patch timed out")
            return res
        f2p_ok = [i for i in f2p if sa.get(i) == "PASSED"]
        p2p_ok = [i for i in p2p if sa.get(i) == "PASSED"]
        res.update(f2p=len(f2p), f2p_pass_before=len(passed_before), f2p_pass_after=len(f2p_ok), p2p=len(p2p), p2p_pass_after=len(p2p_ok))
        if passed_before:
            res.update(verdict="f2p_passes_before", reason=f"{len(passed_before)}/{len(f2p)} F2P already pass with only the test patch")
        elif len(f2p_ok) != len(f2p):
            bad = [(i, sa.get(i)) for i in f2p if sa.get(i) != "PASSED"][:3]
            res.update(verdict="f2p_fails_after", reason=f"{bad} | {_tail(out_a, 300)}")
        elif len(p2p_ok) != len(p2p):
            res.update(verdict="p2p_partial", reason=f"{len(p2p) - len(p2p_ok)}/{len(p2p)} P2P not passing after the gold patch")
        else:
            res.update(verdict="valid", reason="")
        return res
    except Exception as e:  # never let one task kill the spike
        res.update(verdict="harness_error", reason=repr(e)[:300])
        return res
    finally:
        if not keep:
            shutil.rmtree(d, ignore_errors=True)
