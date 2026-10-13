"""Unpack the shared wheel set into a side directory for the subprocess sandbox (used by the gold check and by the rollout notebooks).

Facts established from the swegemma 0.2.11 sources and the pilots (2026-10-13):
  * in subprocess mode swegemma does NOT inject wheels (install_test_dependencies returns early); the sandbox venv reaches the host through a `_host_env.pth` listing every
    host sys.path entry that contains 'site-packages'; the venv also has include-system-site-packages, and entries added by that .pth come BEFORE the system dist-packages,
    so anything in the side directory OVERRIDES the host's copy;
  * therefore: never put pytest, its core dependencies or pytest PLUGINS in the side directory (a plugin such as pytest-sanic is auto-loaded for every task and crashes
    importing a package that is not installed). The harness itself skips pytest, pytest_timeout, pytest_asyncio, pytest_httpbin, pytest_xdist, pluggy, pip, setuptools, wheel.
"""

import re
import subprocess
import sys
from pathlib import Path

DENY = {"pytest", "pytest_timeout", "pytest_asyncio", "pytest_httpbin", "pytest_xdist", "pluggy", "pip", "setuptools", "wheel", "iniconfig", "packaging", "exceptiongroup", "tomli"}
PLUGIN_WHITELIST = {"pytest_mock", "pytest_cov", "pytest_asyncio", "pytest_xdist"}   # entry-point plugins that are inert unless the tests ask for them


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def classify(wheel: Path) -> str:
    """'ok' | 'deny' | 'plugin'. A pytest PLUGIN package (name starts with pytest_ / pytest-) is excluded unless whitelisted: its entry point is auto-loaded for every task.
    Ordinary libraries that merely ship a pytest11 entry point (platformdirs, hypothesis, ...) are fine: they are installed anyway, so their plugin imports cleanly."""
    pkg = norm(wheel.name.split("-", 1)[0])
    if pkg in DENY:
        return "deny"
    if pkg.startswith("pytest_") and pkg not in PLUGIN_WHITELIST:
        return "plugin"
    return "ok"


def strip_pytest_entry_points(target: Path) -> list:
    """Remove the [pytest11] section from every dist-info in the side directory (except whitelisted plugins). pytest auto-loads entry-point plugins for EVERY task, and a
    plugin from an ordinary package can crash at import (pilot v3: hydra-core's plugin imported omegaconf/antlr4 with a mismatching runtime in all 12 tasks)."""
    stripped = []
    for ep in Path(target).glob("*.dist-info/entry_points.txt"):
        pkg = norm(ep.parent.name.split("-", 1)[0])   # "pytest_mock-3.14.0.dist-info" -> "pytest_mock"
        text = ep.read_text(encoding="utf-8", errors="ignore")
        if "[pytest11]" not in text or pkg in PLUGIN_WHITELIST:
            continue
        out, skip = [], False
        for line in text.splitlines():
            if line.strip().startswith("["):
                skip = line.strip() == "[pytest11]"
            if not skip:
                out.append(line)
        ep.write_text("\n".join(out) + "\n", encoding="utf-8")
        stripped.append(pkg)
    return sorted(stripped)


def build_taskdeps(wheels_dir: Path, target: Path) -> dict:
    wheels = sorted(Path(wheels_dir).glob("*.whl"))
    kinds = {w: classify(w) for w in wheels}
    use = [str(w) for w, k in kinds.items() if k == "ok"]
    target.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", str(target), *use], capture_output=True, text=True)
    stripped = strip_pytest_entry_points(target) if r.returncode == 0 else []
    return {"rc": r.returncode, "installed": len(use), "pytest_entry_points_stripped": stripped, "denied": sorted(w.name for w, k in kinds.items() if k == "deny"),
            "plugins_excluded": sorted(w.name for w, k in kinds.items() if k == "plugin"), "stderr_tail": (r.stderr or "")[-300:]}
