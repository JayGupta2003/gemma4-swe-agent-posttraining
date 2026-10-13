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


def build_taskdeps(wheels_dir: Path, target: Path) -> dict:
    wheels = sorted(Path(wheels_dir).glob("*.whl"))
    kinds = {w: classify(w) for w in wheels}
    use = [str(w) for w, k in kinds.items() if k == "ok"]
    target.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", str(target), *use], capture_output=True, text=True)
    return {"rc": r.returncode, "installed": len(use), "denied": sorted(w.name for w, k in kinds.items() if k == "deny"),
            "plugins_excluded": sorted(w.name for w, k in kinds.items() if k == "plugin"), "stderr_tail": (r.stderr or "")[-300:]}
