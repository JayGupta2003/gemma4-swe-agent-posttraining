"""Era-aware dependency resolution for third-party task pools (e.g. SWE-Pallets-X).

The problem this solves: our shared data/wheels/ cache has ONE version of each
package. Two failure modes result:
  1. A task from an old era needs an OLDER version than what we have (e.g.
     werkzeug's `_environ_property` was renamed/removed between versions).
  2. Worse: if we add e.g. werkzeug to the shared cache so *flask* tasks can
     import it, that same wheel gets unconditionally injected into *werkzeug's
     own* task containers too (swegemma's container_setup.py has no concept of
     "this package is the repo under test, don't shadow its editable install")
     -- silently overwriting the snapshot's own editable-installed source with
     the wrong version, corrupting werkzeug's own self-tests.

The fix: never put a self-testable repo's package in the shared wheels dir at
all. Instead, build a small PER-TASK wheels directory containing the shared
"safe" wheels (infra libs no task's own repo could collide with) plus
era-correct versions of whatever that specific task's tests actually import,
excluding the task's own repo package (which is always provided via the
snapshot's own editable install, never a wheel).

Usage as a library from a verification script -- see check_gold_patches_era.py.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Packages that are themselves "the repo under test" somewhere in our task pools
# (official 4 + SWE-Pallets-X's 8). A wheel for any of these must never be added
# to a shared bucket -- only ever into a per-task dir that excludes the current
# task's own repo.
SELF_TESTABLE_PACKAGES = {
    "fastapi", "rich", "requests", "httpx",
    "click", "flask", "werkzeug", "jinja2", "markupsafe", "itsdangerous",
    "pytest", "sympy",
}

PYPI_CACHE_DIR = PROJECT_ROOT / "data_extra" / "pypi_release_cache"
WHEEL_CACHE_DIR = PROJECT_ROOT / "data_extra" / "era_wheel_cache"


def _normalize_pkg(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_cutoff(task) -> datetime:
    raw = getattr(task, "fix_date", None) or getattr(task, "created_at", None)
    if raw is None and isinstance(task, dict):
        raw = task.get("fix_date") or task.get("created_at")
    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _pypi_releases(pkg: str) -> dict[str, str]:
    """Version -> first upload_time_iso_8601, cached to disk so repeated eras don't re-hit PyPI."""
    PYPI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PYPI_CACHE_DIR / f"{_normalize_pkg(pkg)}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=20) as r:
                data = json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"  [era_wheels] WARNING: could not fetch PyPI metadata for {pkg}: {e}", file=sys.stderr)
            return {}
        cache_file.write_text(json.dumps(data), encoding="utf-8")

    out: dict[str, str] = {}
    for ver, files in (data.get("releases") or {}).items():
        for fh in files:
            t = fh.get("upload_time_iso_8601") or fh.get("upload_time")
            if t:
                out.setdefault(ver, t)
                break
    return out


def latest_version_before(pkg: str, cutoff: datetime) -> str | None:
    """Latest release of `pkg` published strictly before `cutoff`. None if unknown."""
    best, best_t = None, None
    for ver, t in _pypi_releases(pkg).items():
        try:
            dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt <= cutoff and (best_t is None or dt > best_t):
            best, best_t = ver, dt
    return best


def download_pinned_wheel(pkg: str, version: str) -> Path | None:
    """Download pkg==version (Linux cp313, matching the sandbox container) into a shared
    version-keyed cache, reused across tasks/repos that happen to need the same pin."""
    WHEEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(WHEEL_CACHE_DIR.glob(f"{_normalize_pkg(pkg).replace('-', '_')}-{version}-*.whl"))
    if existing:
        return existing[0]

    before = set(WHEEL_CACHE_DIR.glob("*.whl"))
    cmd = [
        sys.executable, "-m", "pip", "download", "--no-deps",
        "-d", str(WHEEL_CACHE_DIR),
        "--platform", "manylinux2014_x86_64", "--python-version", "313",
        "--implementation", "cp", "--abi", "cp313", "--only-binary=:all:",
        f"{pkg}=={version}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Some pure-python packages have no platform-specific build; retry unrestricted.
        cmd_fallback = [
            sys.executable, "-m", "pip", "download", "--no-deps",
            "-d", str(WHEEL_CACHE_DIR), f"{pkg}=={version}",
        ]
        result = subprocess.run(cmd_fallback, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [era_wheels] FAILED to download {pkg}=={version}: {result.stderr[-500:]}", file=sys.stderr)
            return None

    after = set(WHEEL_CACHE_DIR.glob("*.whl"))
    new_files = after - before
    return next(iter(new_files), None)


PATCHED_SNAPSHOTS_DIR = PROJECT_ROOT / "data_extra" / "patched_snapshots"

_SETUPTOOLS_SCM_STUB_TEMPLATE = (
    "# stub generated locally -- the original snapshot has no git tag history for\n"
    "# setuptools_scm to compute a real version from, and this dataset didn't ship\n"
    "# a pre-generated one. Exact version string doesn't matter for test purposes.\n"
    "__version__ = version = '0.0.0.dev0+unknown'\n"
    "__version_tuple__ = version_tuple = (0, 0, 0, 'dev0', 'unknown')\n"
)


def ensure_setuptools_scm_stub(instance_id: str, snapshot_tgz: Path) -> Path:
    """If a snapshot's pyproject.toml declares [tool.setuptools_scm] write_to = "<path>"
    but that file isn't actually present in the archive (common for these single-commit,
    no-tag-history snapshots), return a patched copy with a stub version file added so the
    editable install doesn't fail with ModuleNotFoundError for it. Returns the original
    path unchanged if no such gap exists.
    """
    PATCHED_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    patched_path = PATCHED_SNAPSHOTS_DIR / snapshot_tgz.name
    if patched_path.exists():
        return patched_path

    with tarfile.open(snapshot_tgz, "r:gz") as tar:
        names = set(tar.getnames())
        pyproject_name = next((n for n in names if n.endswith("pyproject.toml") and n.count("/") <= 1), None)
        if not pyproject_name:
            return snapshot_tgz
        try:
            pyproject = tomllib.load(tar.extractfile(pyproject_name))
        except Exception:
            return snapshot_tgz

        write_to = (pyproject.get("tool") or {}).get("setuptools_scm", {}).get("write_to")
        if not write_to:
            return snapshot_tgz

        prefix = pyproject_name.rsplit("pyproject.toml", 1)[0]
        target_name = prefix + write_to
        if target_name in names or write_to in names:
            return snapshot_tgz  # already present, nothing to patch

        members = tar.getmembers()
        member_data = {m.name: tar.extractfile(m).read() for m in members if m.isfile()}

    stub_bytes = _SETUPTOOLS_SCM_STUB_TEMPLATE.encode("utf-8")
    with tarfile.open(patched_path, "w:gz") as out:
        for m in members:
            if m.isfile():
                data = member_data[m.name]
                info = tarfile.TarInfo(m.name)
                info.size = len(data)
                info.mode = m.mode
                out.addfile(info, io.BytesIO(data))
            else:
                out.addfile(m)
        info = tarfile.TarInfo(target_name)
        info.size = len(stub_bytes)
        out.addfile(info, io.BytesIO(stub_bytes))

    print(f"  [era_wheels] {instance_id}: patched missing setuptools_scm version file ({target_name})")
    return patched_path


def ensure_pytest_ini_addopts_preserved(instance_id: str, snapshot_tgz: Path) -> Path:
    """swegemma's setup_workspace_test_config() writes its own generic pytest.ini UNLESS
    one already exists containing a "[pytest]" marker -- but many modern projects (pytest
    itself included) configure pytest entirely via pyproject.toml's [tool.pytest.ini_options],
    which then gets silently ignored once any pytest.ini exists (standard pytest config
    precedence). If a project's `addopts` there enables something load-bearing (pytest's own
    tests need `-p pytester`, an opt-in built-in plugin), losing it breaks the whole suite.
    Pre-inject a minimal pytest.ini preserving just `addopts` so the harness's "already
    configured" check finds it and leaves it alone.
    """
    patched_path = PATCHED_SNAPSHOTS_DIR / snapshot_tgz.name
    source = patched_path if patched_path.exists() else snapshot_tgz

    with tarfile.open(source, "r:gz") as tar:
        names = set(tar.getnames())
        pyproject_name = next((n for n in names if n.endswith("pyproject.toml") and n.count("/") <= 1), None)
        if not pyproject_name:
            return source
        prefix = pyproject_name.rsplit("pyproject.toml", 1)[0]

        # The harness's own check (setup_workspace_test_config) looks ONLY for a literal
        # <workspace_root>/pytest.ini -- tox.ini, setup.cfg's [tool:pytest], or a pytest.ini
        # nested under docs/examples do NOT stop it from overwriting, even though pytest
        # itself would normally honor some of those.
        if (prefix + "pytest.ini") in names:
            return source  # already present at workspace root; harness will detect+preserve it

        try:
            pyproject = tomllib.load(tar.extractfile(pyproject_name))
        except Exception:
            return source

        addopts = ((pyproject.get("tool") or {}).get("pytest") or {}).get("ini_options", {}).get("addopts")
        if not addopts:
            return source

        members = tar.getmembers()
        member_data = {m.name: tar.extractfile(m).read() for m in members if m.isfile()}

    ini_bytes = f"[pytest]\naddopts = {addopts}\n".encode("utf-8")
    tmp_path = patched_path.with_suffix(".tmp.tgz")
    with tarfile.open(tmp_path, "w:gz") as out:
        for m in members:
            if m.isfile():
                data = member_data[m.name]
                info = tarfile.TarInfo(m.name)
                info.size = len(data)
                info.mode = m.mode
                out.addfile(info, io.BytesIO(data))
            else:
                out.addfile(m)
        info = tarfile.TarInfo(prefix + "pytest.ini")
        info.size = len(ini_bytes)
        out.addfile(info, io.BytesIO(ini_bytes))
    tmp_path.replace(patched_path)

    print(f"  [era_wheels] {instance_id}: preserved pyproject.toml addopts ({addopts!r}) into a pytest.ini stub")
    return patched_path


def prepare_task_wheels_dir(task, extra_pkgs: list[str], safe_base_dir: Path, work_dir: Path) -> Path:
    """Build (or reuse) a per-task wheels directory: every wheel in safe_base_dir except
    ones matching the task's own repo package, plus era-pinned wheels for extra_pkgs.
    """
    repo_short = _normalize_pkg(task.repo.rsplit("/", 1)[-1])
    cutoff = _parse_cutoff(task)
    key = f"{repo_short}__{cutoff.date().isoformat()}__{'-'.join(sorted(extra_pkgs))}" or "base"
    task_dir = work_dir / "task_wheels" / key
    marker = task_dir / ".ready"
    if marker.exists():
        return task_dir
    task_dir.mkdir(parents=True, exist_ok=True)

    for whl in safe_base_dir.glob("*.whl"):
        pkg_name = _normalize_pkg(whl.name.split("-", 1)[0])
        if pkg_name == repo_short:
            continue  # never shadow the task's own editable-installed source
        dest = task_dir / whl.name
        if not dest.exists():
            dest.write_bytes(whl.read_bytes())

    for pkg in extra_pkgs:
        if _normalize_pkg(pkg) == repo_short:
            continue
        version = latest_version_before(pkg, cutoff)
        if not version:
            print(f"  [era_wheels] no PyPI release of {pkg} found before {cutoff.date()}, skipping", file=sys.stderr)
            continue
        src = download_pinned_wheel(pkg, version)
        if src:
            dest = task_dir / src.name
            if not dest.exists():
                dest.write_bytes(src.read_bytes())
            print(f"  [era_wheels] {task.instance_id}: pinned {pkg}=={version}")

    marker.write_text("ok", encoding="utf-8")
    return task_dir
