"""Local tests of scripts/g25/fresh_env.py (no swegemma needed: a fake module stands in for swegemma.harness.container_setup)."""
import subprocess
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fresh_env as fe  # noqa: E402

# 1. lock parsing: extras, markers, comments, options, normalisation, SKIP is applied by build_store (not by parse_lock)
lock = "# comment\n-e .\nFoo_Bar[extra]==1.2.3 ; python_version >= '3.9'\npytest==9.0.2\nbaz==0.1 \\\n    --hash=sha256:abc\n\nqux>=2\n"
assert fe.parse_lock(lock) == [("foo-bar", "1.2.3"), ("pytest", "9.0.2"), ("baz", "0.1")], fe.parse_lock(lock)
assert fe.store_dir_name("Foo_Bar", "1.2.3") == "foo-bar-1.2.3"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # 2. base commit from the snapshot convention, with a harness baseline commit on top
    ws = td / "sb" / "workspace"; (ws / "src" / "pkg").mkdir(parents=True); (ws / ".hidden").mkdir(); (ws / "a.py").write_text("x=1\n")
    sha = "0123456789abcdef0123456789abcdef01234567"
    g = ["git", "-C", str(ws), "-c", "user.email=a@b", "-c", "user.name=a"]
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    subprocess.run(g + ["add", "-A"], check=True); subprocess.run(g + ["commit", "-q", "-m", f"base {sha}"], check=True)
    subprocess.run(g + ["commit", "-q", "--allow-empty", "-m", "baseline"], check=True)
    assert fe.base_commit_of(ws) == sha, fe.base_commit_of(ws)
    assert fe.base_commit_of(td) is None

    # 3. .pth content and placement (every venv site-packages)
    venv = td / "sb" / "venv"; sp = venv / "lib" / "python3.12" / "site-packages"; sp.mkdir(parents=True)
    store = td / "store"
    paths = {"workspace": ws, "venv": venv}
    assert fe.write_env_pth(paths, store, {}) == f"no-env:{sha[:10]}"
    st = fe.write_env_pth(paths, store, {sha: ["foo-bar-1.2.3", "baz-0.1"]})
    assert st == f"ok:{sha[:10]}:2:1", st
    lines = (sp / fe.PTH_NAME).read_text().splitlines()
    assert lines == [str(ws), str(ws / "src"), str(store / "foo-bar-1.2.3" / "site-packages"), str(store / "baz-0.1" / "site-packages")], lines
    assert fe.PTH_NAME < "_host_env.pth"   # processed first -> earlier on sys.path than the host entries

    # 4. hook: patches every module that imported the function, calls the original, idempotent on re-install
    calls = []
    def orig(docker, cid, repo="", **kw): calls.append((cid, repo, kw))
    cs = types.ModuleType("swegemma.harness.container_setup"); cs.install_test_dependencies = orig
    ver = types.ModuleType("swegemma.harness.verification"); ver.install_test_dependencies = orig
    other = types.ModuleType("swegemma.other"); other.install_test_dependencies = lambda *a: None
    for m in (types.ModuleType("swegemma"), types.ModuleType("swegemma.harness"), cs, ver, other): sys.modules[m.__name__] = m
    sys.modules["swegemma.harness"].container_setup = cs
    class SubprocessManager:
        sandboxes = {"s1": paths}
    (sp / fe.PTH_NAME).unlink()
    patched = fe.install_hook(store, {sha: ["baz-0.1"]})
    assert sorted(patched) == ["swegemma.harness.container_setup", "swegemma.harness.verification"], patched
    ver.install_test_dependencies(SubprocessManager(), "s1", "o/r", config=None)
    assert calls == [("s1", "o/r", {"config": None})] and (sp / fe.PTH_NAME).exists() and fe.install_hook.stats == {"ok": 1}
    assert sorted(fe.install_hook(store, {sha: ["baz-0.1"]})) == sorted(patched)        # re-install wraps the ORIGINAL, not the hook
    ver.install_test_dependencies(SubprocessManager(), "s1", "o/r"); assert len(calls) == 2
    class DockerManager: sandboxes = {}
    ver.install_test_dependencies(DockerManager(), "s1"); assert len(calls) == 3        # other sandboxes untouched

    # 5. editable-install emulation: hatch-vcs version file (src layout, path relative to the root), dist-info, static version, setup.cfg name
    p1 = td / "p1"; (p1 / "src" / "platformdirs").mkdir(parents=True)
    (p1 / "pyproject.toml").write_text('[project]\nname = "platformdirs"\ndynamic = ["version"]\n[tool.hatch.build.hooks.vcs]\nversion-file = "src/platformdirs/version.py"\n')
    sp1 = td / "v1"; sp1.mkdir()
    assert fe.materialize_project(p1, [sp1]) == "name=platformdirs,ver=99.0.0,vfiles=1"
    ns = {}; exec((p1 / "src" / "platformdirs" / "version.py").read_text(), ns); assert ns["__version__"] == "99.0.0" and ns["version_tuple"] == (99, 0, 0)
    assert "Name: platformdirs" in (sp1 / "platformdirs-99.0.0.dist-info" / "METADATA").read_text()
    (p1 / "src" / "platformdirs" / "version.py").write_text("KEEP\n"); fe.materialize_project(p1, [sp1])
    assert (p1 / "src" / "platformdirs" / "version.py").read_text() == "KEEP\n"           # never overwrites
    p2 = td / "p2"; (p2 / "humanize").mkdir(parents=True)
    (p2 / "pyproject.toml").write_text('[project]\nname = "humanize"\nversion = "4.1.0"\n[tool.setuptools_scm]\nwrite_to = "humanize/_version.py"\n')
    assert fe.materialize_project(p2, [sp1]) == "name=humanize,ver=4.1.0,vfiles=1" and (sp1 / "humanize-4.1.0.dist-info").is_dir()
    p3 = td / "p3"; p3.mkdir(); (p3 / "setup.cfg").write_text("[metadata]\nname = python-multipart\n")
    assert fe.materialize_project(p3, [sp1]) == "name=python-multipart,ver=99.0.0,vfiles=0" and (sp1 / "python_multipart-99.0.0.dist-info").is_dir()
    import importlib.metadata as im
    sys.path.insert(0, str(sp1))
    assert im.version("humanize") == "4.1.0" and im.version("platformdirs") == "99.0.0"   # first on sys.path wins over any installed copy
    sys.path.remove(str(sp1))

    # 6. one real pin through pip --target (PyPI binary path; skipped offline)
    r = fe.install_pin("six", "1.16.0", store, None)
    if r["source"] is not None:
        assert (store / "six-1.16.0" / "site-packages" / "six.py").exists() and fe.install_pin("six", "1.16.0", store, None) == {"source": "cached"}
    print("pip pin:", r["source"])
print("test_fresh_env: OK")
