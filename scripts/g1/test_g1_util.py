"""Local tests for g1_util: parallel staging correctness + abort paths, supervisor milestone logic."""

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import g1_util as U  # noqa: E402

tmp = Path(tempfile.mkdtemp())
src, dst = tmp / "src", tmp / "dst"
src.mkdir()
(src / "config.json").write_text('{"a": 1}')
(src / "tok.json").write_bytes(os.urandom(10_000))
(src / "model-00001.safetensors").write_bytes(os.urandom(200 * 1024 * 1024 + 12345))   # not a multiple of the block size
(src / "model-00002.safetensors").write_bytes(os.urandom(70 * 1024 * 1024))
r = U.stage_checkpoint(src, dst, workers=6, block=16 << 20, log=lambda *a: None)
print("staged:", r)
for f in src.iterdir():
    assert hashlib.sha256(f.read_bytes()).hexdigest() == hashlib.sha256((dst / f.name).read_bytes()).hexdigest(), f.name
print("all files byte-identical")
try:
    U.stage_checkpoint(src, tmp / "dst2", workers=1, block=16 << 20, min_mbps_after_s=0.0, min_mbps=1e9, log=lambda *a: None)
    raise SystemExit("expected StageAbort")
except U.StageAbort as e:
    print("abort path OK:", e)

py = sys.executable
ok = U.run_supervised([py, "-c", "import time;print('[g1] loaded in 1s',flush=True);time.sleep(1.5);print('[g1] micro 0',flush=True);time.sleep(1)"],
                      dict(os.environ), tmp / "a.log", [("[g1] loaded in", 6), ("[g1] micro 0", 8)], max_seconds=30, poll=0.5)
print("supervised ok:", ok); assert ok["returncode"] == 0 and len(ok["milestones_reached"]) == 2
bad = U.run_supervised([py, "-c", "import time;print('hello',flush=True);time.sleep(60)"], dict(os.environ), tmp / "b.log",
                       [("[g1] loaded in", 3)], max_seconds=30, poll=0.5)
print("supervised stalled:", bad); assert bad["verdict"].startswith("killed") and bad["seconds"] < 10
shutil.rmtree(tmp, ignore_errors=True)
print("UTIL TESTS OK")
