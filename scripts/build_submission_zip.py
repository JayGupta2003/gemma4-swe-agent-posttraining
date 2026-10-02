"""Build submission.zip from submission/ with POSIX entry names, then verify the RAW bytes.

Why this exists: PowerShell's Compress-Archive stores Windows backslash paths
(`configs\\sampling.yaml`). On the Linux scorer those extract as flat files with
backslashes in their names, so `!include configs/sampling.yaml` cannot resolve and the
run dies with "unhandled error" (submissions 56717963, 56752702, 56780281 all failed
this way; the only zip built with forward slashes, 56682704, was the only one that ran).
Python's zipfile.namelist() CANNOT be used to verify on Windows -- ZipInfo rewrites
os.sep to '/' on read -- so verification here parses the central directory directly.

Usage: python scripts/build_submission_zip.py [output_path]
"""

from __future__ import annotations

import struct
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "submission"
DEFAULT_OUT = PROJECT_ROOT / "submission_drafts" / "zips" / "submission.zip"
ALLOWED_EXTS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}


def build(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    files = sorted(p for p in SRC.rglob("*") if p.is_file())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            name = p.relative_to(SRC).as_posix()
            assert p.suffix.lower() in ALLOWED_EXTS, f"disallowed extension: {name}"
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())


def raw_central_directory_names(path: Path) -> list[str]:
    """Entry names exactly as stored (what a Linux extractor sees), no normalization."""
    data = path.read_bytes()
    names, pos = [], 0
    while True:
        pos = data.find(b"PK\x01\x02", pos)
        if pos < 0:
            return names
        name_len = struct.unpack_from("<H", data, pos + 28)[0]
        names.append(data[pos + 46 : pos + 46 + name_len].decode("utf-8"))
        pos += 46 + name_len


def verify(path: Path) -> None:
    raw = raw_central_directory_names(path)
    print("raw stored entry names:", raw)
    assert raw, "no entries found"
    assert not any("\\" in n for n in raw), f"BACKSLASH in stored names: {raw}"
    assert not any(n.endswith("/") for n in raw), "directory entries present"
    assert "agent.yaml" in raw, "agent.yaml must be at the archive root"
    expected = sorted(p.relative_to(SRC).as_posix() for p in SRC.rglob("*") if p.is_file())
    assert sorted(raw) == expected, f"file set mismatch: {sorted(raw)} vs {expected}"
    with zipfile.ZipFile(path) as z:
        for n in raw:
            assert z.read(n) == (SRC / n).read_bytes(), f"content mismatch: {n}"
    print(f"OK: {len(raw)} files, POSIX names, byte-identical to submission/")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    build(out)
    verify(out)
    print("wrote", out)
