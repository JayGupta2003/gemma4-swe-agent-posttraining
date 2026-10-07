"""Submit ONE exact, pre-verified zip. Rebuilds from the source dir, refuses unless the sha256 prefix matches the Linux-verified artifact,
refuses if no slot is open, submits once, then confirms the allowance was consumed.

Usage: python scripts/submit_pinned_zip.py <source_dir> <expected_sha256_prefix> "<message>"
"""

import hashlib
import subprocess
import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent.parent
COMPETITION = "gemma-4-developer-agent"
src, pin, message = sys.argv[1], sys.argv[2], sys.argv[3]
# The competition requires the archive to be NAMED submission.zip (a differently named file gets HTTP 400), so use a per-source folder.
zip_path = ROOT / "submission_drafts" / "zips" / f"pinned_{Path(src).name}" / "submission.zip"

subprocess.run([sys.executable, str(ROOT / "scripts" / "build_submission_zip.py"), str(zip_path), src], check=True)
sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
print("zip sha256:", sha[:16], "| pinned:", pin)
if not sha.startswith(pin):
    sys.exit("REFUSING TO SUBMIT: zip differs from the Linux-verified artifact")

api = KaggleApi()
api.authenticate()
lim = api.competition_get_submission_limits(COMPETITION)
print("limits before:", lim)
import json
_l = json.loads(str(lim))
if _l.get("numAllowedNow", 0) < 1:
    sys.exit("REFUSING TO SUBMIT: no slot open")
before = {s.ref for s in api.competition_submissions(COMPETITION)}
try:
    res = api.competition_submit(str(zip_path), message, COMPETITION)
except Exception as e:  # show the server's reason instead of a bare traceback
    body = getattr(getattr(e, "response", None), "text", "")
    sys.exit(f"SUBMIT FAILED: {type(e).__name__}: {e} | body: {body[:500]}")
print("submit result:", res)
time.sleep(8)
after = api.competition_submissions(COMPETITION)
new = [s for s in after if s.ref not in before]
print("limits after:", api.competition_get_submission_limits(COMPETITION))
print("new submission(s):", [(s.ref, str(s.status), s.public_score) for s in new])
