"""Poll the competition's daily submission allowance and submit ONE pinned, verified zip when a slot opens.

Safety gates (the whole point -- slots are one per day):
  1. Rebuilds the zip from submission/ with scripts/build_submission_zip.py, which refuses
     backslash entry names / directory entries / any content drift from submission/.
  2. The rebuilt zip's sha256 must EQUAL EXPECTED_SHA256 -- the exact artifact that was
     compiled successfully on Linux in the zip-artifact test notebook. If submission/ was
     edited since, the hashes differ and this script refuses to submit (and says so).
  3. Submits at most once. After an attempt it re-reads the allowance and stops; it only
     retries if the allowance is still unused (i.e. the upload itself failed).

Usage: python scripts/auto_submit_when_open.py [--dry-run] [--poll-seconds 300] [--max-hours 30]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPETITION = "gemma-4-developer-agent"
ZIP_PATH = PROJECT_ROOT / "submission_drafts" / "zips" / "submission.zip"
EXPECTED_SHA256_PREFIX = "ae4afbb1493b7415"  # zip verified to compile on Linux (zip_test notebook)
MESSAGE = (
    "Same validated scaffold (no search_similar_code, pytest output tail fix, async graph warning, "
    "skip_summarization=false, max_output_tokens 8192, max_time_minutes 5) -- now packaged with "
    "POSIX zip entry names. Earlier PowerShell Compress-Archive zips stored backslash paths, which "
    "breaks !include on the Linux scorer."
)


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z] {msg}", flush=True)


def allowance(api: KaggleApi) -> tuple[int, int]:
    lim = api.competition_get_submission_limits(COMPETITION)
    return lim.num_allowed_now, lim.num_today


def rebuild_and_gate() -> None:
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_submission_zip.py")],
                       capture_output=True, text=True)
    log("build_submission_zip: " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:]))
    if r.returncode != 0:
        raise SystemExit(f"REFUSING TO SUBMIT: zip build/verification failed:\n{r.stdout}\n{r.stderr}")
    sha = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
    if not sha.startswith(EXPECTED_SHA256_PREFIX):
        raise SystemExit(
            f"REFUSING TO SUBMIT: zip sha256 {sha[:16]} != pinned {EXPECTED_SHA256_PREFIX} "
            "(submission/ changed since the Linux-verified build). Re-verify, then update EXPECTED_SHA256_PREFIX."
        )
    log(f"gate passed: sha256 {sha[:16]} matches the Linux-verified artifact")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=30)
    args = ap.parse_args()

    api = KaggleApi()
    api.authenticate()
    deadline = time.time() + args.max_hours * 3600
    log(f"watching {COMPETITION} (dry_run={args.dry_run}, poll={args.poll_seconds}s, max={args.max_hours}h)")

    while time.time() < deadline:
        try:
            allowed, today = allowance(api)
        except Exception as e:  # transient API/network error: keep polling
            log(f"allowance check failed ({type(e).__name__}: {e}); will retry")
            time.sleep(args.poll_seconds)
            continue
        log(f"allowed_now={allowed} num_today={today}")
        if allowed >= 1:
            rebuild_and_gate()
            if args.dry_run:
                log("DRY RUN: would submit now. Stopping.")
                return
            for attempt in range(1, 4):
                try:
                    log(f"submitting (attempt {attempt}) ...")
                    resp = api.competition_submit(str(ZIP_PATH), MESSAGE, COMPETITION)
                    log(f"submit call returned: {resp}")
                except Exception as e:
                    log(f"submit raised {type(e).__name__}: {e}")
                time.sleep(15)
                allowed_after, today_after = allowance(api)
                log(f"after attempt: allowed_now={allowed_after} num_today={today_after}")
                if allowed_after == 0:
                    log("allowance consumed -> submission registered. DONE (will not submit again).")
                    return
                time.sleep(300)
            raise SystemExit("allowance still unused after 3 attempts; stopping so a human can look.")
        time.sleep(args.poll_seconds)
    log("max-hours reached without a free slot; exiting without submitting.")


if __name__ == "__main__":
    main()
