#!/usr/bin/env bash
# The competition's public data/wheels/ offline dependency cache is missing at least
# one real dependency: `typing_inspection`, a transitive dependency of `pydantic` that
# fastapi requires just to import itself. Without it, ~66 of the 67 public fastapi
# tasks fail at pytest collection time regardless of patch correctness (see
# docs/STRATEGY.md for the full investigation, including two OTHER small packages --
# inline-snapshot and dirty-equals -- that were tried and reverted because they
# register pytest plugins that auto-load for every container's tests, not just
# fastapi's, breaking unrelated `rich` tasks).
#
# This only affects OUR OWN local dev/training loop on the 129 public tasks -- the
# real Kaggle scorer uses a separate, presumably correctly-provisioned wheel bundle
# for the hidden private-repo test set.
#
# Usage: run once after (re-)downloading the competition dataset, before building
# swebench-sandbox:latest or running scripts/check_gold_patches.py.
set -euo pipefail
cd "$(dirname "$0")/.."

pip download --no-deps -d data/wheels typing_inspection

# The harness caches unpacked wheels host-side, keyed by a content-independent name --
# it will NOT notice data/wheels/ changed unless this is cleared.
rm -rf "${TEMP:-/tmp}/swegemma_sp_cache_v8"

echo "Wheel added. Now rebuild the sandbox image:"
echo "  cd data && docker build -f docker/Dockerfile.public -t swebench-sandbox:latest ."
