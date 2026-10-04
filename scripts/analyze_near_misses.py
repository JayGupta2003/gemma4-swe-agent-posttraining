"""Compare agent patches to the reference (gold) patches: localization vs fix-quality failures.

For every attempt that produced a non-empty patch, classify it against the gold patch by
  - file overlap  (same files / agent subset / agent superset / partial / disjoint)
  - hunk overlap  (do the agent's edited line ranges intersect the gold's edited ranges, in the same file?)
  - whether the agent touched test files (a rule violation that can break verification)
and report the rates for resolved vs not-resolved attempts.

Usage: python scripts/analyze_near_misses.py <rollouts_dir containing round_*/{task_results.jsonl,patches/}>
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_patch(patch: str) -> dict[str, list[tuple[int, int]]]:
    """file -> list of (old_start, old_end) edited line ranges (new-file path used as key)."""
    files: dict[str, list[tuple[int, int]]] = {}
    cur = None
    for line in patch.splitlines():
        m = re.match(r"^diff --git a/(.*?) b/(.*)$", line)
        if m:
            cur = m.group(2).strip().strip('"')
            files.setdefault(cur, [])
            continue
        # The dataset's gold patches are plain unified diffs (no `diff --git` header line), so
        # also key off the `+++ b/<path>` line; agent patches carry both forms.
        m = re.match(r"^\+\+\+ (?:b/)?(\S+)", line)
        if m and m.group(1) != "/dev/null":
            cur = m.group(1).strip('"')
            files.setdefault(cur, [])
            continue
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m and cur is not None:
            start = int(m.group(1))
            length = int(m.group(2)) if m.group(2) is not None else 1
            files[cur].append((start, start + max(length, 1) - 1))
    return files


def is_test_path(p: str) -> bool:
    name = p.rsplit("/", 1)[-1]
    parts = {x.lower() for x in p.split("/")[:-1]}
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py" or bool({"tests", "test", "testing"} & parts)


def classify(agent: dict, gold: dict) -> tuple[str, bool]:
    a = {f for f in agent if not is_test_path(f)}
    g = {f for f in gold if not is_test_path(f)}
    if not a:
        file_cls = "agent edited only tests/none"
    elif a == g:
        file_cls = "same files"
    elif a < g:
        file_cls = "agent subset of gold files"
    elif a > g:
        file_cls = "agent superset (gold files + extra)"
    elif a & g:
        file_cls = "partial file overlap"
    else:
        file_cls = "DISJOINT files (wrong place)"
    hunk = False
    for f in a & g:
        for (s1, e1) in agent[f]:
            for (s2, e2) in gold[f]:
                if s1 <= e2 + 3 and s2 <= e1 + 3:  # within 3 lines counts as the same region
                    hunk = True
    return file_cls, hunk


def main(rollouts: Path) -> None:
    gold = {}
    for line in (ROOT / "data" / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        gold[t["instance_id"]] = parse_patch(t.get("patch", ""))
    groups = {"resolved": [], "not_resolved": []}
    for rd in sorted(rollouts.glob("round_*")):
        for line in (rd / "task_results.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            pf = rd / "patches" / f"{r['instance_id']}.patch"
            if not pf.exists():
                continue  # no patch produced
            agent = parse_patch(pf.read_text(encoding="utf-8"))
            file_cls, hunk = classify(agent, gold.get(r["instance_id"], {}))
            touched_tests = any(is_test_path(f) for f in agent)
            groups["resolved" if r["resolved"] else "not_resolved"].append((r["instance_id"], file_cls, hunk, touched_tests))
    for name, items in groups.items():
        n = len(items)
        print(f"\n== {name}: {n} attempts that produced a patch ==")
        if not n:
            continue
        for k, v in collections.Counter(c for _, c, _, _ in items).most_common():
            print(f"   {k:38s} {v:3d} ({100*v/n:4.0f}%)")
        print(f"   edited the SAME REGION as the gold patch (+-3 lines): {sum(h for _,_,h,_ in items):3d} ({100*sum(h for _,_,h,_ in items)/n:4.0f}%)")
        print(f"   touched a test file (rule violation):                {sum(t for *_,t in items):3d} ({100*sum(t for *_,t in items)/n:4.0f}%)")
    nr = groups["not_resolved"]
    right = [i for i, c, h, _ in nr if h]
    print(f"\nNot-resolved patches that hit the gold region (candidate 'near misses' = right place, wrong fix): {len(right)}/{len(nr)}")
    print("  ", sorted(set(right))[:40])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
