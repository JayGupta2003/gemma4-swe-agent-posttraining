"""Build hindsight-pruned teacher episodes (scripts/g18/teacher_convert.py) from the Nebius OpenHands trajectories.

Usage: python scripts/g18/build_teacher_episodes.py <trajectories.parquet> <out_dir> [per_repo_cap=6]

Filters: resolved trajectories only; the competition's own repos and instance ids (data/tasks.jsonl) are EXCLUDED so nothing the leaderboard is built from is trained on;
one episode per issue (the one with the fewest estimated tokens); at most `per_repo_cap` issues per repo so no project dominates. Writes <out_dir>/episodes.jsonl
(messages + metadata, never executed) and <out_dir>/stats.json (why trajectories were dropped, size and repo distribution).
"""

import collections
import json
import statistics as st
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teacher_convert as TC  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
src, out = Path(sys.argv[1]), Path(sys.argv[2])
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 6
out.mkdir(parents=True, exist_ok=True)
tasks = [json.loads(l) for l in open(ROOT / "data" / "tasks.jsonl", encoding="utf-8")]
ban_repos = {t["repo"].lower() for t in tasks}
ban_ids = {t["instance_id"] for t in tasks}
f = pq.ParquetFile(src)
reasons = collections.Counter()
best = {}
seen = 0
for rg in range(f.metadata.num_row_groups):
    for r in f.read_row_group(rg, columns=["trajectory_id", "instance_id", "repo", "trajectory", "model_patch", "resolved"]).to_pylist():
        seen += 1
        if r["resolved"] != 1:
            reasons["unresolved"] += 1
            continue
        if r["repo"].lower() in ban_repos or r["instance_id"] in ban_ids:
            reasons["competition_repo"] += 1
            continue
        ep, why = TC.convert(r)
        if ep is None:
            reasons[why] += 1
            continue
        reasons["ok"] += 1
        k = r["instance_id"]
        if k not in best or ep["est_tokens"] < best[k]["est_tokens"]:
            best[k] = ep
    print(f"row group {rg + 1}/{f.metadata.num_row_groups}: seen {seen}, ok {reasons['ok']}, issues {len(best)}", flush=True)

per_repo = collections.defaultdict(list)
for ep in sorted(best.values(), key=lambda e: e["est_tokens"]):
    per_repo[ep["repo"]].append(ep)
kept = [e for lst in per_repo.values() for e in lst[:cap]]
kept.sort(key=lambda e: (e["repo"], e["instance_id"]))
with open(out / "episodes.jsonl", "w", encoding="utf-8") as fh:
    for e in kept:
        fh.write(json.dumps(e) + "\n")
toks = [e["est_tokens"] for e in kept]
stats = {"seen": seen, "drop_reasons": dict(reasons.most_common()), "issues_with_episode": len(best), "episodes_kept": len(kept), "repos": len(per_repo), "per_repo_cap": cap,
         "est_tokens": {"p10": sorted(toks)[len(toks) // 10], "median": st.median(toks), "p90": sorted(toks)[9 * len(toks) // 10], "sum_m": round(sum(toks) / 1e6, 2)},
         "calls": dict(collections.Counter(e["n_calls"] for e in kept).most_common()), "banned_repos": len(ban_repos)}
(out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
print(json.dumps(stats, indent=1))
