"""Profile nebius/SWE-rebench-openhands-trajectories for post-training use (reads the parquet by row group, never executes anything from it).

For every trajectory: resolved flag, number of assistant turns, total characters (tokens ~ chars / 3.3, the ratio measured on our own Gemma captures), the tool
names used, and the largest single tool output. Prints how many RESOLVED trajectories survive each (turns, tokens) cut-off, which repos they cover, and writes a
compact per-trajectory table to <out>/profile.jsonl for the converter / selector.

Usage: python scripts/g7/profile_nebius.py <parquet> <out_dir>
"""

import collections
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
f = pq.ParquetFile(src)
rows = []
tool_names = collections.Counter()
for rg in range(f.metadata.num_row_groups):
    t = f.read_row_group(rg, columns=["trajectory_id", "instance_id", "repo", "trajectory", "model_patch", "exit_status", "resolved"]).to_pylist()
    for r in t:
        msgs = r["trajectory"]
        chars = sum(len(m.get("content") or "") for m in msgs) + sum(len(tc["function"]["arguments"] or "") for m in msgs for tc in (m.get("tool_calls") or []))
        asst = [m for m in msgs if m["role"] == "assistant"]
        names = collections.Counter(tc["function"]["name"] for m in asst for tc in (m.get("tool_calls") or []))
        tool_names.update(names)
        biggest_tool = max((len(m.get("content") or "") for m in msgs if m["role"] == "tool"), default=0)
        rows.append({"trajectory_id": r["trajectory_id"], "instance_id": r["instance_id"], "repo": r["repo"], "resolved": r["resolved"], "exit_status": r["exit_status"],
                     "n_msgs": len(msgs), "n_assistant": len(asst), "chars": chars, "est_tokens": round(chars / 3.3), "biggest_tool_chars": biggest_tool,
                     "patch_chars": len(r["model_patch"] or ""), "tools": dict(names)})
    print(f"row group {rg + 1}/{f.metadata.num_row_groups}: {len(rows)} rows", flush=True)
with open(out / "profile.jsonl", "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

res = [r for r in rows if r["resolved"] == 1]
print(f"\ntrajectories {len(rows)} | resolved {len(res)} | exit_status: {dict(collections.Counter(r['exit_status'] for r in rows).most_common(6))}")
print("tool usage (all):", dict(tool_names.most_common(10)))


def q(vals, p):
    vals = sorted(vals)
    return vals[int(p * (len(vals) - 1))]


for name, key in (("assistant turns", "n_assistant"), ("estimated tokens", "est_tokens"), ("biggest tool output (chars)", "biggest_tool_chars")):
    v = [r[key] for r in res]
    print(f"resolved {name}: p10 {q(v, .1)} p25 {q(v, .25)} p50 {q(v, .5)} p75 {q(v, .75)} p90 {q(v, .9)} min {min(v)}")
print("\nresolved trajectories surviving cut-offs (assistant turns <= T and estimated tokens <= N):")
for T in (15, 20, 25, 30, 40):
    print("  turns <= %2d: " % T + " | ".join(f"<= {N // 1000}k tok: {sum(1 for r in res if r['n_assistant'] <= T and r['est_tokens'] <= N):6d}" for N in (8000, 12000, 16000)))
short = [r for r in res if r["n_assistant"] <= 25 and r["est_tokens"] <= 16000]
print(f"\nshort resolved (<= 25 turns, <= 16k tokens): {len(short)} trajectories, {len({r['instance_id'] for r in short})} distinct issues, {len({r['repo'] for r in short})} repos")
print("top repos:", dict(collections.Counter(r["repo"] for r in short).most_common(12)))
print("tool names in short resolved:", dict(sum((collections.Counter(r["tools"]) for r in short), collections.Counter()).most_common(8)))
