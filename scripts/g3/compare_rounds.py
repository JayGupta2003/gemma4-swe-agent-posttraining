"""Compare the G3 adapter round (e1) with the two G2 base rounds (r1, r2): outcomes per task, effort per task, and how the model behaves turn by turn.

Everything here is computed from files already downloaded (task_results.jsonl of each round + the capture of every LLM request), so it costs no GPU.

Usage: python scripts/g3/compare_rounds.py <g2_dir containing capture.jsonl.gz + r1 + r2> <g3_dir containing capture.jsonl.gz + e1>
"""

import collections
import gzip
import json
import statistics as st
import sys
from pathlib import Path

g2, g3 = Path(sys.argv[1]), Path(sys.argv[2])
MAIN = "You are an expert autonomous software engineer"
ROUNDS = {"r1": g2, "r2": g2, "e1": g3}


def msg_text(m):
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c)


res = {rd: {r["instance_id"]: r for r in map(json.loads, open(d / rd / "task_results.jsonl", encoding="utf-8"))} for rd, d in ROUNDS.items()}
tasks = sorted(res["e1"])
solved = {rd: {t for t, r in res[rd].items() if r["resolved"]} for rd in res}
union_base = solved["r1"] | solved["r2"]

print("== outcomes ==")
for rd in res:
    rr = list(res[rd].values())
    cap = sum("timeout" in (r.get("error") or "") for r in rr)
    patch = sum(r.get("agent_patch_size", 0) > 0 for r in rr)
    print(f"{rd}: resolved {len(solved[rd])}/75 | patch written {patch} | session-timeout {cap} | mean duration {st.mean(r['duration_seconds'] for r in rr):.0f}s | "
          f"mean llm calls {st.mean(r['total_llm_calls'] for r in rr):.1f} | mean tool calls {st.mean(r['tool_calls'] for r in rr):.1f}")
print("\ntask flips (S = solved, . = not):")
pat = collections.defaultdict(list)
for t in tasks:
    pat["".join("S" if t in solved[rd] else "." for rd in ("r1", "r2", "e1"))].append(t)
for p, ts in sorted(pat.items(), key=lambda x: -len(x[1])):
    print(f"  r1 r2 e1 = {p}: {len(ts):3d}", "" if len(ts) > 8 else ts)

# does the adapter work *faster* on tasks solved in the same way? compare llm calls and duration on tasks solved in all three rounds
both = sorted(solved["r1"] & solved["r2"] & solved["e1"])
print(f"\ntasks solved in all three rounds: {len(both)}")
for name, key in (("llm calls", "total_llm_calls"), ("duration s", "duration_seconds")):
    b = [st.mean([res['r1'][t][key], res['r2'][t][key]]) for t in both]
    a = [res["e1"][t][key] for t in both]
    wins = sum(x < y for x, y in zip(a, b))
    print(f"  {name}: base mean {st.mean(b):.1f} vs adapter {st.mean(a):.1f} | adapter lower on {wins}/{len(both)} tasks")
# noise yardstick: how different are r1 and r2 on the same tasks?
for name, key in (("llm calls", "total_llm_calls"), ("duration s", "duration_seconds")):
    x = [res["r1"][t][key] for t in both]
    y = [res["r2"][t][key] for t in both]
    print(f"  yardstick r1 vs r2 ({name}): r1 mean {st.mean(x):.1f} vs r2 {st.mean(y):.1f} | r2 lower on {sum(a < b for a, b in zip(y, x))}/{len(both)}")

# ---- turn-level behaviour from the captures
stats = {rd: {"turn_ctok": [], "prompt": [], "tok_s": [], "finish": collections.Counter(), "tools": collections.Counter(), "think_chars": [], "first_tool": collections.Counter(),
              "repeat": 0, "turns": 0, "notool": 0, "short_think": 0, "per_task_turn": collections.defaultdict(list)} for rd in res}
for d in (g2, g3):
    last_call = {}
    with gzip.open(d / "capture.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rd = rec["round"]
            if rd not in stats:
                continue
            msgs = rec["request"]["messages"]
            sys_txt = next((msg_text(m) for m in msgs if m.get("role") == "system"), "")
            if not sys_txt.startswith(MAIN):
                continue
            ch = rec["response"]["choices"][0]
            m = ch["message"]
            s = stats[rd]
            u = rec["response"].get("usage") or {}
            s["turns"] += 1
            s["turn_ctok"].append(u.get("completion_tokens", 0))
            s["prompt"].append(u.get("prompt_tokens", 0))
            dt = rec["t1"] - rec["t0"]
            if dt > 0 and u.get("completion_tokens", 0) > 100:
                s["tok_s"].append(u["completion_tokens"] / dt)
            s["finish"][ch.get("finish_reason")] += 1
            think = m.get("reasoning") or m.get("reasoning_content") or ""
            s["think_chars"].append(len(think))
            calls = m.get("tool_calls") or []
            if not calls:
                s["notool"] += 1
            sig = json.dumps([(c["function"]["name"], c["function"]["arguments"]) for c in calls], sort_keys=True)
            for c in calls:
                s["tools"][c["function"]["name"]] += 1
            key = rec["task_key"]
            if key in last_call and last_call[key] == sig and calls:
                s["repeat"] += 1
            if key not in last_call and calls:
                s["first_tool"][calls[0]["function"]["name"]] += 1
            last_call[key] = sig

print("\n== turn-level behaviour (main-agent requests only) ==")
for rd, s in stats.items():
    n = s["turns"]
    print(f"\n{rd}: turns {n} | completion tokens/turn mean {st.mean(s['turn_ctok']):.0f} median {st.median(s['turn_ctok']):.0f} | prompt tokens mean {st.mean(s['prompt']):.0f} | "
          f"decode tok/s per request median {st.median(s['tok_s']):.1f} (concurrent load)")
    print(f"   finish: {dict(s['finish'])} | turns without tool call {s['notool']} ({100 * s['notool'] / n:.1f}%) | exact repeat of previous call {s['repeat']} ({100 * s['repeat'] / n:.1f}%) | "
          f"thinking chars mean {st.mean(s['think_chars']):.0f}")
    tot = sum(s["tools"].values())
    print("   tool mix:", {k: f"{100 * v / tot:.0f}%" for k, v in s["tools"].most_common(8)})
    print("   first tool:", dict(s["first_tool"].most_common(5)))
