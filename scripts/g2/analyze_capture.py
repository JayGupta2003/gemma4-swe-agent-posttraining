"""Analysis of a G2 capture (main-agent requests only): attribution to tasks/outcomes, REAL sequence lengths, and the structure that decides how
training sequences are built.

Findings that drove the design (G2 data, 2026-10-11): (1) every prompt extends the previous prompt token-exactly (so the final prompt contains the whole
history); (2) the generated completion of turn k differs from its re-rendering in prompt k+1 only trivially (one extra newline after the thought, tool
arguments sorted alphabetically); (3) the generation stops at the `<|tool_response>` token, which the rendered history continues from.
=> one sequence per attempt segment = final prompt ids + final completion ids, with the loss mask on (a) every earlier assistant span, located as the
tokens between consecutive prompts up to and including the first `<|tool_response>`, and (b) the final completion.

Usage: python scripts/g2/analyze_capture.py <g2_out_dir>
"""

import collections
import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
g2 = Path(sys.argv[1]) / "g2"
MAIN_PREFIX = "You are an expert autonomous software engineer"
tasks = [json.loads(l) for l in open(ROOT / "data" / "tasks.jsonl", encoding="utf-8")]
outcomes = {}
for rd in ("smoke", "r1", "r2"):
    p = g2 / rd / "task_results.jsonl"
    if p.exists():
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            outcomes[(rd, r["instance_id"])] = r


def msg_text(m):
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c)


attempts = collections.defaultdict(list)      # (round, task_key) -> [(t0, prompt_ids, completion_ids, finish, n_reasoning_in_history)]
key_task = {}
first_user_by_key = {}
summaries = []
other = collections.Counter()
n_total = 0
with gzip.open(g2 / "capture.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        n_total += 1
        msgs = rec["request"]["messages"]
        sys_txt = next((msg_text(m) for m in msgs if m.get("role") == "system"), "")
        if not sys_txt.startswith(MAIN_PREFIX):
            other["empty system (compaction summary)" if not sys_txt else "other system: " + sys_txt[:40]] += 1
            if not sys_txt:   # ADK compaction summarizer call: its request holds the events being summarised, its response is the summary text
                req_txt = " ".join(msg_text(m) for m in msgs)
                hit = [t["instance_id"] for t in tasks if t["problem_statement"][:200] and t["problem_statement"][:200] in req_txt]
                summaries.append({"round": rec["round"], "task": hit[0] if len(hit) == 1 else None,
                                  "text": (rec["response"]["choices"][0]["message"].get("content") or "")})
            continue
        k = (rec["round"], rec["task_key"])
        if k not in first_user_by_key:
            first_user_by_key[k] = next((msg_text(m) for m in msgs if m.get("role") == "user"), "")
        if k not in key_task:
            fu = next((msg_text(m) for m in msgs if m.get("role") == "user"), "")
            hit = [t["instance_id"] for t in tasks if t["problem_statement"][:300] and t["problem_statement"][:300] in fu]
            key_task[k] = hit[0] if len(hit) == 1 else "UNKNOWN"
        ch = rec["response"]["choices"][0]
        attempts[k].append((rec["t0"], rec["response"].get("prompt_token_ids") or [], ch.get("token_ids") or [], ch.get("finish_reason"),
                            sum(1 for m in msgs if m.get("role") == "assistant" and (m.get("reasoning") or m.get("reasoning_content")))))

def link_continuations():
    n_lin = n_unl = 0
    for (rd, tk), recs in attempts.items():
        if key_task[(rd, tk)] != "UNKNOWN":
            continue
        recs.sort(key=lambda r: r[0])
        fu = first_user_by_key[(rd, tk)]
        cand = [sm for sm in summaries if sm["round"] == rd and sm["task"] and sm["text"][:150] and sm["text"][:150] in fu]
        tasks_hit = {sm["task"] for sm in cand}
        if len(tasks_hit) == 1:
            key_task[(rd, tk)] = tasks_hit.pop(); n_lin += 1
        else:
            n_unl += 1
    return n_lin, n_unl


print(f"records {n_total} | main-agent requests {sum(len(v) for v in attempts.values())} | other requests {dict(other)}")
LINKED, UNLINKED = link_continuations()
print(f"continuation segments linked to a task via the compaction summary: {LINKED} | unlinked: {UNLINKED}")
print("main-agent segments:", len(attempts), "| attribution:", collections.Counter("ok" if v != "UNKNOWN" else "UNKNOWN" for v in key_task.values()))
per_round = collections.Counter(rd for rd, _ in attempts)
print("attempts per round:", dict(per_round), "(expected smoke 4, r1 75, r2 75)")

STOP = None
pair = collections.Counter()
rows = []
for (rd, tk), recs in attempts.items():
    recs.sort(key=lambda r: r[0])
    iid = key_task[(rd, tk)]
    o = outcomes.get((rd, iid))
    seg_breaks = 0
    for (t0a, pa, ca, fa, _), (t0b, pb, cb, fb, _) in zip(recs, recs[1:]):
        if pb[:len(pa)] != pa:
            pair["new context (compaction)"] += 1
            seg_breaks += 1
            continue
        tail = pb[len(pa):]
        stop = ca[-1] if ca else None
        idx = tail.index(stop) if stop in tail else -1
        if idx < 0:
            pair["generation stop token NOT found in the next prompt tail"] += 1
        else:
            pair["extends exactly; stop token located in next prompt"] += 1
            pair[f"  assistant span rendered {idx + 1} tokens vs generated {len(ca)}: " + ("same length" if idx + 1 == len(ca) else "differs by %+d" % (idx + 1 - len(ca)))] += 0
            pair["  rendered-length difference (tokens): " + str(max(-3, min(3, idx + 1 - len(ca))))] += 1
        pair["finish_reason of non-final turns: " + str(fa)] += 1
    last = recs[-1]
    rows.append({"round": rd, "iid": iid, "n_req": len(recs), "seg_breaks": seg_breaks, "last_ctx": len(last[1]) + len(last[2]), "final_finish": last[3],
                 "resolved": bool(o and o["resolved"]), "error": (o or {}).get("error"), "has_outcome": o is not None, "reasoning_hist": last[4]})
print("\nconsecutive main-agent request pairs:")
for k, v in sorted(pair.items()):
    if v:
        print(f"  {v:5d}  {k}")

print("\nattempts with an outcome row:", sum(r["has_outcome"] for r in rows), "of", len(rows))
for rd in ("smoke", "r1", "r2"):
    rr = [r for r in rows if r["round"] == rd]
    if rr:
        print(f"{rd}: attempts {len(rr)} | resolved {sum(r['resolved'] for r in rr)} | requests/attempt median {statistics.median(r['n_req'] for r in rr)} | "
              f"final context tokens median {statistics.median(r['last_ctx'] for r in rr):.0f} p90 {sorted(r['last_ctx'] for r in rr)[int(.9 * len(rr))]} max {max(r['last_ctx'] for r in rr)} | "
              f"attempts with a compaction break: {sum(r['seg_breaks'] > 0 for r in rr)}")
res = [r for r in rows if r["resolved"] and r["round"] in ("r1", "r2")]
print(f"\nRESOLVED attempts (r1+r2): {len(res)} | unique tasks: {len({r['iid'] for r in res})}")
L = sorted(r["last_ctx"] for r in res)
print("  final-sequence tokens: median", L[len(L) // 2], "p90", L[int(.9 * len(L))], "max", L[-1], "| <=16,384:", sum(x <= 16384 for x in L), "| <=18,432:", sum(x <= 18432 for x in L), "of", len(L))
print("  with a compaction break:", sum(r["seg_breaks"] > 0 for r in res), "| thoughts present in history:", sum(r["reasoning_hist"] > 0 for r in res), "of", len(res))
print("  final request finish reasons:", dict(collections.Counter(r["final_finish"] for r in res)), "| outcome errors:", dict(collections.Counter((r["error"] or "none")[:35] for r in res)))
solve = collections.Counter(r["iid"] for r in res)
print("\ntasks solved in both rounds:", sum(v == 2 for v in solve.values()), "| in exactly one:", sum(v == 1 for v in solve.values()), "| never:", 75 - len(solve))
print("unique solved tasks by repo:", dict(collections.Counter(next(t['repo'] for t in tasks if t['instance_id'] == i).split('/')[-1] for i in solve)))
exp2 = set(json.loads((ROOT / "docs" / "exp2_task_sets.json").read_text(encoding="utf-8"))["discovery"])
for rd in ("r1", "r2"):
    print(f"{rd}: resolved on the 48 Exp2 tasks = {sum(1 for r in rows if r['round'] == rd and r['iid'] in exp2 and r['resolved'])}/48   (old harness, B1: Exp2 23, Exp3 18)")
json.dump(rows, open(g2 / "analysis_rows.json", "w"))
