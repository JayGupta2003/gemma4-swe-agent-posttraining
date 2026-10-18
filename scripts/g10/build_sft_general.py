"""[GENERAL VERSION of scripts/g2/build_sft_dataset.py] Build SFT sequences from a capture of any run (G2 on the competition tasks, G10 on the pool). Offline, no GPU.

One sequence per SEGMENT of a resolved attempt (a compaction starts a new segment whose context is system + summary + later turns):
    input_ids = final prompt token ids of the segment + completion token ids of its final request
    loss_spans = [start, end) token ranges on which the loss is taken:
        - every earlier assistant turn, located in the final prompt as the tokens after the previous request's prompt up to and INCLUDING the first
          stop token (`<|tool_response>`, the last token the model generated), i.e. the history re-rendering of what the model produced;
        - the final completion.
All conditioning is token-exact (each prompt extends the previous one); only the assistant spans use the template's re-rendering of earlier turns
(one newline after the thought, tool arguments sorted alphabetically), which differs from the raw generation by <= 1 token in 99% of turns.

Filters: attempt resolved; patch does not touch test files; segment fits MAX_TOKENS; no malformed turns (stop token must be found in every pair);
a per-task cap on attempts so no task dominates. Output: <out>/sft_sequences.jsonl + stats.

Usage: python scripts/g10/build_sft_general.py <run_dir with capture.jsonl.gz + round dirs> <tasks.jsonl> <rounds comma-separated> <out_dir> [max_tokens=16384] [max_attempts_per_task=2]
"""

import collections
import gzip
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
g2 = Path(sys.argv[1])
TASKS_PATH = Path(sys.argv[2])
ROUNDS = tuple(sys.argv[3].split(","))
out = Path(sys.argv[4])
MAX_TOKENS = int(sys.argv[5]) if len(sys.argv) > 5 else 16384
MAX_PER_TASK = int(sys.argv[6]) if len(sys.argv) > 6 else 2
MAIN_PREFIX = "You are an expert autonomous software engineer"
out.mkdir(parents=True, exist_ok=True)
tasks = [json.loads(l) for l in open(TASKS_PATH, encoding="utf-8")]
outcomes = {}
for rd in ROUNDS:
    for l in open(g2 / rd / "task_results.jsonl", encoding="utf-8"):
        r = json.loads(l)
        outcomes[(rd, r["instance_id"])] = r


def text(m):
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c)


segments = collections.defaultdict(list)
key_task, first_user, summaries = {}, {}, []
with gzip.open(g2 / "capture.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        if rec["round"] not in ROUNDS:
            continue
        msgs = rec["request"]["messages"]
        sys_txt = next((text(m) for m in msgs if m.get("role") == "system"), "")
        ch = rec["response"]["choices"][0]
        if not sys_txt:
            req_txt = " ".join(text(m) for m in msgs)
            hit = [t["instance_id"] for t in tasks if t["problem_statement"][:200] and t["problem_statement"][:200] in req_txt]
            summaries.append({"round": rec["round"], "task": hit[0] if len(hit) == 1 else None, "text": ch["message"].get("content") or ""})
            continue
        if not sys_txt.startswith(MAIN_PREFIX):
            continue
        k = (rec["round"], rec["task_key"])
        if k not in first_user:
            first_user[k] = next((text(m) for m in msgs if m.get("role") == "user"), "")
            hit = [t["instance_id"] for t in tasks if t["problem_statement"][:300] and t["problem_statement"][:300] in first_user[k]]
            key_task[k] = hit[0] if len(hit) == 1 else None
        segments[k].append((rec["t0"], rec["response"].get("prompt_token_ids") or [], ch.get("token_ids") or []))
for k in list(segments):                               # link post-compaction continuations to their task through the summary text
    if key_task[k] is None:
        cand = {sm["task"] for sm in summaries if sm["round"] == k[0] and sm["task"] and sm["text"][:150] and sm["text"][:150] in first_user[k]}
        key_task[k] = cand.pop() if len(cand) == 1 else None

drop = collections.Counter()
sequences = []
for k, recs in segments.items():
    iid = key_task[k]
    if iid is None:
        drop["segment not attributable to a task"] += 1
        continue
    o = outcomes.get((k[0], iid))
    if not o or not o["resolved"]:
        drop["attempt not resolved"] += 1
        continue
    patch_file = g2 / k[0] / "patches" / f"{iid}.patch"
    patch = patch_file.read_text(encoding="utf-8", errors="ignore") if patch_file.exists() else ""
    touched = re.findall(r"^\+\+\+ b/(\S+)", patch, flags=re.M)
    if any(re.search(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$", p) for p in touched):
        drop["patch touches test files"] += 1
        continue
    recs.sort(key=lambda r: r[0])
    ids = list(recs[0][1])
    spans, ok = [], True
    for (_, pa, ca), (_, pb, _) in zip(recs, recs[1:]):
        if pb[:len(pa)] != pa or not ca or ca[-1] not in pb[len(pa):]:
            ok = False
            break
        idx = pb[len(pa):].index(ca[-1])
        spans.append((len(pa), len(pa) + idx + 1))
        ids = list(pb)
    if not ok:
        drop["malformed pair inside a segment"] += 1
        continue
    last_prompt, last_comp = recs[-1][1], recs[-1][2]
    ids = list(last_prompt) + list(last_comp)
    spans.append((len(last_prompt), len(ids)))
    if len(ids) > MAX_TOKENS:
        drop[f"longer than {MAX_TOKENS} tokens"] += 1
        continue
    sequences.append({"task": iid, "round": k[0], "n_requests": len(recs), "input_ids": ids, "loss_spans": spans, "n_tokens": len(ids),
                      "n_loss_tokens": sum(e - s for s, e in spans), "patch_files": touched})

# per-task cap (keep the shortest attempts of each task: they are the most efficient ones)
by_task = collections.defaultdict(list)
for s in sequences:
    by_task[s["task"]].append(s)
kept = []
for iid, lst in by_task.items():
    per_attempt = collections.defaultdict(list)
    for s in lst:
        per_attempt[s["round"]].append(s)
    rounds = sorted(per_attempt, key=lambda r: sum(x["n_tokens"] for x in per_attempt[r]))[:MAX_PER_TASK]
    for r in rounds:
        kept += per_attempt[r]
with open(out / "sft_sequences.jsonl", "w", encoding="utf-8") as f:
    for s in kept:
        f.write(json.dumps(s) + "\n")

print("dropped:", dict(drop))
print(f"sequences before per-task cap: {len(sequences)} over {len(by_task)} tasks | after cap ({MAX_PER_TASK} attempts/task): {len(kept)} over {len({s['task'] for s in kept})} tasks")
nt, nl = [s["n_tokens"] for s in kept], [s["n_loss_tokens"] for s in kept]
print(f"tokens per sequence: median {statistics.median(nt):.0f} p90 {sorted(nt)[int(.9 * len(nt))]} max {max(nt)} | total {sum(nt)/1e6:.2f}M tokens | loss tokens total {sum(nl)/1e3:.0f}k ({100*sum(nl)/sum(nt):.0f}%)")
print(f"estimated training time at 78 tok/s: {sum(nt)/78/3600:.1f} h of L4x4 (~{2*sum(nt)/78/3600:.1f} quota h) per epoch")
print("sequences by repo:", dict(collections.Counter(next(t['repo'] for t in tasks if t['instance_id'] == s['task']).split('/')[-1] for s in kept)))
print("tasks:", sorted({s["task"] for s in kept}))
