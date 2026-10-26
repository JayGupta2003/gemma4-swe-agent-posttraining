"""Preference pairs for the RPO run (docs/PREFERENCE_RUN.md, section 3). Offline, no GPU.

Input: one or more POOL rounds, each a run directory with `capture.jsonl.gz` (exact token ids from scripts/g2/capture_proxy.py) and `<round>/task_results.jsonl` (+ `<round>/patches/<id>.patch`).
Per (task, round) there is one attempt. Its FINAL context segment (the segment after the last compaction) becomes a training sequence exactly like in scripts/g10/build_sft_general.py:
    input_ids  = final prompt token ids of the segment + completion token ids of its final request
    loss_spans = [start, end) token ranges that the model generated (every earlier assistant turn of the segment, located inside the final prompt, and the final completion)
A segment is dropped when its history is not token-exact (a request's prompt must extend the previous prompt + completion), when it is longer than --max-tokens, or when it cannot be attributed to a task.

Classes: S = resolved, F = finished but not resolved, C = capped (error mentions a timeout).
Pairs per task (at most --max-pairs-per-task = 2, every attempt used at most once):
    chosen   = the fastest S attempt (fewest seconds of wall time), whose patch does not touch test files
    rejected = the F or C attempt with the MOST seconds
    pair k uses the k-th fastest S and the k-th slowest non-S.
Within-S pairs (faster over slower by >= 30%) are added only when there are fewer than --min-main-pairs (40) S-vs-non-S pairs.
All tasks must be POOL tasks: the 75 competition tasks stay untouched as the evaluation set (guard: ids found in --exclude are rejected).

Usage: python scripts/g15/build_pairs.py --tasks <jsonl with instance_id + problem_statement> --out <dir> --run <run_dir>:<round> [--run ...] [--max-tokens 16384] [--exclude ids.json]
Output: <out>/pairs.jsonl (one pair per line with both token sequences) and <out>/stats.json.
"""

import argparse
import collections
import gzip
import json
import re
import statistics
import sys
from pathlib import Path

MAIN_PREFIX = "You are an expert autonomous software engineer"
TEST_FILE_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$")


def text(m):
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c)


def read_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class TaskIndex:
    """Attribute a request to a task through the problem statement (first 300 characters of the statement inside the first user message)."""

    def __init__(self, tasks):
        self.entries = [(t["instance_id"], t["problem_statement"][:300], t["problem_statement"][:200]) for t in tasks if t.get("problem_statement")]

    def of_first_user(self, first_user):
        hit = [iid for iid, p300, _ in self.entries if p300 and p300 in first_user]
        return hit[0] if len(hit) == 1 else None

    def of_request_text(self, req_txt):
        hit = [iid for iid, _, p200 in self.entries if p200 and p200 in req_txt]
        return hit[0] if len(hit) == 1 else None


def collect(capture_path, round_tag, index):
    """Return (segments, key_task, summaries): segments[key] = [(t0, t1, prompt_ids, completion_ids)] of the main agent, key_task[key] = task id or None,
    summaries = compaction summary requests (no system message) with their task, timing and text."""
    segments = collections.defaultdict(list)
    first_user, key_task, summaries = {}, {}, []
    for rec in read_records(capture_path):
        if rec["round"] != round_tag:
            continue
        msgs = rec["request"]["messages"]
        sys_txt = next((text(m) for m in msgs if m.get("role") == "system"), "")
        ch = rec["response"]["choices"][0]
        if not sys_txt:
            summaries.append({"task": index.of_request_text(" ".join(text(m) for m in msgs)), "t0": rec["t0"], "t1": rec["t1"], "text": ch["message"].get("content") or ""})
            continue
        if not sys_txt.startswith(MAIN_PREFIX):
            continue
        k = rec["task_key"]
        if k not in first_user:
            first_user[k] = next((text(m) for m in msgs if m.get("role") == "user"), "")
            key_task[k] = index.of_first_user(first_user[k])
        segments[k].append((rec["t0"], rec["t1"], rec["response"].get("prompt_token_ids") or [], ch.get("token_ids") or []))
    for k in segments:                                   # continuation segments: link through the summary text that starts their context
        if key_task[k] is None:
            cand = {sm["task"] for sm in summaries if sm["task"] and sm["text"][:150] and sm["text"][:150] in first_user[k]}
            key_task[k] = cand.pop() if len(cand) == 1 else None
    return segments, key_task, summaries


def segment_sequence(recs, max_tokens):
    """recs: [(t0, t1, prompt_ids, completion_ids)] of ONE segment. Returns (ids, spans, None) or (None, None, reason)."""
    recs = sorted(recs, key=lambda r: r[0])
    for (_, _, pa, ca), (_, _, pb, _) in zip(recs, recs[1:]):
        if pb[:len(pa)] != pa or not ca or ca[-1] not in pb[len(pa):]:
            return None, None, "malformed history"
    spans = []
    for (_, _, pa, ca), (_, _, pb, _) in zip(recs, recs[1:]):
        idx = pb[len(pa):].index(ca[-1])                  # the stop token is the last token the model generated
        spans.append((len(pa), len(pa) + idx + 1))
    last_prompt, last_comp = recs[-1][2], recs[-1][3]
    ids = list(last_prompt) + list(last_comp)
    spans.append((len(last_prompt), len(ids)))
    if len(ids) > max_tokens:
        return None, None, f"longer than {max_tokens} tokens"
    return ids, spans, None


def classify(row):
    if row["resolved"]:
        return "S"
    return "C" if "timeout" in (row.get("error") or "") else "F"


def build_attempts(run_dir, round_tag, index, max_tokens):
    """One attempt per task of the round: class, seconds, final-segment sequence (or the reason it is unusable), patch file list."""
    run_dir = Path(run_dir)
    segments, key_task, summaries = collect(run_dir / "capture.jsonl.gz", round_tag, index)
    results = {}
    for line in open(run_dir / round_tag / "task_results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        results[r["instance_id"]] = r
    keys_of = collections.defaultdict(list)
    for k, iid in key_task.items():
        if iid is not None:
            keys_of[iid].append(k)
    attempts, drop = {}, collections.Counter()
    for iid, row in results.items():
        ks = keys_of.get(iid, [])
        if not ks:
            drop["no captured main-agent requests"] += 1
            continue
        final_key = max(ks, key=lambda k: max(r[0] for r in segments[k]))
        ids, spans, why = segment_sequence(segments[final_key], max_tokens)
        patch_file = run_dir / round_tag / "patches" / f"{iid}.patch"
        patch = patch_file.read_text(encoding="utf-8", errors="ignore") if patch_file.exists() else None
        touched = re.findall(r"^\+\+\+ b/(\S+)", patch or "", flags=re.M)
        model_s = sum(r[1] - r[0] for k in ks for r in segments[k]) + sum(sm["t1"] - sm["t0"] for sm in summaries if sm["task"] == iid)
        attempts[iid] = {"task": iid, "round": round_tag, "cls": classify(row), "seconds": float(row["duration_seconds"]), "model_seconds": round(model_s, 1),
                         "n_segments": len(ks), "n_requests": len(segments[final_key]), "ids": ids, "spans": spans, "unusable": why,
                         "patch_known": patch is not None, "touches_tests": any(TEST_FILE_RE.search(p) for p in touched), "patch_files": touched}
        if why:
            drop[why] += 1
    return attempts, drop


def pick_pairs(by_task, max_pairs=2, min_main_pairs=40, within_s_ratio=0.7):
    """by_task: task -> list of attempts (several rounds). Returns (pairs, info). Pure function (unit-tested)."""
    pairs, info = [], collections.Counter()
    usable = {t: [a for a in lst if not a["unusable"]] for t, lst in by_task.items()}
    for t, lst in sorted(usable.items()):
        S = sorted((a for a in lst if a["cls"] == "S" and not a["touches_tests"]), key=lambda a: a["seconds"])
        N = sorted((a for a in lst if a["cls"] in ("F", "C")), key=lambda a: -a["seconds"])
        if any(a["cls"] == "S" and a["touches_tests"] for a in lst):
            info["S attempts dropped: patch touches test files"] += sum(1 for a in lst if a["cls"] == "S" and a["touches_tests"])
        k = min(len(S), len(N), max_pairs)
        for i in range(k):
            pairs.append({"task": t, "kind": f"s_vs_{N[i]['cls'].lower()}", "chosen": S[i], "rejected": N[i]})
        if len(S) == 1 and len(N) >= 2 and max_pairs >= 2:
            info["extra pairs if the single S were reused"] += 1
        if k == 0:
            info["task without a contrast"] += 1
    main_n = len(pairs)
    if main_n < min_main_pairs:
        for t, lst in sorted(usable.items()):
            S = sorted((a for a in lst if a["cls"] == "S" and not a["touches_tests"]), key=lambda a: a["seconds"])
            if len(S) >= 2 and S[0]["seconds"] <= within_s_ratio * S[-1]["seconds"]:
                pairs.append({"task": t, "kind": "within_s", "chosen": S[0], "rejected": S[-1]})
    info["main (S vs non-S) pairs"] = main_n
    info["within-S pairs added"] = len(pairs) - main_n
    return pairs, info


def slim(a):
    return {k: a[k] for k in ("round", "cls", "seconds", "model_seconds", "n_requests", "n_segments")} | {"input_ids": a["ids"], "loss_spans": a["spans"], "n_tokens": len(a["ids"]),
                                                                                                          "n_loss_tokens": sum(e - s for s, e in a["spans"])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, help="<run_dir>:<round>")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-pairs-per-task", type=int, default=2)
    ap.add_argument("--min-main-pairs", type=int, default=40)
    ap.add_argument("--exclude", help="json list of instance ids that must not appear (the competition tasks)")
    a = ap.parse_args(argv)
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8")]
    index = TaskIndex(tasks)
    exclude = set(json.load(open(a.exclude))) if a.exclude else set()
    by_task, drops = collections.defaultdict(list), {}
    for spec in a.run:
        run_dir, round_tag = spec.rsplit(":", 1)
        attempts, drop = build_attempts(run_dir, round_tag, index, a.max_tokens)
        assert not (set(attempts) & exclude), f"competition tasks found in {spec}: {sorted(set(attempts) & exclude)[:3]}"
        drops[spec] = dict(drop)
        for iid, at in attempts.items():
            by_task[iid].append(at)
    pairs, info = pick_pairs(by_task, a.max_pairs_per_task, a.min_main_pairs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"task": p["task"], "kind": p["kind"], "chosen": slim(p["chosen"]), "rejected": slim(p["rejected"])}) + "\n")
    cls_count = collections.Counter(at["cls"] for lst in by_task.values() for at in lst)
    tok = [len(p["chosen"]["ids"]) + len(p["rejected"]["ids"]) for p in pairs]
    stats = {"rounds": a.run, "attempts": sum(len(v) for v in by_task.values()), "tasks": len(by_task), "attempt_classes": dict(cls_count), "dropped_per_round": drops, "selection": dict(info),
             "pairs": len(pairs), "pairs_by_kind": dict(collections.Counter(p["kind"] for p in pairs)), "tasks_with_pairs": len({p["task"] for p in pairs}),
             "tokens_per_pair_median": statistics.median(tok) if tok else 0, "tokens_total_m": round(sum(tok) / 1e6, 2),
             "gate_main_pairs_ge_40": info["main (S vs non-S) pairs"] >= 40}
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    main()
