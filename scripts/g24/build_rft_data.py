"""G24 data: E4B self-RFT training set from several rollout runs (rule pre-registered in docs/STRATEGY.md, 2026-11-17).

For every run, scripts/g10/build_sft_general.py turns each RESOLVED attempt whose patch touches no test file into token-exact SFT sequences (one per
context segment; loss on the model's own generated tokens only). This script merges the runs and applies the pre-registered per-task cap:
at most MAX_PER_TASK attempts per task, the ones with the fewest generated (loss) tokens, i.e. the most efficient successes.
An attempt is identified by (run, round tag); every segment of a kept attempt is kept.

Usage: python scripts/g24/build_rft_data.py <out_dir> <tasks.jsonl> <run_dir>:<round,round,...> [<run_dir>:<rounds> ...]
Writes <out_dir>/sft_sequences.jsonl and <out_dir>/stats.json.
"""
import collections
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_TOKENS = 20480
MAX_PER_TASK = 2


def cap(sequences, max_per_task=MAX_PER_TASK):
    """Keep, per task, the max_per_task attempts with the fewest generated tokens (ties: run/round order). Returns the kept sequences."""
    attempts = collections.defaultdict(list)
    for s in sequences:
        attempts[(s["task"], s["run"], s["round"])].append(s)
    by_task = collections.defaultdict(list)
    for (task, run, rnd), segs in attempts.items():
        by_task[task].append((sum(x["n_loss_tokens"] for x in segs), run, rnd, segs))
    kept = []
    for task in sorted(by_task):
        for _, _, _, segs in sorted(by_task[task], key=lambda a: (a[0], a[1], a[2]))[:max_per_task]:
            kept += segs
    return kept


def main(out_dir, tasks_path, specs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    allseq, per_run = [], {}
    for spec in specs:
        run_dir, rounds = spec.rsplit(":", 1)
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "g10" / "build_sft_general.py"), run_dir, tasks_path, rounds, tmp, str(MAX_TOKENS), "999"],
                               capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-2000:])
            seqs = [json.loads(l) for l in open(Path(tmp) / "sft_sequences.jsonl", encoding="utf-8")]
        for s in seqs:
            s["run"] = Path(run_dir).name
        per_run[Path(run_dir).name] = {"sequences": len(seqs), "attempts": len({(s["task"], s["round"]) for s in seqs}),
                                       "tasks": len({s["task"] for s in seqs}), "builder_log": r.stdout.strip().splitlines()[:2]}
        allseq += seqs
    kept = cap(allseq)
    with open(out_dir / "sft_sequences.jsonl", "w", encoding="utf-8") as f:
        for s in kept:
            f.write(json.dumps(s) + "\n")
    att = collections.Counter(s["task"] for s in {(s["task"], s["run"], s["round"]): s for s in kept}.values())
    st = {"per_run": per_run, "attempts_before_cap": len({(s["task"], s["run"], s["round"]) for s in allseq}),
          "tasks": len(att), "attempts_kept": sum(att.values()), "tasks_with_2_attempts": sum(1 for v in att.values() if v >= 2),
          "sequences": len(kept), "tokens": sum(s["n_tokens"] for s in kept), "loss_tokens": sum(s["n_loss_tokens"] for s in kept),
          "max_len": max((s["n_tokens"] for s in kept), default=0), "max_tokens": MAX_TOKENS, "max_per_task": MAX_PER_TASK}
    (out_dir / "stats.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(json.dumps(st, indent=1))
    return st


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
