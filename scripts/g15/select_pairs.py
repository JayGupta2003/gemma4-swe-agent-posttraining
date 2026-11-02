"""Budgeted pair selection for the RPO run: the full pair set costs ~14 h of wall time per epoch on 4xL4 (measured SFT throughput 81 tok/s, ~3 forward-backward units per pair token),
more than a Kaggle session allows, so keep the most informative pairs that fit a token budget.
Priority: solved-vs-wrong (s_vs_f) first, then solved-vs-capped (s_vs_c) by ascending size, then within-solved pairs by ascending size.

Usage: python scripts/g15/select_pairs.py <pairs.jsonl> <out_dir> <token_budget>
"""
import json
import sys
from pathlib import Path

src, out, budget = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(line) for line in open(src, encoding="utf-8")]
size = lambda r: len(r["chosen"]["input_ids"]) + len(r["rejected"]["input_ids"])  # noqa: E731
rank = {"s_vs_f": 0, "s_vs_c": 1, "within_s": 2}
chosen, used = [], 0
for r in sorted(rows, key=lambda r: (rank[r["kind"]], size(r))):
    if used + size(r) <= budget:
        chosen.append(r)
        used += size(r)
out.mkdir(parents=True, exist_ok=True)
with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
    for r in chosen:
        f.write(json.dumps(r) + "\n")
kinds = {k: sum(1 for r in chosen if r["kind"] == k) for k in rank}
print(json.dumps({"selected": len(chosen), "of": len(rows), "tokens": used, "kinds": kinds, "tasks": len({r["task"] for r in chosen}),
                  "est_train_wall_h_1_epoch": round(used * 3 / 81 / 3600, 1)}, indent=1))
