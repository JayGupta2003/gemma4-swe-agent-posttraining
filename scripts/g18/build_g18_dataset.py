"""Select teacher episodes, render them with the E4B chat template and write the G18 training dataset.

Usage: python scripts/g18/build_g18_dataset.py <episodes.jsonl> <tokenizer_dir> <out_dir> [n_episodes=700] [max_tokens=16384] [seed=7]

Selection (deterministic): every episode with >= 7 calls first (they carry the multi-step behaviour: several reads, several hunks, more than one test run), then the rest
round-robin over repositories (one episode per repo per pass) until n_episodes. Each is rendered by scripts/g18/render_teacher.py (token-exact vLLM format, loss only on the
tokens the model generates). Output: <out_dir>/sft_sequences.jsonl (rows for scripts/g3/train_g3.py, `task` = instance id), <out_dir>/stats.json, <out_dir>/E4B/ = the E4B
arm (submission_e4b + the B2 eval_config) for the evaluation, copied verbatim.
"""

import collections
import json
import random
import re
import shutil
import statistics as st
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_teacher as RT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
src, tokdir, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 700
MAXTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 16384
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 7
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(tokdir)
tools = json.loads((Path(__file__).resolve().parent / "tools_b2.json").read_text(encoding="utf-8"))
eps = [json.loads(l) for l in open(src, encoding="utf-8")]
rng = random.Random(SEED)
multi = [e for e in eps if e["n_calls"] >= 7]
rest = [e for e in eps if e["n_calls"] < 7]
by_repo = collections.defaultdict(list)
for e in rest:
    by_repo[e["repo"]].append(e)
for v in by_repo.values():
    rng.shuffle(v)
order = list(multi)
rng.shuffle(order)
repos = sorted(by_repo)
rng.shuffle(repos)
while any(by_repo.values()):
    for r in repos:
        if by_repo[r]:
            order.append(by_repo[r].pop())
FALLBACK_FINAL = "The fix is in place and the targeted tests pass."
FALLBACK_SEARCH = "Search for the identifiers from the issue to find where they live."


def polish(e):
    """Two blemishes of the teacher's free text: a final message that is only a heading, and a first thought that talks about environment setup."""
    msgs = e["messages"]
    final = msgs[-1]
    body = "\n".join(l for l in (final.get("content") or "").splitlines() if l.strip() and not l.lstrip().startswith("#")).strip()
    final["content"] = body if len(body) >= 30 else FALLBACK_FINAL
    first = next(m for m in msgs if m.get("tool_calls"))
    if re.search(r"environment|set up|setup|working correctly|Phase", first.get("reasoning_content") or "", re.I):
        first["reasoning_content"] = FALLBACK_SEARCH
    return e


rows, why = [], collections.Counter()
for e in order:
    if len(rows) >= N:
        break
    e = polish(e)
    r, w = RT.render_episode(tok, e, tools, max_tokens=MAXTOK)
    why[w or "ok"] += 1
    if r:
        rows.append(r)
with open(out / "sft_sequences.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
nt = [r["n_tokens"] for r in rows]
nl = [r["n_loss_tokens"] for r in rows]
stats = {"episodes_available": len(eps), "rendered": dict(why), "sequences": len(rows), "repos": len({r["repo"] for r in rows}), "tokens_total_m": round(sum(nt) / 1e6, 2),
         "tokens_median": st.median(nt), "tokens_p90": sorted(nt)[int(.9 * len(nt))], "loss_tokens_total_k": round(sum(nl) / 1e3), "loss_share": round(sum(nl) / sum(nt), 3),
         "calls": dict(sorted(collections.Counter(r["n_calls"] for r in rows).items())),
         "est_train_h_at_300_toks": round(sum(nt) / 300 / 3600, 2), "est_quota_h_train": round(2 * sum(nt) / 300 / 3600, 1)}
(out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
arm = out / "E4B"
shutil.rmtree(arm, ignore_errors=True)
shutil.copytree(ROOT / "submission_e4b", arm)
print(json.dumps(stats, indent=1))
print("E4B arm eval_config:", (arm / "eval_config.yaml").read_text(encoding="utf-8").splitlines()[-4:])
