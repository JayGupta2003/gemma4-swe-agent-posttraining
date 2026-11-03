"""Exactness check of the teacher renderer: render the messages of REAL captured E4B vLLM requests with the E4B chat template and compare with the captured prompt token ids.
Usage: python scripts/g18/render_check.py <tokenizer_dir> <capture.jsonl.gz> [max_requests=150]"""
import copy
import gzip
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 150


def fix(msgs):
    out = []
    for m in msgs:
        m = copy.deepcopy(m)
        for tc in m.get("tool_calls") or []:
            a = tc["function"]["arguments"]
            if isinstance(a, str):
                tc["function"]["arguments"] = json.loads(a or "{}")
        out.append(m)
    return out


n = ok = shown = 0
for line in gzip.open(sys.argv[2], "rt", encoding="utf-8"):
    r = json.loads(line)
    q = r["request"]
    ms = q["messages"]
    if ms[0]["role"] != "system" or not q.get("tools"):
        continue
    ids = r["response"].get("prompt_token_ids")
    if not ids:
        continue
    kw = q.get("chat_template_kwargs") or {}
    text = tok.apply_chat_template(fix(ms), tools=q["tools"], tokenize=False, add_generation_prompt=True, **kw)
    got = tok(text, add_special_tokens=False)["input_ids"]
    n += 1
    same = got == list(ids)
    ok += same
    if not same and shown < 3:
        shown += 1
        k = next((i for i, (a, b) in enumerate(zip(got, ids)) if a != b), min(len(got), len(ids)))
        print("MISMATCH lens", len(got), len(ids), "first diff at", k)
        print("  mine:", repr(tok.decode(got[max(0, k - 25):k + 30])))
        print("  real:", repr(tok.decode(list(ids)[max(0, k - 25):k + 30])))
    if n >= N:
        break
print("exact prompt-token match:", ok, "/", n)
