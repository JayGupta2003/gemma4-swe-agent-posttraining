"""G20 (on-policy distillation, data stage): teacher scoring of student contexts.

The student (E4B) runs the real harness and the capture proxy records every request with exact prompt and completion token ids. For each context segment of the main
agent (one per capture task_key; compaction starts a new one) the LEAF is the request with the longest prompt, and its sequence is prompt ids + completion ids: it
contains every model turn of that segment (as re-rendered in history, the same form the SFT sequences use). The teacher (31B, same tokenizer and chat template,
verified 60/60 on real captures) scores each leaf in one prefill through vLLM's /v1/completions with `prompt_logprobs`: per position the log-probability of the
actual token and the teacher's top-k tokens. Loss spans (the student's own tokens) are derived offline from the capture, as in scripts/g10/build_sft_general.py.

Output (npz, rewritten as it grows): ids int32 [N], leaf_offsets int64 [L+1], n_prompt int32 [L], act_lp float32 [N] (teacher log p of the actual token; NaN at
position 0), top_ids int32 [N,k], top_lp float16 [N,k] (-inf padding), plus meta.json (task_key, round, t0 of the leaf request, seconds).
"""

import gzip
import json
import math
import time

import numpy as np
import requests

MAIN = "You are an expert autonomous software engineer"


def _text(c):
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def read_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def leaves(capture_path, round_prefix):
    """Main-agent leaves of the rounds whose tag starts with round_prefix, shortest first."""
    best = {}
    for rec in read_records(capture_path):
        if not str(rec.get("round", "")).startswith(round_prefix):
            continue
        msgs = rec["request"].get("messages", [])
        sysm = _text(next((m.get("content") for m in msgs if m.get("role") == "system"), ""))
        if not sysm.startswith(MAIN):
            continue
        resp = rec.get("response") or {}
        p = resp.get("prompt_token_ids")
        c = ((resp.get("choices") or [{}])[0]).get("token_ids")
        if not p or not c:
            continue
        k = rec["task_key"]
        if k not in best or len(p) > best[k]["n_prompt"]:
            best[k] = {"task_key": k, "round": rec["round"], "t0": rec.get("t0"), "n_prompt": len(p), "ids": list(p) + list(c)}
    return sorted(best.values(), key=lambda x: len(x["ids"]))


def _tid(key):
    s = str(key)
    if s.startswith("token_id:"):
        s = s[len("token_id:"):]
    return int(s)


def _lp(v):
    return float(v["logprob"]) if isinstance(v, dict) else float(v)


def parse_prompt_logprobs(plp, ids, k):
    """vLLM prompt_logprobs (list, None at position 0, else {token_id: {logprob, rank, decoded_token}} with the actual token plus the top-k) -> arrays."""
    n = len(ids)
    if len(plp) != n:
        raise ValueError(f"prompt_logprobs has {len(plp)} positions, sequence has {n}")
    act = np.full(n, np.nan, np.float32)
    top_ids = np.full((n, k), -1, np.int32)
    top_lp = np.full((n, k), -np.inf, np.float16)
    for i, d in enumerate(plp):
        if not d:
            continue
        items = sorted(((_tid(t), _lp(v)) for t, v in d.items()), key=lambda x: -x[1])
        dd = dict(items)
        if ids[i] not in dd:
            raise ValueError(f"actual token {ids[i]} missing at position {i}")
        act[i] = dd[ids[i]]
        for j, (t, lp) in enumerate(items[:k]):
            top_ids[i, j] = t
            top_lp[i, j] = lp
    return act, top_ids, top_lp


def score_leaf(url, model, ids, k, timeout=1800):
    r = requests.post(url.rstrip("/") + "/v1/completions",
                      json={"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": k}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()["choices"][0]["prompt_logprobs"]


class Store:
    def __init__(self, k):
        self.k, self.rows, self.meta = k, [], []

    def add(self, leaf, act, top_ids, top_lp, seconds):
        self.rows.append((np.asarray(leaf["ids"], np.int32), leaf["n_prompt"], act, top_ids, top_lp))
        self.meta.append({"task_key": leaf["task_key"], "round": leaf["round"], "t0": leaf["t0"], "n_tokens": len(leaf["ids"]), "n_prompt": leaf["n_prompt"], "seconds": round(seconds, 2)})

    def save(self, npz_path, meta_path):
        if not self.rows:
            return
        lens = [len(r[0]) for r in self.rows]
        np.savez(npz_path, ids=np.concatenate([r[0] for r in self.rows]), leaf_offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                 n_prompt=np.array([r[1] for r in self.rows], np.int32), act_lp=np.concatenate([r[2] for r in self.rows]),
                 top_ids=np.concatenate([r[3] for r in self.rows]), top_lp=np.concatenate([r[4] for r in self.rows]))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"k": self.k, "leaves": self.meta}, f)


def score_all(leaves_, job, store, out_of_time, on_progress, workers=2, every=10):
    """Run job(leaf) -> (act, top_ids, top_lp, seconds) over leaves with `workers` requests in flight; stop submitting once out_of_time() is True.
    Failed leaves are counted and their first errors kept; on_progress() is called every `every` finished leaves and at the end. Returns (n_ok, errors)."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    it, pending, errors, n_done, n_ok = iter(leaves_), {}, [], 0, 0
    with ThreadPoolExecutor(workers) as ex:
        def fill():
            while len(pending) < workers and not out_of_time():
                leaf = next(it, None)
                if leaf is None:
                    return
                pending[ex.submit(job, leaf)] = leaf
        fill()
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for f in done:
                leaf = pending.pop(f)
                try:
                    act, top_ids, top_lp, sec = f.result()
                    store.add(leaf, act, top_ids, top_lp, sec)
                    n_ok += 1
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{leaf['task_key']}: {str(e)[:300]}")
                n_done += 1
                if n_done % every == 0:
                    on_progress()
            fill()
    on_progress()
    return n_ok, errors


def summary(store):
    """Mean teacher log p of the actual tokens over the completion part of each leaf (a rough 'how much does the teacher agree with the student' number)."""
    vals = []
    for ids, n_prompt, act, _, _ in store.rows:
        a = act[n_prompt:]
        a = a[~np.isnan(a)]
        if len(a):
            vals.append(float(a.mean()))
    return {"leaves": len(store.rows), "tokens": int(sum(len(r[0]) for r in store.rows)),
            "mean_teacher_lp_last_completion": round(sum(vals) / len(vals), 4) if vals else None,
            "frac_last_completion_tokens_teacher_lp_below_ln0.1": round(float(np.mean(np.concatenate([r[2][r[1]:] for r in store.rows]) < math.log(0.1))), 4) if store.rows else None}
