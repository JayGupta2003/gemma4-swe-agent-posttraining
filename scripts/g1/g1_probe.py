"""HTTP probes against an OpenAI-compatible vLLM server: decode speed, TTFT, concurrency, long-prompt (hang) check, adapter-activity check.

Pure `requests` + stdlib so it can be tested against a mock server locally (see test_probe_mock.py) before spending GPU time.
"""

import json
import statistics
import threading
import time

import requests


def list_models(base_url: str) -> list[str]:
    r = requests.get(f"{base_url}/models", timeout=30)
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"]]


def stream_completion(base_url: str, model: str, messages: list[dict], max_tokens: int = 256, min_tokens: int | None = None,
                      timeout: float = 600, extra: dict | None = None) -> dict:
    """One streamed chat completion; returns timing split into time-to-first-token and steady decode speed."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    if min_tokens:
        body["min_tokens"] = min_tokens
    body.update(extra or {})
    t0 = time.perf_counter()
    t_first = t_last = None
    n_chunks, usage = 0, {}
    with requests.post(f"{base_url}/chat/completions", json=body, stream=True, timeout=timeout) as r:
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "error": r.text[:300]}
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            payload = raw[6:]
            if payload == b"[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    n_chunks += 1
    t_end = time.perf_counter()
    comp = usage.get("completion_tokens", n_chunks)
    out = {"ok": True, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": comp, "total_s": round(t_end - t0, 2),
           "ttft_s": round(t_first - t0, 2) if t_first else None}
    if t_first and t_last and t_last > t_first and comp > 1:
        out["decode_tok_s"] = round((comp - 1) / (t_last - t_first), 2)
    return out


def decode_probe(base_url: str, model: str, messages: list[dict], reps: int = 3, tokens: int = 256) -> dict:
    """Warm-up + `reps` fixed-length generations at concurrency 1 (the scorer is sequential, so this is the number that matters)."""
    stream_completion(base_url, model, messages, max_tokens=32)  # warm-up (also warms LoRA kernels / prefix cache)
    runs = [stream_completion(base_url, model, messages, max_tokens=tokens, min_tokens=tokens) for _ in range(reps)]
    good = [r for r in runs if r.get("ok") and r.get("decode_tok_s")]
    return {"model": model, "runs": runs,
            "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in good) if good else None,
            "ttft_s_median": statistics.median(r["ttft_s"] for r in good) if good else None}


def concurrency_probe(base_url: str, model: str, messages: list[dict], n: int = 4, tokens: int = 200) -> dict:
    results: list[dict] = []
    lock = threading.Lock()

    def one():
        r = stream_completion(base_url, model, messages, max_tokens=tokens, min_tokens=tokens)
        with lock:
            results.append(r)

    t0 = time.perf_counter()
    ths = [threading.Thread(target=one) for _ in range(n)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.perf_counter() - t0
    toks = sum(r.get("completion_tokens", 0) for r in results if r.get("ok"))
    return {"model": model, "n": n, "wall_s": round(wall, 2), "aggregate_tok_s": round(toks / wall, 1), "per_request": results}


def long_prompt_probe(base_url: str, model: str, target_chars: int, timeout: float = 420) -> dict:
    """Does a long prompt (agent conversations reach 20k+ tokens) complete, or sit in the waiting queue forever?"""
    filler = "".join(f"def helper_{i}(x):\n    return x + {i}  # utility {i}\n\n" for i in range(target_chars // 50))
    msgs = [{"role": "user", "content": filler + "\nSummarize the above in one word."}]
    try:
        return {"model": model, "target_chars": target_chars, **stream_completion(base_url, model, msgs, max_tokens=16, timeout=timeout)}
    except requests.exceptions.RequestException as e:
        return {"model": model, "ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}", "timeout_s": timeout}


LOGPROB_PROMPTS = ["Write a Python function that reverses a singly linked list.",
                   "Explain in two sentences what a race condition is.",
                   "Fix the bug: def add(a, b): return a - b"]


def logprob_probe(base_url: str, model: str, n_tokens: int = 24) -> dict:
    outs = []
    for p in LOGPROB_PROMPTS:
        body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": n_tokens, "temperature": 0.0,
                "logprobs": True, "top_logprobs": 3}
        r = requests.post(f"{base_url}/chat/completions", json=body, timeout=300)
        if r.status_code != 200:
            outs.append({"error": r.text[:200]})
            continue
        ch = r.json()["choices"][0]
        lp = ch.get("logprobs", {}).get("content") or []
        outs.append({"text": ch["message"].get("content") or ch["message"].get("reasoning_content") or "",
                     "tokens": [x["token"] for x in lp], "logprobs": [x["logprob"] for x in lp]})
    return {"model": model, "outs": outs}


def compare_logprobs(a: dict, b: dict) -> dict:
    """Is model b different from model a? Compares tokens and per-token logprobs over the common prefix of identical tokens."""
    res = {"prompts": []}
    for oa, ob in zip(a["outs"], b["outs"]):
        if "error" in oa or "error" in ob:
            res["prompts"].append({"error": oa.get("error") or ob.get("error")})
            continue
        same_text = oa["text"] == ob["text"]
        k = 0
        while k < min(len(oa["tokens"]), len(ob["tokens"])) and oa["tokens"][k] == ob["tokens"][k]:
            k += 1
        max_d = max((abs(x - y) for x, y in zip(oa["logprobs"][:k], ob["logprobs"][:k])), default=0.0)
        res["prompts"].append({"same_text": same_text, "identical_prefix_tokens": k, "max_abs_logprob_diff_on_prefix": round(max_d, 5)})
    res["any_text_differs"] = any(not p.get("same_text", True) for p in res["prompts"])
    res["max_logprob_diff"] = max((p.get("max_abs_logprob_diff_on_prefix", 0) for p in res["prompts"]), default=0)
    return res
