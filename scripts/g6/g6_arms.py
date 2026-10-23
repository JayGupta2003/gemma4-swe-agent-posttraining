"""Config-screen arms (G6): copies of the B1 arm whose only difference is configs/sampling.yaml.

T1: shorter thinking budget and output cap (cut the long turns that eat the 5-minute clock)
T2: T1 + Gemma's usual sampling (temperature 1.0, top_p 0.95, top_k 64)
"""

import re
import shutil
from pathlib import Path

ARMS = {
    "t1": dict(temperature=0.2, top_p=0.95, top_k=None, max_output_tokens=4096, thinking_budget=2048),
    "t2": dict(temperature=1.0, top_p=0.95, top_k=64, max_output_tokens=4096, thinking_budget=2048),
    "t4": dict(temperature=1.0, top_p=0.95, top_k=64, max_output_tokens=8192, thinking_budget=4096),   # G13 probe: B1 settings except Gemma's usual sampling
    "e4b": dict(temperature=0.2, top_p=0.95, top_k=None, max_output_tokens=8192, thinking_budget=4096),   # G14a: E4B baseline with the B1/B2 sampling
    "t3": dict(temperature=0.2, top_p=0.95, top_k=None, max_output_tokens=3072, thinking_budget=1024),   # G12 (31B): half of T1's thinking budget
}


def sampling_yaml(temperature, top_p, top_k, max_output_tokens, thinking_budget) -> str:
    lines = [f"temperature: {temperature}", f"top_p: {top_p}"]
    if top_k is not None:
        lines.append(f"top_k: {top_k}")
    lines += [f"max_output_tokens: {max_output_tokens}", "thinking_config:", f"  thinking_budget: {thinking_budget}", "  include_thoughts: true", ""]
    return "\n".join(lines)


def make_arm(src: Path, dst: Path, **kw) -> Path:
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    p = dst / "configs" / "sampling.yaml"
    old = p.read_text(encoding="utf-8")
    assert re.search(r"^temperature:", old, re.M) and "thinking_budget" in old, "unexpected sampling.yaml layout"
    p.write_text(sampling_yaml(**kw), encoding="utf-8", newline="\n")
    return dst


def request_matches(request: dict, expect: dict) -> list:
    """Compare one captured vLLM request body with the arm's settings; returns the list of mismatches (empty = OK)."""
    bad = []
    if request.get("temperature") != expect["temperature"]:
        bad.append(("temperature", request.get("temperature"), expect["temperature"]))
    if request.get("top_p") != expect["top_p"]:
        bad.append(("top_p", request.get("top_p"), expect["top_p"]))
    if expect.get("top_k") is not None and request.get("top_k") != expect["top_k"]:
        bad.append(("top_k", request.get("top_k"), expect["top_k"]))
    mo = request.get("max_completion_tokens", request.get("max_tokens"))
    if mo != expect["max_output_tokens"]:
        bad.append(("max_output_tokens", mo, expect["max_output_tokens"]))
    if request.get("thinking_token_budget") != expect["thinking_budget"]:
        bad.append(("thinking_token_budget", request.get("thinking_token_budget"), expect["thinking_budget"]))
    return bad
