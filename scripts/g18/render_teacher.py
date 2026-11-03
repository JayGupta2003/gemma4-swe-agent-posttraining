"""Render a teacher episode (scripts/g18/teacher_convert.py) into a training row {input_ids, loss_spans, ...} for scripts/g3/train_g3.py, with the E4B chat template.

Rendering matches what vLLM feeds the model (verified token-for-token on real captured E4B requests, scripts/g18/render_check.py): tool-call arguments are dicts, and the
system message is a one-part list (vLLM converts string contents to parts; the template then adds a trailing space after the part). The whole episode is rendered as one
sequence (the template puts every tool response inside the single model turn), and the loss covers exactly the tokens the model generates at each step:
    for assistant message k:  text[len(render(messages[:k], add_generation_prompt=True)) : end of its stop token]      (stop token: `<|tool_response>` after a call, `<turn|>` after the final text)
The prefix property (the generation-prompt render is a prefix of the full render) is asserted per message; a violation drops the episode.
"""

import copy
import json

STOP_CALL, STOP_END = "<|tool_response>", "<turn|>"


def prep(messages):
    out = []
    for m in messages:
        m = copy.deepcopy(m)
        for tc in m.get("tool_calls") or []:
            a = tc["function"]["arguments"]
            if isinstance(a, str):
                tc["function"]["arguments"] = json.loads(a or "{}")
        if m["role"] == "system" and isinstance(m.get("content"), str):
            m["content"] = [{"type": "text", "text": m["content"]}]
        out.append(m)
    return out


def render_episode(tok, ep, tools, max_tokens=16384):
    """Returns (row, None) or (None, reason)."""
    msgs = prep(ep["messages"])
    kw = dict(tools=tools, tokenize=False, enable_thinking=True)
    full = tok.apply_chat_template(msgs, add_generation_prompt=False, **kw)
    char_spans = []
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prompt = tok.apply_chat_template(msgs[:k], add_generation_prompt=True, **kw)
        if not full.startswith(prompt):
            return None, "not_prefix_stable"
        stop = STOP_CALL if m.get("tool_calls") else STOP_END
        j = full.find(stop, len(prompt))
        if j < 0:
            return None, "no_stop_token"
        char_spans.append((len(prompt), j + len(stop)))
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_tokens:
        return None, "too_long"
    starts = {a: i for i, (a, b) in enumerate(offs) if b > a}
    ends = {b: i for i, (a, b) in enumerate(offs) if b > a}
    spans = []
    for a, b in char_spans:
        if a not in starts or b not in ends:
            return None, "span_not_on_token_boundary"
        spans.append((starts[a], ends[b] + 1))
    return {"task": ep["instance_id"], "repo": ep["repo"], "n_requests": len(spans), "input_ids": ids, "loss_spans": spans, "n_tokens": len(ids),
            "n_loss_tokens": sum(e - s for s, e in spans), "n_calls": ep["n_calls"], "files": ep["files"]}, None
