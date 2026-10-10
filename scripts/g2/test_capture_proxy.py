"""Local test of capture_proxy against a mock vLLM, using the SAME client stack as the harness (LiteLLM). Run in the gemma4agent env."""
import json
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import capture_proxy as CP  # noqa: E402

STRICT = os.environ.get("MOCK_REJECT_IDS") == "1"   # simulate a vLLM that rejects return_token_ids


class Mock(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"data": [{"id": "mock-model"}]})

    def _tokenize(self, req):
        self._send(200, {"tokens": list(range(len(str(req["messages"])) // 4 + 5)), "count": 1})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/tokenize"):
            return self._tokenize(req)
        if STRICT and "return_token_ids" in req:
            return self._send(400, {"error": {"message": "Extra inputs are not permitted: return_token_ids"}})
        n_prompt = sum(len(str(m.get("content", ""))) for m in req["messages"]) // 4 + 5
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
                "choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": "", "reasoning": "think think",
                                         "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_command", "arguments": json.dumps({"command": "ls"})}}]}}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": 7, "total_tokens": n_prompt + 7}}
        if req.get("return_token_ids"):
            resp["prompt_token_ids"] = list(range(n_prompt))
            resp["choices"][0]["token_ids"] = [101, 102, 103]
        self._send(200, resp)


up = ThreadingHTTPServer(("127.0.0.1", 18801), Mock)
threading.Thread(target=up.serve_forever, daemon=True).start()
out = os.path.join(tempfile.mkdtemp(), "cap.jsonl.gz")
proxy = CP.CaptureProxy("http://127.0.0.1:18801", out, port=18802).start()
proxy.round_tag = "r7"
proxy.tokenize_fallback = STRICT   # in the strict case, exercise the /tokenize fallback

import litellm  # noqa: E402  (the client library the harness uses)
tools = [{"type": "function", "function": {"name": "run_command", "description": "run", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]


def call(i):
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": f"task number {i % 3}"}, {"role": "assistant", "content": "", "reasoning_content": "prev thought",
            "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "run_command", "arguments": "{\"command\": \"pwd\"}"}}]}, {"role": "tool", "tool_call_id": "c0", "content": "/workspace"}]
    r = litellm.completion(model="openai/mock-model", api_base="http://127.0.0.1:18802/v1", api_key="EMPTY", messages=msgs, tools=tools, temperature=0.2,
                           extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    return r


with ThreadPoolExecutor(4) as ex:
    results = list(ex.map(call, range(8)))
r0 = results[0]
print("client got tool call:", r0.choices[0].message.tool_calls[0].function.name, "| ids leaked to client:", "prompt_token_ids" in json.dumps(r0.model_dump()))
st = proxy.stats()
print("proxy stats:", st)
recs = list(CP.read_records(out))
assert len(recs) == 8, len(recs)
assert all(r["round"] == "r7" for r in recs)
assert len({r["task_key"] for r in recs}) == 3, "expected 3 distinct task keys"
req0 = recs[0]["request"]
assert req0["chat_template_kwargs"]["enable_thinking"] is True, "extra_body fields must arrive at the top level of the request"
assert req0["messages"][2].get("reasoning_content") == "prev thought" or req0["messages"][2].get("reasoning") == "prev thought", req0["messages"][2]
if STRICT:
    assert st["with_token_ids"] == 0 and st["asking_token_ids"] is False
    assert all(r.get("prompt_token_ids_tokenize") for r in recs), "tokenize fallback should have filled prompt ids"
    print("STRICT-SERVER FALLBACK OK (parameter rejected -> retried plainly, prompt ids obtained via /tokenize)")
else:
    assert st["with_token_ids"] == 8 and recs[0]["response"]["choices"][0]["token_ids"] == [101, 102, 103] and recs[0]["response"]["prompt_token_ids"]
    print("CAPTURE OK: 8 records, token ids captured, stripped from the client's view")
print("request keys seen:", sorted(req0.keys()))
