"""Local test of g1_probe.py against a mock OpenAI-compatible streaming server (no GPU, no vLLM)."""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import g1_probe as P  # noqa: E402


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "/base/model"}, {"id": "g1_attn"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        slow = 0.01 if req["model"] == "/base/model" else 0.013
        if req.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            n = req.get("min_tokens") or req["max_tokens"]
            time.sleep(0.2)
            for i in range(n):
                time.sleep(slow)
                self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": "x"}}]}).encode() + b"\n\n"); self.wfile.flush()
            self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 5000, "completion_tokens": n}}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            toks = ["a", "b", "c", "d"]
            shift = 0.0 if req["model"] == "/base/model" else 0.02
            body = json.dumps({"choices": [{"message": {"content": "abcd"}, "logprobs": {"content": [{"token": t, "logprob": -0.5 - shift} for t in toks]}}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)


srv = ThreadingHTTPServer(("127.0.0.1", 18765), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = "http://127.0.0.1:18765/v1"
print("models", P.list_models(base))
msgs = [{"role": "user", "content": "hi"}]
d1 = P.decode_probe(base, "/base/model", msgs, reps=2, tokens=100)
d2 = P.decode_probe(base, "g1_attn", msgs, reps=2, tokens=100)
print("decode base", d1["decode_tok_s_median"], "ttft", d1["ttft_s_median"], "| adapter", d2["decode_tok_s_median"])
assert 80 < d1["decode_tok_s_median"] < 110 and d2["decode_tok_s_median"] < d1["decode_tok_s_median"]
c = P.concurrency_probe(base, "/base/model", msgs, n=4, tokens=50)
print("concurrency", c["aggregate_tok_s"], "tok/s over", c["wall_s"], "s")
lp = P.long_prompt_probe(base, "/base/model", 20000, timeout=30)
print("long", {k: lp.get(k) for k in ("ok", "prompt_tokens", "total_s")})
cmp = P.compare_logprobs(P.logprob_probe(base, "/base/model"), P.logprob_probe(base, "g1_attn"))
print("logprob compare", cmp["max_logprob_diff"], cmp["any_text_differs"])
assert cmp["max_logprob_diff"] > 0.01
print("MOCK TEST OK")
