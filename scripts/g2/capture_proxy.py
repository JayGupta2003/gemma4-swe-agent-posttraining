"""Transparent logging reverse proxy between the harness (LiteLLM -> ADK) and vLLM, for G2 data collection.

For every non-streaming POST .../chat/completions it records, gzip-compressed, one JSON line with the FULL request body (messages incl. reasoning,
tools, chat_template_kwargs, sampling params) and the FULL response body. It asks vLLM for `return_token_ids` so the record also carries the exact
prompt token ids and completion token ids (no re-rendering guesses); if the server rejects that parameter it retries without it and keeps going.
The harness receives the response with the token-id fields stripped, i.e. exactly what it would get without the proxy.
Everything else (GET /v1/models, /health, streaming requests) is passed through untouched.
"""

import gzip
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests


def task_key(messages: list[dict]) -> str:
    """Stable id of the agent session a request belongs to: hash of the system prompt + the first user message (the task statement)."""
    sys_txt = next((m.get("content") for m in messages if m.get("role") == "system"), "") or ""
    usr_txt = next((m.get("content") for m in messages if m.get("role") == "user"), "") or ""
    h = hashlib.sha1()
    for part in (sys_txt, usr_txt):
        h.update((part if isinstance(part, str) else json.dumps(part, sort_keys=True)).encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class CaptureProxy:
    def __init__(self, upstream: str, out_path: str, port: int = 8001, host: str = "127.0.0.1", ask_token_ids: bool = True):
        self.upstream = upstream.rstrip("/")          # e.g. http://127.0.0.1:8000
        self.out_path = out_path
        self.port, self.host = port, host
        self.ask_token_ids = ask_token_ids
        self.round_tag = "r0"
        self.tokenize_fallback = False        # if True and vLLM returns no token ids, ask /tokenize for the exact prompt ids instead
        self.n_records = 0
        self.n_with_ids = 0
        self.n_errors = 0
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> "CaptureProxy":
        proxy = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # silence
                pass

            def _relay(self, status: int, body: bytes, ctype: str = "application/json"):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                try:
                    r = requests.get(proxy.upstream + self.path, timeout=60)
                    self._relay(r.status_code, r.content, r.headers.get("Content-Type", "application/json"))
                except Exception as e:  # noqa: BLE001
                    self._relay(502, json.dumps({"error": str(e)}).encode())

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                try:
                    req = json.loads(raw) if raw else {}
                except Exception:  # noqa: BLE001
                    req = None
                is_chat = self.path.rstrip("/").endswith("/chat/completions") and isinstance(req, dict) and not req.get("stream")
                if not is_chat:                      # pass-through (streaming, other endpoints)
                    try:
                        r = requests.post(proxy.upstream + self.path, data=raw, headers={"Content-Type": "application/json"}, timeout=1800)
                        self._relay(r.status_code, r.content, r.headers.get("Content-Type", "application/json"))
                    except Exception as e:  # noqa: BLE001
                        self._relay(502, json.dumps({"error": str(e)}).encode())
                    return
                t0 = time.time()
                asked = proxy.ask_token_ids           # decided ONCE per request (another thread may flip the flag meanwhile)
                body = dict(req, return_token_ids=True) if asked else req
                try:
                    r = requests.post(proxy.upstream + self.path, json=body, timeout=1800)
                    if asked and r.status_code == 400 and "return_token_ids" in r.text:
                        proxy.ask_token_ids = False   # this vLLM rejects the parameter: stop asking, redo this call plainly
                        r = requests.post(proxy.upstream + self.path, json=req, timeout=1800)
                except Exception as e:  # noqa: BLE001
                    proxy.n_errors += 1
                    self._relay(502, json.dumps({"error": f"proxy upstream failure: {e}"}).encode())
                    return
                try:
                    resp = r.json()
                except Exception:  # noqa: BLE001
                    resp = None
                if r.status_code == 200 and isinstance(resp, dict):
                    proxy._record(req, resp, t0, time.time())
                    clean = {k: v for k, v in resp.items() if k != "prompt_token_ids"}
                    clean["choices"] = [{k: v for k, v in ch.items() if k != "token_ids"} for ch in resp.get("choices", [])]
                    self._relay(200, json.dumps(clean).encode())
                else:
                    proxy.n_errors += 1
                    self._relay(r.status_code, r.content, r.headers.get("Content-Type", "application/json"))

        self._httpd = ThreadingHTTPServer((self.host, self.port), H)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()

    # ---------------------------------------------------------------- recording
    def _record(self, req: dict, resp: dict, t0: float, t1: float) -> None:
        has_ids = bool(resp.get("prompt_token_ids")) and bool((resp.get("choices") or [{}])[0].get("token_ids"))
        rec = {"round": self.round_tag, "task_key": task_key(req.get("messages", [])), "t0": round(t0, 3), "t1": round(t1, 3), "request": req, "response": resp}
        if not has_ids and self.tokenize_fallback:
            try:
                tk = {k: req[k] for k in ("model", "messages", "tools", "chat_template_kwargs") if k in req}
                tk["add_generation_prompt"] = True
                r = requests.post(self.upstream + "/tokenize", json=tk, timeout=300)
                if r.status_code == 200:
                    rec["prompt_token_ids_tokenize"] = r.json().get("tokens")
            except Exception:  # noqa: BLE001
                pass
        line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:                             # one gzip member per record; the concatenation is a valid gzip stream
            with open(self.out_path, "ab") as f:
                f.write(gzip.compress(line, compresslevel=3))
            self.n_records += 1
            self.n_with_ids += has_ids

    def stats(self) -> dict:
        return {"records": self.n_records, "with_token_ids": self.n_with_ids, "errors": self.n_errors, "asking_token_ids": self.ask_token_ids, "round": self.round_tag}


def read_records(path: str):
    """Iterate the records of a capture file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
