"""Convert a RESOLVED OpenHands trajectory (nebius/SWE-rebench-openhands-trajectories, Qwen3-Coder-480B, CC BY 4.0) into a SHORT hindsight-pruned episode in OUR tool
schema (run_command / read_file / edit_file / write_file / submit_patch) and OUR harness message layout.

Why pruned: the teacher's episodes are 40-90 turns and 40-60k tokens; our agents get 28 tool calls and a 16k-token training window, so whole-trajectory imitation only
ever teaches "keep exploring" (this is what the E4B distilled from 31B windows did: explored to the cap and looped on an empty submit_patch). The pruned episode keeps
only what the final patch proves was needed, in the order a decisive agent would do it:
    search (the first teacher grep/find whose output names a patched file)  ->  read_file (the teacher's own pre-edit view of each patched file, else the diff's
    old-side context)  ->  edit_file (one call per hunk of the FINAL patch: old_string/new_string are rebuilt from the diff, so edits are correct by construction)
    ->  verify (the teacher's last passing pytest run after its last edit)  ->  submit_patch  ->  final text.
Every step's thought is the teacher's own message text/`think` content where one exists (short, no <Phase> scaffolding), else a one-line template.

convert(row, ...) returns (episode, None) or (None, reason). Pure functions, no I/O: tested by scripts/g18/test_teacher_convert.py.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = json.loads((HERE / "tools_b2.json").read_text(encoding="utf-8")) if (HERE / "tools_b2.json").exists() else []
AGENT_NAME = "swe_scaffold_v5_e4b_base"   # the E4B arm's agent name (submission_e4b/agent.yaml); the harness appends it to the instruction
_SYS = (HERE.parent.parent / "submission_e4b" / "prompts" / "system.md")
# real system message = instruction text (right-stripped) + this suffix, verified against a captured E4B request (scripts/g18, G14b capture)
SYSTEM_SUFFIX = '\n\n\nYou are an agent. Your internal name is "%s".' % AGENT_NAME
SYSTEM_PROMPT = (_SYS.read_text(encoding="utf-8").rstrip() + SYSTEM_SUFFIX) if _SYS.exists() else ""

OUT_LIMIT = 5000  # harness: command output limit (characters)
TRAILER = re.compile(r"\n?\[The command completed with exit code (-?\d+)\.\]\n\[Current working directory:.*?\]\n\[Python interpreter:.*?\]\n\[Command finished with exit code -?\d+\]\s*$", re.S)
NUMBERED = re.compile(r"^\s*(\d+)\t(.*)$")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

USER_TEMPLATE = """You are evaluating a software engineering task for repository {repo}.

Problem Statement:
{problem}

## Task Budget (Session terminates when any budget is exhausted)
- Time allowance: 8.0 minutes
- Tool calls allowance: 28 calls

## Execution Environment Rules
- Single command timeout: 150 seconds (commands exceeding this fail without ending the session)
- Command output limit: 5000 characters
- File view limit: 150 lines per read_file call
- File character limit: 10000 characters per read_file call
- Environment is offline (no network/PyPI access). All repository and test dependencies are ALREADY pre-installed. Do NOT attempt to run pip install or download packages.

## Instructions:
0. All source code is under `/workspace`. Do NOT search outside `/workspace` (e.g. `/usr/`, `/wheels/`, system site-packages). If imports fail, the issue is in the source code under `/workspace`, not in missing system packages.
1. Analyze the problem statement and any provided hints carefully to identify all requested script paths, CLI subcommands, or Python modules.
2. Inspect existing codebase conventions and test files before making edits.
3. Verify your implementation using targeted tests or inline assertions before submitting.
4. Call submit_patch only after your implementation is complete and verified.
5. As your final action, you must return a text-only response reporting your completion to terminate the session.
## Code Intelligence Tools
This repository has pre-built code graph and embedding data indexed by dotted symbol path (e.g. `pkg.module.Class.method`). Use these tools for fast, targeted navigation:
- `get_code_neighbors(node)`: Find callers and callees of a symbol node (e.g. `pkg.module.Class.method` or `Class.method`).
- `get_code_subgraph(nodes)`: Get the induced call subgraph connecting a list of symbol nodes.

## Workspace Layout
The repository is located at `/workspace`. Here is the directory tree (up to 3 levels):
```
{tree}
```
"""


def is_test_path(p: str) -> bool:
    parts = p.split("/")
    name = parts[-1]
    return (any(x in ("test", "tests", "testing") for x in parts[:-1]) or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


def parse_patch(patch: str):
    """Unified diff -> list of {path, kind: modify|add, hunks: [{old_start, old_lines, new_lines}]}; None if it holds deletions/renames/binaries."""
    files, cur = [], None
    for line in (patch or "").splitlines():
        if line.startswith("diff --git "):
            cur = {"path": None, "kind": "modify", "hunks": []}
            files.append(cur)
        elif cur is None:
            continue
        elif line.startswith(("deleted file mode", "rename from", "rename to", "Binary files", "GIT binary patch")):
            return None
        elif line.startswith("new file mode"):
            cur["kind"] = "add"
        elif line.startswith("+++ "):
            cur["path"] = line[4:].strip()[2:] if line[4:].startswith("b/") else line[4:].strip()
            if cur["path"] == "/dev/null":
                return None
        elif line.startswith("--- "):
            continue
        elif (m := HUNK.match(line)):
            cur["hunks"].append({"old_start": int(m.group(1)), "old": [], "new": [],
                                 "o_rem": int(m.group(2)) if m.group(2) is not None else 1, "n_rem": int(m.group(4)) if m.group(4) is not None else 1})
        elif cur["hunks"] and (cur["hunks"][-1]["o_rem"] > 0 or cur["hunks"][-1]["n_rem"] > 0) and not line.startswith("\\"):
            h = cur["hunks"][-1]
            tag, body = (line[:1], line[1:]) if line[:1] in (" ", "-", "+") else (" ", line)   # a blank context line may have lost its leading space
            if tag in (" ", "-"):
                h["old"].append(body)
                h["o_rem"] -= 1
            if tag in (" ", "+"):
                h["new"].append(body)
                h["n_rem"] -= 1
        elif line.startswith("\\"):
            continue
    return [f for f in files if f["path"]] or None


def trim_context(h, keep=2):
    """Shrink a hunk's context to `keep` lines each side so old_string stays short and unambiguous enough."""
    old, new = h["old"], h["new"]
    lead = 0
    while lead < min(len(old), len(new)) and old[lead] == new[lead]:
        lead += 1
    trail = 0
    while trail < min(len(old), len(new)) - lead and old[len(old) - 1 - trail] == new[len(new) - 1 - trail]:
        trail += 1
    a, b = max(0, lead - keep), max(0, trail - keep)
    return {"old_start": h["old_start"] + a, "old": old[a:len(old) - b], "new": new[a:len(new) - b]}


def call_json(tc):
    try:
        return json.loads(tc["function"]["arguments"] or "{}")
    except Exception:
        return None


class Norm:
    def __init__(self, repo_dir: str):
        self.d = repo_dir.rstrip("/")

    def text(self, s: str) -> str:
        s = (s or "").replace(self.d + "/", "").replace(self.d, ".")
        return s.replace("/workspace/", "")

    def path(self, p: str) -> str:
        p = p or ""
        if p.startswith(self.d + "/"):
            return p[len(self.d) + 1:]
        if p == self.d:
            return "."
        return p[len("/workspace/"):] if p.startswith("/workspace/") else p


def clean_text(t: str, limit=700) -> str:
    t = re.sub(r"^#{1,6}\s*Phase.*$", "", t or "", flags=re.M)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    t = re.sub(r"/workspace/\S+?/(?=\S)", "", t)
    return t[:limit].rstrip()


def short_thought(t: str, fallback: str, limit=320) -> str:
    """A step's thought: the teacher's own words if they are brief prose, else the fallback line (no headings, lists or Phase scaffolding)."""
    t = (t or "").strip()
    if not t or len(t) > limit or "###" in t or "\n- " in t or "\n1." in t or "Phase" in t or t.count("\n") > 3:
        return fallback
    return t


def parse_view(obs: str, norm: Norm):
    """OpenHands `cat -n` view -> {lineno: text}; {} if the output is not a numbered file view."""
    out = {}
    for line in (obs or "").splitlines():
        m = NUMBERED.match(line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def bash_result(obs: str, cmd: str = ""):
    m = TRAILER.search(obs or "")
    code = int(m.group(1)) if m else 0
    body = TRAILER.sub("", obs or "")
    first, _, rest = body.partition("\n")
    if cmd and first.strip() and (first.strip() == cmd.strip() or cmd.strip().startswith(first.strip()[:60])):
        body = rest   # OpenHands sometimes echoes the command as the first output line
    return body, code


def cmd_obs(body: str, code: int, tail=False):
    body = body if len(body) <= OUT_LIMIT else (body[-OUT_LIMIT:] if tail else body[:OUT_LIMIT])
    return json.dumps({"status": "ok", "stdout": body, "stderr": "", "exit_code": code})


def build_tree(norm: Norm, messages) -> str | None:
    for i, m in enumerate(messages):
        for tc in (m.get("tool_calls") or []):
            a = call_json(tc)
            if tc["function"]["name"] == "str_replace_editor" and a and a.get("command") == "view" and a.get("path") in ("/workspace", norm.d):
                obs = next((x.get("content") or "" for x in messages[i + 1:i + 3] if x.get("tool_call_id") == tc["id"]), "")
                lines = [norm.path(l.strip()) for l in obs.splitlines() if l.startswith("/workspace/") or l.startswith(norm.d)]
                lines = [l.rstrip("/") for l in lines if l and l not in (".", "workspace")]
                lines = sorted({"./" + l for l in lines if l and "/" + l != norm.d})
                if len(lines) >= 5:
                    return "\n".join(lines[:150])
    return None


def convert(row, max_edit_hunks=6, max_hunk_lines=60):
    msgs = row["trajectory"]
    if not row.get("resolved") or not row.get("model_patch"):
        return None, "not_resolved"
    m0 = next((m for m in msgs if m["role"] == "user"), None)
    um = (m0 or {}).get("content") or ""
    mrepo = re.search(r"<uploaded_files>\s*(/workspace/\S+)\s*</uploaded_files>", um)
    mprob = re.search(r"<issue_description>\s*(.*?)\s*</issue_description>", um, re.S)
    if not mrepo or not mprob:
        return None, "no_task_text"
    norm = Norm(mrepo.group(1))
    problem = mprob.group(1)
    if len(problem) > 6000:
        problem = problem[:6000].rstrip() + "\n..."
    pf = parse_patch(row["model_patch"])
    if not pf:
        return None, "bad_patch"
    src = [f for f in pf if not is_test_path(f["path"])]
    if any(not f["path"].endswith(".py") for f in src):
        return None, "non_py_file"
    if not src or len(src) > 3:
        return None, "no_source_file" if not src else "too_many_files"
    nh = sum(len(f["hunks"]) for f in src)
    if nh == 0 or nh > max_edit_hunks:
        return None, "hunks"
    hunks = {f["path"]: [trim_context(h) for h in f["hunks"]] for f in src}
    for f in src:
        for h in hunks[f["path"]]:
            if len(h["old"]) > max_hunk_lines or len(h["new"]) > max_hunk_lines:
                return None, "big_hunk"
            if f["kind"] == "modify" and (not h["old"] or not "\n".join(h["old"]).strip()):
                return None, "empty_old"
    tree = build_tree(norm, msgs)
    if tree is None:
        return None, "no_tree"

    # ---- index the teacher trajectory
    steps = []   # (msg_index, call, args, observation, thought)
    obs_by_id = {m.get("tool_call_id"): m.get("content") or "" for m in msgs if m["role"] == "tool"}
    last_think = ""
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        txt = clean_text(m.get("content") or "")
        for tc in (m.get("tool_calls") or []):
            a = call_json(tc)
            if a is None:
                continue
            nm = tc["function"]["name"]
            if nm == "think":
                last_think = clean_text(a.get("thought", ""))
                continue
            steps.append((i, nm, a, obs_by_id.get(tc["id"], ""), txt or last_think))
            last_think = ""
    first_edit = {}
    last_edit_idx = -1
    for k, (i, nm, a, obs, th) in enumerate(steps):
        if nm == "str_replace_editor" and a.get("command") in ("str_replace", "insert", "create", "undo_edit"):
            first_edit.setdefault(norm.path(a.get("path", "")), k)
            last_edit_idx = k
    paths = [f["path"] for f in src]
    if any(p not in first_edit for p in paths if next(f for f in src if f["path"] == p)["kind"] == "modify"):
        return None, "edit_not_in_trace"
    first_edit_k = min(first_edit.values())

    ep = []   # list of (thought, name, args, observation)

    # ---- 1. search: first grep/find before the first edit whose output names a patched file
    search = None
    for k, (i, nm, a, obs, th) in enumerate(steps[:first_edit_k]):
        if nm != "execute_bash" or a.get("is_input") in (True, "true"):
            continue
        cmd = norm.text(a.get("command", ""))
        if not re.search(r"\b(grep|rg|find|ag)\b", cmd) or "git " in cmd or len(cmd) > 300 or "\n" in cmd.strip():
            continue
        cmd = re.sub(r"^cd \S+ && ", "", cmd)
        body, code = bash_result(obs, a.get("command", ""))
        body = norm.text(body)
        if any((p in body) or re.search(r"(?<![\w.])" + re.escape(p.split("/")[-1]) + r"(?![\w])", body) for p in paths) and len(body) > 5:
            search = (short_thought(th, "Search for the identifiers from the issue to find where they live."), "run_command", {"command": cmd}, cmd_obs(body, code))
            break
    if search is None:
        return None, "no_search"
    ep.append(search)

    # ---- 2. read each patched (existing) file: teacher's own pre-edit views, merged; else the diff's old-side context
    for f in src:
        p = f["path"]
        if f["kind"] == "add":
            continue
        view = {}
        for k, (i, nm, a, obs, th) in enumerate(steps[:first_edit.get(p, 10 ** 9)]):
            if nm == "str_replace_editor" and a.get("command") == "view" and norm.path(a.get("path", "")) == p:
                view.update(parse_view(obs, norm))
        h0, h1 = hunks[p][0], hunks[p][-1]
        lo, hi = h0["old_start"], h1["old_start"] + len(h1["old"]) - 1
        old_text = {h["old_start"] + j: l for h in hunks[p] for j, l in enumerate(h["old"])}
        ok_view = bool(view) and all(view.get(n) == t for n, t in old_text.items())
        if ok_view:
            a_ = max(min(view), lo - 25)
            b_ = min(max(view), hi + 25, a_ + 149)
            seq = []
            for n in range(a_, b_ + 1):
                if n not in view:
                    break
                seq.append(view[n])
            lines, start = seq, a_
            total = max(view) if len(view) == max(view) else b_ + 40
            thought = next((th for (i, nm, a, obs, th) in steps if nm == "str_replace_editor" and a.get("command") == "view" and norm.path(a.get("path", "")) == p and th), "")
        else:
            # no usable teacher view: the diff's old side is exact pre-edit content, if narrow
            lines, start = [], min(h["old_start"] for h in hunks[p])
            for h in hunks[p]:
                if lines and h["old_start"] > start + len(lines) + 40:
                    return None, "no_view_far_hunks"
                if lines:
                    lines += [""] * max(0, h["old_start"] - (start + len(lines)))
                lines += h["old"]
            total, thought = start + len(lines) + 40, ""
        content = "\n".join(lines)
        if len(content) > 10000:
            return None, "view_too_long"
        thought = short_thought(thought, f"The issue points at `{p}`; let me read the relevant code.")
        ep.append((thought, "read_file", {"filepath": p, "start_line": start, "end_line": start + len(lines) - 1},
                   json.dumps({"status": "ok", "filepath": p, "content": content, "start_line": start, "end_line": start + len(lines) - 1, "total_lines": total, "is_truncated": False})))

    # ---- 3. edits rebuilt from the final patch
    for f in src:
        p = f["path"]
        if f["kind"] == "add":
            body = "\n".join(l for h in f["hunks"] for l in h["new"]) + "\n"
            if len(body) > 6000:
                return None, "big_new_file"
            ep.append((f"Create `{p}`.", "write_file", {"filepath": p, "content": body}, json.dumps({"status": "ok", "filepath": p, "bytes_written": len(body.encode())})))
            continue
        th_edit = next((th for (i, nm, a, obs, th) in steps if nm == "str_replace_editor" and a.get("command") == "str_replace" and norm.path(a.get("path", "")) == p and th), "")
        for j, h in enumerate(hunks[p]):
            old, new = "\n".join(h["old"]), "\n".join(h["new"])
            diff = "--- a/%s\n+++ b/%s\n@@ -%d,%d +%d,%d @@\n%s%s" % (p, p, h["old_start"], len(h["old"]), h["old_start"], len(h["new"]),
                                                                     "".join("-" + l + "\n" for l in h["old"]), "".join("+" + l + "\n" for l in h["new"]))
            ep.append((short_thought(th_edit if j == 0 else "", f"Apply the fix in `{p}`."), "edit_file", {"filepath": p, "old_string": old, "new_string": new},
                       json.dumps({"status": "ok", "filepath": p, "occurrences": 1, "strategy": "exact", "diff": diff, "is_truncated": False})))

    # ---- 4. verify: teacher's last passing pytest run after its last edit
    verify = None
    for k in range(len(steps) - 1, last_edit_idx, -1):
        i, nm, a, obs, th = steps[k]
        if nm != "execute_bash" or a.get("is_input") in (True, "true"):
            continue
        cmd = norm.text(a.get("command", ""))
        cmd = re.sub(r"^cd \S+ && ", "", cmd)
        body, code = bash_result(obs, a.get("command", ""))
        body = norm.text(body)
        if "pytest" in cmd and "\n" not in cmd.strip() and len(cmd) < 300 and code == 0 and re.search(r"\d+ passed", body) and not re.search(r"\d+ (failed|error)", body):
            verify = (short_thought(th, "Run the relevant tests."), "run_command", {"command": cmd}, cmd_obs(body, code, tail=True))
            break
    if verify is None:
        return None, "no_verify"
    ep.append(verify)

    npatch = len("".join(l + "\n" for f in src for h in f["hunks"] for l in h["new"]).encode())
    ep.append(("Tests pass. Submitting the patch.", "submit_patch", {}, json.dumps({"status": "ok", "patch_size": npatch, "files_changed": len(src)})))
    fin = next((clean_text((m.get("tool_calls") and (call_json(m["tool_calls"][0]) or {}).get("message")) or m.get("content") or "", 500) for m in reversed(msgs) if m["role"] == "assistant"
                and any(tc["function"]["name"] == "finish" for tc in (m.get("tool_calls") or []))), "")
    final_text = (fin.split("\n\n")[0][:300].strip() if fin else "") or "The fix is in place and the targeted tests pass."

    # ---- messages in OUR harness layout
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_TEMPLATE.format(repo=row["repo"], problem=problem, tree=tree)}]
    for n, (thought, name, args, obs) in enumerate(ep):
        cid = f"call_{n}"
        messages.append({"role": "assistant", "content": "", "reasoning_content": thought, "tool_calls": [{"type": "function", "id": cid, "function": {"name": name, "arguments": json.dumps(args)}}]})
        messages.append({"role": "tool", "tool_call_id": cid, "content": obs})
    messages.append({"role": "assistant", "content": final_text})
    chars = sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "") + sum(len(t["function"]["arguments"]) for t in m.get("tool_calls", [])) for m in messages)
    return {"trajectory_id": row.get("trajectory_id"), "instance_id": row["instance_id"], "repo": row["repo"], "n_calls": len(ep), "est_tokens": round(chars / 3.3),
            "files": paths, "messages": messages}, None
