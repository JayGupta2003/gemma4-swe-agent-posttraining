"""Tests for scripts/g18/teacher_convert.py on a small synthetic OpenHands trajectory.  Run: python -m unittest scripts/g18/test_teacher_convert.py"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teacher_convert as TC  # noqa: E402

D = "/workspace/acme__lib__1.0"
SRC_LINES = ["import os", "", "def add(a, b):", "    return a - b", "", "def sub(a, b):", "    return a - b", ""]
PATCH = """diff --git a/lib/calc.py b/lib/calc.py
--- a/lib/calc.py
+++ b/lib/calc.py
@@ -1,7 +1,7 @@
 import os

 def add(a, b):
-    return a - b
+    return a + b

 def sub(a, b):
     return a - b
"""
TRAILER = "\n[The command completed with exit code 0.]\n[Current working directory: %s]\n[Python interpreter: /usr/bin/python]\n[Command finished with exit code 0]" % D


def call(i, name, args):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def tool(i, content):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": content}


def view_obs(lines):
    return "Here's the result of running `cat -n` on %s/lib/calc.py:\n" % D + "\n".join("%6d\t%s" % (n + 1, t) for n, t in enumerate(lines))


def make_row(**over):
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "<uploaded_files>\n%s\n</uploaded_files>\n\nIssue:\n<issue_description>\nadd() subtracts instead of adding\n</issue_description>\n" % D},
        call(0, "str_replace_editor", {"command": "view", "path": "/workspace"}),
        tool(0, "Here's the files and directories up to 2 levels deep in /workspace, excluding hidden items:\n/workspace/\n%s/\n%s/README.md\n%s/setup.py\n%s/lib/\n%s/lib/calc.py\n%s/tests/\n%s/tests/test_calc.py" % ((D,) * 7)),
        call(1, "execute_bash", {"command": "cd %s && grep -rn \"def add\" ." % D}),
        tool(1, "./lib/calc.py:3:def add(a, b):" + TRAILER),
        call(2, "str_replace_editor", {"command": "view", "path": D + "/lib/calc.py"}),
        tool(2, view_obs(SRC_LINES)),
        call(3, "str_replace_editor", {"command": "str_replace", "path": D + "/lib/calc.py", "old_str": "def add(a, b):\n    return a - b", "new_str": "def add(a, b):\n    return a + b"}),
        tool(3, "The file has been edited."),
        call(4, "execute_bash", {"command": "cd %s && python -m pytest tests/test_calc.py -q" % D}),
        tool(4, "..\n2 passed in 0.01s" + TRAILER),
        call(5, "finish", {"message": "Fixed add()."}),
        tool(5, ""),
    ]
    row = {"trajectory_id": "t", "instance_id": "acme__lib-1", "repo": "acme/lib", "resolved": 1, "model_patch": PATCH, "trajectory": msgs}
    row.update(over)
    return row


class T(unittest.TestCase):
    def test_happy_path_shape(self):
        ep, why = TC.convert(make_row())
        self.assertIsNone(why)
        names = [m["tool_calls"][0]["function"]["name"] for m in ep["messages"] if m.get("tool_calls")]
        self.assertEqual(names, ["run_command", "read_file", "edit_file", "run_command", "submit_patch"])
        roles = [m["role"] for m in ep["messages"]]
        self.assertEqual(roles[:2], ["system", "user"])
        self.assertEqual(roles[-1], "assistant")
        for a, b in zip(roles[2:-1:2], roles[3:-1:2]):
            self.assertEqual((a, b), ("assistant", "tool"))

    def test_paths_are_workspace_relative_and_no_leak(self):
        ep, _ = TC.convert(make_row())
        blob = json.dumps(ep["messages"][2:])
        self.assertNotIn("/workspace", blob)
        self.assertNotIn("acme__lib__1.0", blob)
        self.assertIn("lib/calc.py", blob)

    def test_edit_is_rebuilt_from_patch_and_applies(self):
        ep, _ = TC.convert(make_row())
        msgs = ep["messages"]
        edit = next(m for m in msgs if m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "edit_file")
        a = json.loads(edit["tool_calls"][0]["function"]["arguments"])
        src = "\n".join(SRC_LINES)
        self.assertEqual(src.count(a["old_string"]), 1)
        fixed = src.replace(a["old_string"], a["new_string"])
        self.assertIn("return a + b", fixed)
        self.assertEqual(fixed.count("return a - b"), 1)   # only add() was fixed, sub() untouched

    def test_read_file_content_matches_teacher_view_lines_around_the_hunk(self):
        ep, _ = TC.convert(make_row())
        rd = next(m for m in ep["messages"] if m["role"] == "tool" and '"read_file"' not in m["content"] and '"start_line"' in m["content"])
        d = json.loads(rd["content"])
        self.assertEqual(d["content"].splitlines()[4 - d["start_line"]], "    return a - b")
        self.assertEqual(d["status"], "ok")

    def test_search_output_has_trailer_stripped(self):
        ep, _ = TC.convert(make_row())
        first_tool = next(m for m in ep["messages"] if m["role"] == "tool")
        d = json.loads(first_tool["content"])
        self.assertNotIn("Current working directory", d["stdout"])
        self.assertEqual(d["exit_code"], 0)

    def test_drops(self):
        r = make_row(resolved=0)
        self.assertEqual(TC.convert(r)[1], "not_resolved")
        r = make_row(model_patch=PATCH.replace("lib/calc.py", "lib/calc.md"))
        self.assertEqual(TC.convert(r)[1], "non_py_file")
        r = make_row(model_patch=PATCH.replace("lib/calc.py", "tests/test_calc.py"))
        self.assertEqual(TC.convert(r)[1], "no_source_file")
        # no passing pytest after the last edit
        row = make_row()
        row["trajectory"][11]["content"] = "1 failed in 0.1s" + TRAILER
        self.assertEqual(TC.convert(row)[1], "no_verify")

    def test_is_test_path(self):
        for p in ("tests/a.py", "pkg/tests/x.py", "test_a.py", "a_test.py", "conftest.py", "pkg/testing/a.py"):
            self.assertTrue(TC.is_test_path(p), p)
        for p in ("pkg/a.py", "pkg/contest.py", "latest/a.py"):
            self.assertFalse(TC.is_test_path(p), p)

    def test_short_thought(self):
        self.assertEqual(TC.short_thought("Let me look at x.", "fb"), "Let me look at x.")
        self.assertEqual(TC.short_thought("## Phase 3 start", "fb"), "fb")
        self.assertEqual(TC.short_thought("x" * 500, "fb"), "fb")
        self.assertEqual(TC.short_thought("", "fb"), "fb")

    def test_trim_context_keeps_two_lines(self):
        h = {"old_start": 10, "old": ["a", "b", "c", "X", "d", "e", "f"], "new": ["a", "b", "c", "Y", "d", "e", "f"]}
        t = TC.trim_context(h)
        self.assertEqual((t["old"], t["new"], t["old_start"]), (["b", "c", "X", "d", "e"], ["b", "c", "Y", "d", "e"], 11))


if __name__ == "__main__":
    unittest.main()
