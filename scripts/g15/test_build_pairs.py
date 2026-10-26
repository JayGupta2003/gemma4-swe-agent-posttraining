"""Unit tests for scripts/g15/build_pairs.py on synthetic captures (CPU, seconds). Run: python -m unittest scripts/g15/test_build_pairs.py -v"""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pairs as B  # noqa: E402

STOP = 99
SYS = B.MAIN_PREFIX + " assigned to resolve an issue."


def rec(rnd, key, t0, t1, user, prompt, comp, system=SYS, content=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return {"round": rnd, "task_key": key, "t0": t0, "t1": t1, "request": {"messages": msgs},
            "response": {"prompt_token_ids": prompt, "choices": [{"message": {"content": content}, "token_ids": comp}]}}


def two_turn(rnd, key, user, t0=0.0, base=(1, 2, 3)):
    """Two requests of one segment: prompt2 = prompt1 + completion1 + tool tokens."""
    p1, c1 = list(base), [10, 11, STOP]
    p2, c2 = p1 + c1 + [20, 21], [12, STOP]
    return [rec(rnd, key, t0, t0 + 2.0, user, p1, c1), rec(rnd, key, t0 + 3.0, t0 + 5.0, user, p2, c2)]


def write_run(root: Path, rnd, records, results):
    (root / rnd / "patches").mkdir(parents=True, exist_ok=True)
    with gzip.open(root / "capture.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(root / rnd / "task_results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def row(iid, resolved, seconds, error=None):
    return {"instance_id": iid, "resolved": resolved, "duration_seconds": seconds, "error": error}


class SegmentTests(unittest.TestCase):
    def test_spans_are_the_generated_tokens(self):
        recs = [(r["t0"], r["t1"], r["response"]["prompt_token_ids"], r["response"]["choices"][0]["token_ids"]) for r in two_turn("p1", "k", "u")]
        ids, spans, why = B.segment_sequence(recs, 100)
        self.assertIsNone(why)
        self.assertEqual(ids, [1, 2, 3, 10, 11, STOP, 20, 21, 12, STOP])
        self.assertEqual(spans, [(3, 6), (8, 10)])                       # first completion inside the final prompt (incl. its stop token), then the final completion
        self.assertEqual([ids[s:e] for s, e in spans], [[10, 11, STOP], [12, STOP]])

    def test_single_request_segment(self):
        ids, spans, why = B.segment_sequence([(0.0, 1.0, [1, 2], [5, STOP])], 100)
        self.assertEqual((ids, spans, why), ([1, 2, 5, STOP], [(2, 4)], None))

    def test_requests_are_ordered_by_time(self):
        a, b = [(r["t0"], r["t1"], r["response"]["prompt_token_ids"], r["response"]["choices"][0]["token_ids"]) for r in two_turn("p1", "k", "u")]
        self.assertEqual(B.segment_sequence([b, a], 100)[0], B.segment_sequence([a, b], 100)[0])

    def test_history_must_be_token_exact(self):
        r1, r2 = [(r["t0"], r["t1"], r["response"]["prompt_token_ids"], r["response"]["choices"][0]["token_ids"]) for r in two_turn("p1", "k", "u")]
        bad = (r2[0], r2[1], [1, 2, 4, 10, 11, STOP, 20, 21], r2[3])      # prompt no longer extends the first prompt
        self.assertEqual(B.segment_sequence([r1, bad], 100)[2], "malformed history")
        no_stop = (r1[0], r1[1], r1[2], [10, 15])                          # the first completion's last token (15) is not found in the next prompt's new part: not token-exact
        self.assertEqual(B.segment_sequence([no_stop, r2], 100)[2], "malformed history")

    def test_length_limit(self):
        recs = [(0.0, 1.0, list(range(50)), [5, STOP])]
        self.assertIn("longer than 10", B.segment_sequence(recs, 10)[2])
        self.assertIsNone(B.segment_sequence(recs, 52)[2])


class AttributionTests(unittest.TestCase):
    PS = {"t_a": "Fix the table caption bug " + "x" * 300, "t_b": "Handle empty header in the CLI " + "y" * 300}

    def tasks(self):
        return [{"instance_id": k, "problem_statement": v} for k, v in self.PS.items()]

    def test_compaction_continuation_is_one_attempt_and_the_last_segment_wins(self):
        idx = B.TaskIndex(self.tasks())
        recs = two_turn("p1", "keyA", "Issue: " + self.PS["t_a"], t0=0.0)
        summary = rec("p1", "sumA", 6.0, 9.0, "Please summarize: " + self.PS["t_a"], [7], [8, STOP], system="", content="SUMMARY OF WORK SO FAR: " + "s" * 200)
        recs += [summary]
        recs += two_turn("p1", "keyB", "Context so far -- SUMMARY OF WORK SO FAR: " + "s" * 200, t0=10.0, base=(1, 2, 8))
        recs += two_turn("p1", "keyC", "Issue: " + self.PS["t_b"], t0=20.0)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_run(root, "p1", recs, [row("t_a", True, 40.0), row("t_b", False, 300.0, "Agent exceeded session timeout (5.0 min)")])
            att, drop = B.build_attempts(root, "p1", idx, 100)
        self.assertEqual(sorted(att), ["t_a", "t_b"])
        self.assertEqual(att["t_a"]["n_segments"], 2)
        self.assertEqual(att["t_a"]["ids"][:3], [1, 2, 8])                 # the final segment is the post-compaction one
        self.assertEqual((att["t_a"]["cls"], att["t_b"]["cls"]), ("S", "C"))
        self.assertAlmostEqual(att["t_a"]["model_seconds"], 2 + 2 + 3 + 2 + 2, places=3)   # 4 main requests (2 s each) + the summary request (3 s)

    def test_classes(self):
        self.assertEqual(B.classify(row("a", True, 1)), "S")
        self.assertEqual(B.classify(row("a", False, 1)), "F")
        self.assertEqual(B.classify(row("a", False, 1, "Agent exceeded session timeout (5.0 min)")), "C")
        self.assertEqual(B.classify(row("a", False, 1, "Agent exceeded tool call budget (28 calls)")), "F")


def att(task, rnd, cls, seconds, unusable=None, touches_tests=False):
    return {"task": task, "round": rnd, "cls": cls, "seconds": seconds, "model_seconds": seconds, "n_segments": 1, "n_requests": 1, "ids": [1, 2, STOP], "spans": [(2, 3)],
            "unusable": unusable, "patch_known": True, "touches_tests": touches_tests, "patch_files": []}


class PairTests(unittest.TestCase):
    def test_fastest_success_against_slowest_failure(self):
        by = {"t": [att("t", "p1", "S", 200), att("t", "p2", "S", 90), att("t", "p3", "C", 300), att("t", "p4", "F", 120)]}
        pairs, info = B.pick_pairs(by, min_main_pairs=0)
        self.assertEqual([(p["chosen"]["round"], p["rejected"]["round"], p["kind"]) for p in pairs], [("p2", "p3", "s_vs_c"), ("p1", "p4", "s_vs_f")])

    def test_at_most_two_pairs_and_each_attempt_once(self):
        by = {"t": [att("t", f"p{i}", "S", 50 + i) for i in range(1, 4)] + [att("t", f"q{i}", "C", 300) for i in range(1, 4)]}
        pairs, _ = B.pick_pairs(by, min_main_pairs=0)
        self.assertEqual(len(pairs), 2)
        used = [p["chosen"]["round"] for p in pairs] + [p["rejected"]["round"] for p in pairs]
        self.assertEqual(len(used), len(set(used)))

    def test_no_pair_without_contrast(self):
        by = {"all_s": [att("all_s", "p1", "S", 50), att("all_s", "p2", "S", 55)], "all_c": [att("all_c", "p1", "C", 300), att("all_c", "p2", "C", 300)]}
        pairs, info = B.pick_pairs(by, min_main_pairs=0)
        self.assertEqual(pairs, [])
        self.assertEqual(info["task without a contrast"], 2)

    def test_unusable_attempts_and_test_touching_successes_are_excluded(self):
        by = {"t": [att("t", "p1", "S", 50, unusable="malformed history"), att("t", "p2", "C", 300)],
              "u": [att("u", "p1", "S", 50, touches_tests=True), att("u", "p2", "F", 100)]}
        pairs, info = B.pick_pairs(by, min_main_pairs=0)
        self.assertEqual(pairs, [])
        self.assertEqual(info["S attempts dropped: patch touches test files"], 1)

    def test_within_s_pairs_only_below_the_main_count_and_with_30_percent_gap(self):
        by = {"fast": [att("fast", "p1", "S", 60), att("fast", "p2", "S", 100)], "close": [att("close", "p1", "S", 90), att("close", "p2", "S", 100)],
              "mixed": [att("mixed", "p1", "S", 70), att("mixed", "p2", "C", 300)]}
        pairs, info = B.pick_pairs(by, min_main_pairs=40)
        self.assertEqual(sorted((p["task"], p["kind"]) for p in pairs), [("fast", "within_s"), ("mixed", "s_vs_c")])
        pairs, info = B.pick_pairs(by, min_main_pairs=1)                 # enough main pairs: no within-S pairs
        self.assertEqual([p["kind"] for p in pairs], ["s_vs_c"])

    def test_single_success_is_not_reused_but_counted(self):
        by = {"t": [att("t", "p1", "S", 50), att("t", "p2", "C", 300), att("t", "p3", "F", 100)]}
        pairs, info = B.pick_pairs(by, min_main_pairs=0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(info["extra pairs if the single S were reused"], 1)


class EndToEndTests(unittest.TestCase):
    PS = {f"t{i}": f"Problem number {i}: " + chr(97 + i) * 300 for i in range(3)}

    def run_dir(self, root, rnd, outcomes):
        recs, results = [], []
        for i, (cls, sec) in enumerate(outcomes):
            iid = f"t{i}"
            recs += two_turn(rnd, f"{rnd}-{i}", "Issue: " + self.PS[iid], t0=100.0 * i)
            results.append(row(iid, cls == "S", sec, "Agent exceeded session timeout (5.0 min)" if cls == "C" else None))
        write_run(root, rnd, recs, results)

    def test_main_writes_pairs_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.run_dir(d / "r1", "p1", [("S", 100), ("C", 300), ("F", 150)])
            self.run_dir(d / "r2", "p2", [("C", 300), ("S", 80), ("F", 160)])
            (d / "tasks.jsonl").write_text("\n".join(json.dumps({"instance_id": k, "problem_statement": v}) for k, v in self.PS.items()), encoding="utf-8")
            stats = B.main(["--tasks", str(d / "tasks.jsonl"), "--out", str(d / "out"), "--run", f"{d / 'r1'}:p1", "--run", f"{d / 'r2'}:p2", "--min-main-pairs", "0"])
            pairs = [json.loads(l) for l in open(d / "out" / "pairs.jsonl", encoding="utf-8")]
        self.assertEqual(stats["pairs"], 2)
        self.assertEqual(sorted((p["task"], p["kind"], p["chosen"]["round"], p["rejected"]["round"]) for p in pairs), [("t0", "s_vs_c", "p1", "p2"), ("t1", "s_vs_c", "p2", "p1")])
        self.assertEqual(stats["attempt_classes"], {"S": 2, "C": 2, "F": 2})
        self.assertEqual(pairs[0]["chosen"]["loss_spans"], [[3, 6], [8, 10]])
        self.assertEqual(stats["gate_main_pairs_ge_40"], False)

    def test_exclude_guard_rejects_competition_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.run_dir(d / "r1", "p1", [("S", 100), ("C", 300), ("F", 150)])
            (d / "tasks.jsonl").write_text("\n".join(json.dumps({"instance_id": k, "problem_statement": v}) for k, v in self.PS.items()), encoding="utf-8")
            (d / "excl.json").write_text(json.dumps(["t1"]), encoding="utf-8")
            with self.assertRaises(AssertionError):
                B.main(["--tasks", str(d / "tasks.jsonl"), "--out", str(d / "out"), "--run", f"{d / 'r1'}:p1", "--exclude", str(d / "excl.json")])


if __name__ == "__main__":
    unittest.main()
