"""Unit test of the G24 per-task cap: at most 2 attempts per task, fewest generated tokens first, all segments of a kept attempt kept."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_rft_data import cap, cap_repos  # noqa: E402


def seq(task, run, rnd, loss, seg=0):
    return {"task": task, "run": run, "round": rnd, "n_loss_tokens": loss, "n_tokens": 10 * loss, "seg": seg}


data = [
    seq("A", "g20", "e1_0", 500),
    seq("A", "g24a", "r2_3", 300), seq("A", "g24a", "r2_3", 50, seg=1),   # 2 segments, 350 total
    seq("A", "g24a", "r3_1", 900),
    seq("A", "g24a", "r4_0", 200),
    seq("B", "g24a", "r3_2", 700),
]
kept = cap(data)
got = sorted((s["task"], s["run"], s["round"], s["seg"]) for s in kept)
assert got == [("A", "g24a", "r2_3", 0), ("A", "g24a", "r2_3", 1), ("A", "g24a", "r4_0", 0), ("B", "g24a", "r3_2", 0)], got
assert cap(data, 1) == [data[4], data[5]], cap(data, 1)
repo_of = {"A": "o/x", "B": "o/x", "C": "o/x", "D": "o/y"}
many = [seq(t, "g25b", "r1_0", 10) for t in "ABCD"] + [seq("A", "g25b", "r1_0", 5, seg=1)]
k2 = cap_repos(many, repo_of, 2)
assert len({s["task"] for s in k2 if repo_of[s["task"]] == "o/x"}) == 2 and any(s["task"] == "D" for s in k2), k2
assert all(len([s for s in k2 if s["task"] == "A"]) in (0, 2) for _ in [0])        # segments of a kept task stay together
assert cap_repos(many, repo_of, 2) == k2 and cap_repos(many, repo_of, 0) == many  # deterministic; 0 = off
print("test_build_rft_data: OK")
