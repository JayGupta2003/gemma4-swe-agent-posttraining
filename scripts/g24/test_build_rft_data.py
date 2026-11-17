"""Unit test of the G24 per-task cap: at most 2 attempts per task, fewest generated tokens first, all segments of a kept attempt kept."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_rft_data import cap  # noqa: E402


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
print("test_build_rft_data: OK")
