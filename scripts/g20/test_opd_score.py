"""Tests for scripts/g20/opd_score.py on synthetic captures and synthetic vLLM responses (no GPU). Run: python scripts/g20/test_opd_score.py"""

import gzip
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opd_score as O  # noqa: E402


def rec(rnd, key, sysm, p, c):
    return {"round": rnd, "task_key": key, "t0": 1.0, "request": {"messages": [{"role": "system", "content": sysm}, {"role": "user", "content": "task"}]},
            "response": {"prompt_token_ids": p, "choices": [{"token_ids": c}]}}


def write(path, recs):
    with open(path, "wb") as f:
        for r in recs:
            f.write(gzip.compress((json.dumps(r) + "\n").encode()))


def test_leaves():
    d = Path(tempfile.mkdtemp())
    cap = d / "c.jsonl.gz"
    main = O.MAIN + " ..."
    write(cap, [rec("e1_0", "A", main, [1, 2], [3]), rec("e1_0", "A", main, [1, 2, 3, 4], [5, 6]), rec("e1_1", "B", [{"type": "text", "text": main}], [9], [8]),
                rec("e1_0", "S", "You are a code analyzer", [1, 2, 3, 4, 5, 6, 7], [1]), rec("smoke", "C", main, [1, 1, 1, 1, 1, 1], [1]),
                rec("e1_0", "D", main, [], [1])])
    ls = O.leaves(str(cap), "e1_")
    assert [l["task_key"] for l in ls] == ["B", "A"], ls            # sub-agent, other rounds and id-less records excluded; shortest first
    assert ls[1]["ids"] == [1, 2, 3, 4, 5, 6] and ls[1]["n_prompt"] == 4


def plp_for(ids, k, drop_actual_at=None):
    out = [None]
    for i in range(1, len(ids)):
        d = {str(1000 + j): {"logprob": -0.1 * (j + 1), "rank": j + 1, "decoded_token": "x"} for j in range(k)}
        if i != drop_actual_at:
            d[str(ids[i])] = {"logprob": -5.0, "rank": 99, "decoded_token": "y"}
        out.append(d)
    return out


def test_parse():
    ids = [7, 8, 9, 1001]
    act, ti, tl = O.parse_prompt_logprobs(plp_for(ids, 3), ids, 3)
    assert math.isnan(act[0]) and act[1] == -5.0 and act[3] == -5.0   # at position 3 the actual token (1001) shares its key with a top entry; the dict holds one value
    assert ti[1].tolist() == [1000, 1001, 1002] and abs(float(tl[1, 0]) + 0.1) < 1e-3 and ti[0].tolist() == [-1, -1, -1]
    try:
        O.parse_prompt_logprobs(plp_for(ids, 3)[:3], ids, 3); raise AssertionError("length mismatch not caught")
    except ValueError:
        pass
    try:
        O.parse_prompt_logprobs(plp_for([7, 8, 9], 3, drop_actual_at=2), [7, 8, 9], 3); raise AssertionError("missing actual token not caught")
    except ValueError:
        pass
    pl = plp_for([7, 8], 2)
    pl[1] = {"token_id:1000": -0.3, "token_id:8": -2.0}                 # tokens-as-ids keys and bare floats
    act, ti, _ = O.parse_prompt_logprobs(pl, [7, 8], 2)
    assert act[1] == -2.0 and ti[1].tolist() == [1000, 8]


def test_store_roundtrip():
    d = Path(tempfile.mkdtemp())
    st = O.Store(3)
    for key, ids, n_prompt in (("A", [7, 8, 9, 1001], 2), ("B", [5, 6, 7], 1)):
        st.add({"task_key": key, "round": "e1_0", "t0": 1.0, "n_prompt": n_prompt, "ids": ids}, *O.parse_prompt_logprobs(plp_for(ids, 3), ids, 3), 0.5)
    st.save(d / "t.npz", d / "m.json")
    z = np.load(d / "t.npz")
    assert z["leaf_offsets"].tolist() == [0, 4, 7] and z["ids"].tolist() == [7, 8, 9, 1001, 5, 6, 7] and z["top_ids"].shape == (7, 3) and z["n_prompt"].tolist() == [2, 1]
    assert json.loads((d / "m.json").read_text())["leaves"][1]["task_key"] == "B"
    s = O.summary(st)
    assert s["leaves"] == 2 and s["tokens"] == 7 and s["mean_teacher_lp_last_completion"] is not None


def test_score_all():
    leaves = [{"task_key": f"L{i}", "round": "e1_0", "t0": 1.0, "n_prompt": 1, "ids": [5, 6, 7 + i]} for i in range(25)]

    def job(leaf):
        if leaf["task_key"] == "L3":
            raise RuntimeError("HTTP 400: too long")
        return (*O.parse_prompt_logprobs(plp_for(leaf["ids"], 2), leaf["ids"], 2), 0.01)

    st, calls = O.Store(2), []
    n_ok, errs = O.score_all(leaves, job, st, lambda: False, lambda: calls.append(len(st.rows)), workers=3, every=10)
    assert n_ok == 24 and len(errs) == 1 and errs[0].startswith("L3") and len(st.rows) == 24 and len(calls) == 3   # progress at 10, 20 and the end
    st2, n = O.Store(2), {"k": 0}

    def out_of_time():
        n["k"] += 1
        return n["k"] > 5                                                   # time runs out after a few submissions: no new leaves are started
    n_ok2, _ = O.score_all(leaves, job, st2, out_of_time, lambda: None, workers=2)
    assert 0 < n_ok2 < 25


if __name__ == "__main__":
    for t in (test_leaves, test_parse, test_store_roundtrip, test_score_all):
        t(); print("ok", t.__name__)
