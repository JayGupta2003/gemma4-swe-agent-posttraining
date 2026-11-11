"""Tests for scripts/g23/build_action_only.py (no GPU): reasoning positions (<|channel> .. <channel|> inclusive) leave the loss, teacher arrays stay aligned."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import build_action_only as B

# prompt 0-2 | turn 1: <|channel> r r <channel|> a a | user 9 9 | turn 2: a a <|channel> r <channel|> a
ids = [5, 6, 7, 100, 11, 12, 101, 21, 22, 9, 9, 23, 24, 100, 13, 101, 25]
spans = [[3, 9], [11, 17]]
js = B.targets({"input_ids": ids, "loss_spans": spans})
row = {"task": "t", "input_ids": ids, "loss_spans": spans, "n_loss_tokens": len(js),
       "t_ids": [[j, 0] for j in js], "t_lp": [[-0.1, -3.0] for _ in js], "t_act_lp": [-0.1 * j for j in js]}
new = B.convert(row)
want = [7, 8, 11, 12, 16]                        # the action tokens 21 22 23 24 25
assert B.targets(new) == want, B.targets(new)
assert [ids[j] for j in want] == [21, 22, 23, 24, 25]
assert [t[0] for t in new["t_ids"]] == want and new["t_act_lp"] == [-0.1 * j for j in want] and new["n_loss_tokens"] == 5
assert new["loss_spans"] == [[7, 9], [11, 13], [16, 17]]
assert B.convert({**row, "loss_spans": [[3, 7]], "n_loss_tokens": 4, "t_ids": row["t_ids"][:4], "t_lp": row["t_lp"][:4], "t_act_lp": row["t_act_lp"][:4]}) is None
print("test_build_action_only: OK")
