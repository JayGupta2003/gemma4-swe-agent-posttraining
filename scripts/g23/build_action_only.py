"""G23 data: the G21 on-policy-distillation rows restricted to ACTION tokens.

G21/G22 showed that reverse KL on all student tokens imported the 31B's self-revising reasoning style ("Wait" 31x more expected by the teacher than written by
the student) and broke E4B under the clock. Here every target position inside a reasoning block (from <|channel> = id 100 through <channel|> = id 101, both
included, so neither entering nor leaving reasoning is trained) is removed from the loss; what remains is the visible output: tool names, arguments, edit text,
submit calls, short content. The reasoning tokens stay in the input as context, so the teacher's action distribution is still conditioned on the student's own
reasoning.

Usage: python scripts/g23/build_action_only.py <opd_sequences.jsonl> <out_dir>
Writes <out_dir>/opd_sequences.jsonl (same schema, read unchanged by scripts/g20/train_opd.py) and <out_dir>/stats.json.
"""
import json
import math
import sys
from pathlib import Path

OPEN, CLOSE = 100, 101
WAIT = {20470, 29202}          # 'Wait', ' Wait' in the Gemma 4 tokenizer


def reasoning_mask(ids):
    inside, s = [False] * len(ids), False
    for j, t in enumerate(ids):
        if t == OPEN:
            s = True
        inside[j] = s
        if t == CLOSE:
            s = False
    return inside


def targets(row):
    ids = row["input_ids"]
    return sorted({j for a, b in row["loss_spans"] for j in range(max(a, 1), min(b, len(ids)))})


def runs(js):
    out = []
    for j in js:
        if out and out[-1][1] == j:
            out[-1][1] = j + 1
        else:
            out.append([j, j + 1])
    return out


def convert(row):
    js = targets(row)
    assert len(js) == row["n_loss_tokens"] == len(row["t_ids"]) == len(row["t_lp"]) == len(row["t_act_lp"]), row["task"]
    inside = reasoning_mask(row["input_ids"])
    keep = [k for k, j in enumerate(js) if not inside[j]]
    if not keep:
        return None
    new = dict(row)
    new["loss_spans"] = runs([js[k] for k in keep])
    for f in ("t_ids", "t_lp", "t_act_lp"):
        new[f] = [row[f][k] for k in keep]
    new["n_loss_tokens"] = len(keep)
    assert targets(new) == [js[k] for k in keep]
    return new


def main(src, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    st = {"rows_in": 0, "rows_out": 0, "loss_tokens_in": 0, "loss_tokens_out": 0, "tokens_out": 0}
    agree = p_sum = wait_student = wait_teacher = 0.0
    with open(src, encoding="utf-8") as f, open(out_dir / "opd_sequences.jsonl", "w", encoding="utf-8") as g:
        for line in f:
            row = json.loads(line)
            st["rows_in"] += 1
            st["loss_tokens_in"] += row["n_loss_tokens"]
            new = convert(row)
            if new is None:
                continue
            st["rows_out"] += 1
            st["loss_tokens_out"] += new["n_loss_tokens"]
            st["tokens_out"] += len(new["input_ids"])
            for j, ti, tl, ta in zip(targets(new), new["t_ids"], new["t_lp"], new["t_act_lp"]):
                y = new["input_ids"][j]
                agree += ti[0] == y
                p_sum += math.exp(ta)
                wait_student += y in WAIT
                wait_teacher += sum(math.exp(l) for t, l in zip(ti, tl) if t in WAIT)
            g.write(json.dumps(new) + "\n")
    n = max(st["loss_tokens_out"], 1)
    st.update({"teacher_top1_agree": round(agree / n, 4), "teacher_p_actual_mean": round(p_sum / n, 4),
               "wait_per_10k_student": round(wait_student / n * 1e4, 2), "wait_per_10k_teacher_expected": round(wait_teacher / n * 1e4, 2)})
    (out_dir / "stats.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(json.dumps(st, indent=1))
    return st


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
