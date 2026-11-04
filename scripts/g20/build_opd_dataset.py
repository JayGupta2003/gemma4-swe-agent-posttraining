"""Build the on-policy distillation dataset: G20 student capture + G20b teacher scores -> rows for scripts/g20/train_opd.py. Offline, no GPU.

One row per main-agent context segment (capture task_key), for EVERY attempt (solved or not: the teacher signal does not need task success):
    input_ids  = the leaf scored by the teacher (prompt ids + completion ids of the segment's last request; asserted identical to the stored teacher ids)
    loss_spans = the student's own turns: each earlier assistant turn located in the final prompt (the tokens after the previous request's prompt up to and
                 including its stop token, the logic of scripts/g10/build_sft_general.py) + the final completion; a segment whose request chain does not extend
                 token-exactly keeps only its final completion
    t_ids/t_lp = the teacher's top-k token ids and log-probs at every target position j inside the spans, in increasing j (the order train_opd.py uses):
                 the distribution over token j given tokens < j, i.e. what the student predicts from position j-1.
Usage: python scripts/g20/build_opd_dataset.py <capture.jsonl.gz> <teacher_scores.npz> <teacher_meta.json> <out_dir> [round_prefix=e1_]
"""

import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opd_score as O  # noqa: E402


def segments(capture, prefix):
    seg = collections.defaultdict(list)
    for rec in O.read_records(capture):
        if not str(rec.get("round", "")).startswith(prefix):
            continue
        msgs = rec["request"].get("messages", [])
        if not O._text(next((m.get("content") for m in msgs if m.get("role") == "system"), "")).startswith(O.MAIN):
            continue
        resp = rec.get("response") or {}
        p, c = resp.get("prompt_token_ids"), ((resp.get("choices") or [{}])[0]).get("token_ids")
        if p and c:
            seg[rec["task_key"]].append((rec.get("t0") or 0.0, list(p), list(c)))
    return seg


def spans_of(recs):
    """(ids, spans, chain_ok) for one segment; recs = [(t0, prompt_ids, completion_ids)]."""
    recs = sorted(recs, key=lambda r: (len(r[1]), r[0]))
    spans, ok = [], True
    for (_, pa, ca), (_, pb, _) in zip(recs, recs[1:]):
        if pb[:len(pa)] != pa or ca[-1] not in pb[len(pa):]:
            ok = False
            break
        spans.append((len(pa), len(pa) + pb[len(pa):].index(ca[-1]) + 1))
    last_p, last_c = recs[-1][1], recs[-1][2]
    ids = last_p + last_c
    if not ok:
        spans = []
    spans.append((len(last_p), len(ids)))
    return ids, spans, ok


def build(capture, npz, meta_path, prefix="e1_"):
    z = np.load(npz)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    off = z["leaf_offsets"]
    teacher = {m["task_key"]: k for k, m in enumerate(meta["leaves"])}
    rows, why = [], collections.Counter()
    for key, recs in segments(capture, prefix).items():
        if key not in teacher:
            why["no teacher scores"] += 1
            continue
        k = teacher[key]
        a, b = int(off[k]), int(off[k + 1])
        ids, spans, ok = spans_of(recs)
        if z["ids"][a:b].tolist() != ids:
            why["teacher ids differ from the segment"] += 1
            continue
        why["chain ok" if ok else "chain broken: final completion only"] += 1
        pos = sorted({j for s, e in spans for j in range(max(s, 1), min(e, len(ids)))})
        rows.append({"task": key, "round": meta["leaves"][k]["round"], "input_ids": ids, "loss_spans": spans, "n_tokens": len(ids), "n_loss_tokens": len(pos),
                     "t_ids": z["top_ids"][a:b][pos].tolist(), "t_lp": [[round(float(x), 4) for x in r] for r in z["top_lp"][a:b][pos].astype(np.float32)],
                     "t_act_lp": [round(float(x), 4) for x in z["act_lp"][a:b][pos]]})
    return rows, why


if __name__ == "__main__":
    cap, npz, meta_path, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    prefix = sys.argv[5] if len(sys.argv) > 5 else "e1_"
    out.mkdir(parents=True, exist_ok=True)
    rows, why = build(cap, npz, meta_path, prefix)
    with open(out / "opd_sequences.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    nt, nl = [r["n_tokens"] for r in rows], [r["n_loss_tokens"] for r in rows]
    act = np.concatenate([np.array(r["t_act_lp"], np.float32) for r in rows])
    top1 = np.concatenate([np.array([t[0] for t in r["t_ids"]]) == np.array([r["input_ids"][j] for j in sorted({j for s, e in r["loss_spans"] for j in range(max(s, 1), min(e, r["n_tokens"]))})]) for r in rows])
    stats = {"segments": dict(why), "rows": len(rows), "tokens_total_m": round(sum(nt) / 1e6, 3), "tokens_median": statistics.median(nt), "tokens_max": max(nt),
             "loss_tokens_total_k": round(sum(nl) / 1e3, 1), "loss_share": round(sum(nl) / sum(nt), 4), "loss_tokens_median": statistics.median(nl),
             "teacher_mean_lp_on_student_tokens": round(float(act.mean()), 4), "teacher_top1_agrees": round(float(top1.mean()), 4),
             "teacher_p_below_0.1": round(float((act < np.log(0.1)).mean()), 4), "est_train_h_at_375_toks": round(sum(nt) / 375 / 3600, 2)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))
