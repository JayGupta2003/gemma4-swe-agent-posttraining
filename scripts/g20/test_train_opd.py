"""Tests for the OPD loss and trainer (run with the Kaggle-mirroring venv: .venv_g1/Scripts/python.exe scripts/g20/test_train_opd.py).

1. opd_loss.coarse_kl equals a brute-force coarse-grained KL (rkl and fkl), is a lower bound of the full-vocabulary KL, is ~0 when the teacher's top-k equals the
   student's own log-probs, and gradient descent on it drives the student to the teacher.
2. End to end on a tiny random Gemma 4 (plain and E4B-like with per-layer inputs + shared KV): with a teacher equal to the student itself (top-k computed from the
   same deterministic tiny model at the trainer's positions) the first-step loss is ~0 -- an off-by-one in position/teacher alignment would make it large; with a
   different teacher the loss is clearly positive and falls over training, and the adapter is saved.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import opd_loss as OL  # noqa: E402


def brute(logits, t_ids, t_lp, mode):
    out = []
    for i in range(logits.shape[0]):
        q = torch.softmax(logits[i].double(), -1)
        ids = [int(t) for t in t_ids[i] if t >= 0]
        p_top = torch.tensor([math.exp(float(x)) for t, x in zip(t_ids[i], t_lp[i]) if t >= 0], dtype=torch.double)
        qc = torch.cat([q[ids], (1 - q[ids].sum()).clamp_min(1e-6).view(1)])
        pc = torch.cat([p_top, (1 - p_top.sum()).clamp_min(1e-6).view(1)])
        out.append(float((qc * (qc.log() - pc.log())).sum() if mode == "rkl" else (pc * (pc.log() - qc.log())).sum()))
    return out


def test_loss_unit():
    g = torch.Generator().manual_seed(0)
    V, n, k = 50, 6, 8
    logits = torch.randn(n, V, generator=g) * 2
    teacher_full = torch.log_softmax(torch.randn(n, V, generator=g) * 2, -1)
    t_lp, t_ids = teacher_full.topk(k, -1)
    t_ids[2, 5:] = -1                                     # padding inside a row
    for mode in ("rkl", "fkl"):
        got = OL.coarse_kl(logits, t_ids, t_lp, mode).tolist()
        ref = brute(logits, t_ids, t_lp, mode)
        assert all(abs(a - b) < 1e-4 for a, b in zip(got, ref)), (mode, got, ref)
    full_rkl = (torch.softmax(logits, -1) * (torch.log_softmax(logits, -1) - teacher_full)).sum(-1)
    t_ids2 = teacher_full.topk(k, -1).indices
    assert (OL.coarse_kl(logits, t_ids2, t_lp, "rkl") <= full_rkl + 1e-5).all()          # data-processing inequality
    own_lp, own_ids = torch.log_softmax(logits, -1).topk(k, -1)
    assert OL.coarse_kl(logits, own_ids, own_lp, "rkl").abs().max() < 1e-5               # teacher == student -> 0
    x = logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.1)
    l0 = None
    for _ in range(300):
        loss = OL.coarse_kl(x, t_ids2, t_lp, "rkl").mean()
        l0 = l0 if l0 is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.02 * l0, (l0, float(loss))


def make_rows(rng, n_seqs):
    rows = []
    for i in range(n_seqs):
        n = rng.randint(600, 1200)
        ids = [rng.randint(5, 990) for _ in range(n)]
        spans = [[100, 180], [400, 450], [n - 120, n]]
        rows.append({"task": f"t{i}", "input_ids": ids, "loss_spans": spans})
    return rows


def positions(r):
    return sorted({j for s, e in r["loss_spans"] for j in range(max(s, 1), min(e, len(r["input_ids"])))})


def run_trainer(rows, tmp, extra):
    (tmp / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    env = dict(os.environ, G3_MODEL="x", G3_DATA=str(tmp / "d.jsonl"), G3_OUT=str(tmp / "out"), G3_TINY="1", G3_ACC="1", G3_CKPT_EVERY="100", G3_SEED="0", **extra)
    p = subprocess.run([sys.executable, str(HERE / "train_opd.py")], env=env, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-3000:])
    assert p.returncode == 0
    return json.loads((tmp / "out" / "train_metrics.json").read_text())


def tiny_teacher(rows, extra, k=8):
    """Top-k log-probs of the deterministic tiny model itself at every target position (same code path as the trainer: inner -> lm_head -> softcap)."""
    for kk, v in dict(G3_MODEL="x", G3_DATA="x", G3_OUT=tempfile.mkdtemp(), G3_TINY="1", G3_SEED="0", **extra).items():
        os.environ[kk] = v
    sys.path.insert(0, str(HERE.parent / "g1"))
    import importlib
    import train_opd as T
    T = importlib.reload(T)
    model = T.load_model()
    model.eval()
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    with torch.no_grad():
        for r in rows:
            x = torch.tensor([r["input_ids"]], device=model.lm_head.weight.device)
            h = model.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).last_hidden_state[0]
            logits = model.lm_head(h).float()
            if softcap:
                logits = torch.tanh(logits / softcap) * softcap
            lp, ti = torch.log_softmax(logits, -1).topk(k, -1)
            pos = positions(r)
            r["t_ids"] = ti[[j - 1 for j in pos]].tolist()
            r["t_lp"] = lp[[j - 1 for j in pos]].tolist()
            r["n_loss_tokens"] = len(pos)
    for kk in ("G3_TINY_PLE",):
        os.environ.pop(kk, None)
    return rows


def test_end_to_end(extra):
    import random
    rng = random.Random(1)
    rows = tiny_teacher(make_rows(rng, 4), extra)
    m = run_trainer(rows, Path(tempfile.mkdtemp()), dict(extra, G3_LR="1e-3", G3_EPOCHS="1"))
    first = m["steps"][0]["loss"]
    assert m["stage"] == "done" and abs(first) < 2e-3, ("teacher == student must give ~0 loss at step 1", first)
    rows2 = make_rows(rng, 4)                                # a different teacher: prefers the fixed token 7 with p = 0.6 at every position (learnable by LoRA)
    for r in rows2:
        pos = positions(r)
        r["t_ids"] = [[7] + [8 + q for q in range(7)] for _ in pos]
        r["t_lp"] = [[math.log(0.6)] + [math.log(0.02)] * 7 for _ in pos]
        r["n_loss_tokens"] = len(pos)
    # forward KL (mass-covering) must clearly pull the student towards the teacher (the tiny random model caps how far: final norm + random 64-dim tied
    # embeddings bound any single logit, so the loss plateaus; full convergence of the loss itself is shown in test_loss_unit)
    m2 = run_trainer(rows2, Path(tempfile.mkdtemp()), dict(extra, G3_LR="5e-3", G3_EPOCHS="6", G3_WARMUP="1", OPD_DIV="fkl"))
    lf = [s["loss"] for s in m2["steps"]]
    assert m2["stage"] == "done" and lf[0] > 0.5 and min(lf[-4:]) < 0.9 * lf[0], lf
    # reverse KL (mode-seeking): from a near-uniform student its pull towards a teacher mode scales with the student's own mass there, so it moves slowly
    # here (the real E4B already agrees with the 31B's top-1 on 82% of its tokens); require a steady decrease and the adapter on disk
    m3 = run_trainer(rows2, Path(tempfile.mkdtemp()), dict(extra, G3_LR="5e-3", G3_EPOCHS="6", G3_WARMUP="1"))
    lr_ = [s["loss"] for s in m3["steps"]]
    assert m3["stage"] == "done" and lr_[-1] < lr_[0] - 0.01 and all(b <= a + 0.002 for a, b in zip(lr_, lr_[1:])), lr_
    return {"step1_teacher_eq_student": first, "fkl_first_last": (lf[0], lf[-1]), "rkl_first_last": (lr_[0], lr_[-1])}


if __name__ == "__main__":
    test_loss_unit(); print("ok test_loss_unit")
    for extra in ({}, {"G3_TINY_PLE": "1", "G3_HF_GC": "1"}):
        print("ok test_end_to_end", extra, "| step-1 loss with teacher==student, first/last loss with a different teacher:", test_end_to_end(extra))
