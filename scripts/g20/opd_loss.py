"""Per-token distillation loss for on-policy distillation (OPD) from teacher top-k log-probabilities.

The teacher's top-k log-probs (from vLLM prompt_logprobs) are TRUE log-probs over the full vocabulary, so the probability of "any other token" is known too:
p_rest = 1 - sum_k p(x_k). The loss is the exact KL divergence between the student and teacher distributions coarse-grained to k+1 categories (the teacher's
top-k tokens, plus one bucket for the rest of the vocabulary). By the data-processing inequality it is a lower bound on the full-vocabulary KL, and it is 0 iff
the student puts the same mass as the teacher on each top-k token and on the remainder. No hyperparameter, no renormalisation that would hide student mass
outside the teacher's support.

  reverse KL (default, mode-seeking, MiniLLM / on-policy distillation):   KL(q || p) = sum_c q_c (log q_c - log p_c)
  forward KL:                                                            KL(p || q) = sum_c p_c (log p_c - log q_c)

Inputs per position: student logits [n, V] (after softcapping), teacher top-k ids [n, k] (int64; -1 = padding) and log-probs [n, k] (float).
"""

import torch

MIN_P = 1e-6


def _teacher(t_ids, t_lp):
    valid = t_ids >= 0
    lp = torch.where(valid, t_lp.float(), torch.full_like(t_lp.float(), -float("inf")))
    p_top = lp.exp()                                                     # [n, k], 0 at padding
    p_rest = (1.0 - p_top.sum(-1)).clamp_min(MIN_P)                      # [n]
    return valid, p_top, p_rest


def coarse_kl(logits, t_ids, t_lp, mode="rkl"):
    """Returns the per-position loss [n] (float32)."""
    logq = torch.log_softmax(logits.float(), dim=-1)                     # [n, V]
    valid, p_top, p_rest = _teacher(t_ids, t_lp)
    lq_top = torch.gather(logq, 1, t_ids.clamp_min(0))                   # [n, k]
    lq_top = torch.where(valid, lq_top, torch.full_like(lq_top, -float("inf")))
    q_top = lq_top.exp()
    lq_rest = torch.log((1.0 - q_top.sum(-1)).clamp_min(MIN_P))          # [n]
    lp_top = torch.where(valid, t_lp.float(), torch.zeros_like(lq_top))  # finite at padding (masked below)
    lp_rest = p_rest.log()
    if mode == "rkl":
        top = torch.where(valid, q_top * (lq_top.clamp_min(-1e4) - lp_top), torch.zeros_like(q_top)).sum(-1)
        rest = lq_rest.exp() * (lq_rest - lp_rest)
    elif mode == "fkl":
        top = torch.where(valid, p_top * (lp_top - lq_top.clamp_min(-1e4)), torch.zeros_like(p_top)).sum(-1)
        rest = p_rest * (lp_rest - lq_rest)
    else:
        raise ValueError(mode)
    return top + rest
