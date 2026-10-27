"""Tests of scripts/g15/train_rpo.py on a tiny random Gemma 4 (run with the Kaggle-mirroring venv: .venv_g1/Scripts/python.exe -m unittest scripts/g15/test_train_rpo.py -v).

1. the two-pass gradient equals the plain autograd gradient of the full RPO loss (with and without per-layer inputs / HF gradient checkpointing, with and without length normalisation);
2. with a fresh adapter (lora_B = 0) policy == reference: margin 0 and DPO loss = ln 2;
3. a small step along the negative gradient raises the margin (sign conventions of the two coefficients);
4. end to end through main(): reference cache, reward margin rises on a tiny overfit run, adapter saved, RPO_KINDS filter.
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import train_rpo as T  # noqa: E402


def seq(rng, n, spans):
    ids = [rng.randint(5, 990) for _ in range(n)]
    return {"input_ids": ids, "loss_spans": [list(s) for s in spans], "n_loss_tokens": sum(e - s for s, e in spans)}


def make_pair(rng, kind="s_vs_c", task="t"):
    return {"task": task, "kind": kind, "chosen": seq(rng, 300, [(40, 90), (200, 300)]), "rejected": seq(rng, 260, [(30, 70), (150, 260)])}


def randomize_lora_b(pm, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if p.requires_grad and "lora_B" in n:
                p.copy_((torch.randn(p.shape, generator=g) * 0.05).to(p.dtype))


def flat_grads(pm):
    return torch.cat([p.grad.detach().float().flatten() if p.grad is not None else torch.zeros(p.numel()) for p in pm.parameters() if p.requires_grad])


def zero(pm):
    for p in pm.parameters():
        p.grad = None


class GradientTests(unittest.TestCase):
    def build(self, ple, gc, norm):
        model = T.load_model(tiny=True, ple=ple)
        pm = T.lora_wrap(model, hf_gc=gc)
        pm.train()
        return pm, T.RPO(pm, beta=0.1, alpha=1.0, norm=norm)

    def check_equivalence(self, ple, gc, norm):
        torch.manual_seed(0)
        pm, rpo = self.build(ple, gc, norm)
        randomize_lora_b(pm)
        pair = make_pair(random.Random(1))
        ref = (rpo.ref_logp(pair["chosen"]), rpo.ref_logp(pair["rejected"]))
        zero(pm)
        info = rpo.pair_backward(pair, ref, 1)
        g_two = flat_grads(pm).clone()
        zero(pm)
        rpo.direct_loss(pair, ref).backward()
        g_direct = flat_grads(pm).clone()
        self.assertGreater(float(g_direct.norm()), 0.0)
        cos = float(torch.dot(g_two, g_direct) / (g_two.norm() * g_direct.norm()))
        rel = float((g_two - g_direct).norm() / g_direct.norm())
        self.assertGreater(cos, 0.999, (ple, gc, norm, cos, rel))
        self.assertLess(rel, 0.03, (ple, gc, norm, cos, rel))
        self.assertNotEqual(info["margin"], 0.0)   # the adapter really moved the policy away from the reference

    def test_two_pass_gradient_equals_autograd_plain(self):
        self.check_equivalence(ple=False, gc=False, norm=False)

    def test_two_pass_gradient_equals_autograd_length_normalised(self):
        self.check_equivalence(ple=False, gc=False, norm=True)

    def test_two_pass_gradient_equals_autograd_per_layer_inputs_with_hf_gc(self):
        self.check_equivalence(ple=True, gc=True, norm=False)

    def test_fresh_adapter_has_zero_margin_and_ln2_loss(self):
        pm, rpo = self.build(False, False, False)
        pair = make_pair(random.Random(2))
        ref = (rpo.ref_logp(pair["chosen"]), rpo.ref_logp(pair["rejected"]))
        zero(pm)
        info = rpo.pair_backward(pair, ref, 1)
        self.assertAlmostEqual(info["margin"], 0.0, places=3)
        self.assertAlmostEqual(info["dpo_loss"], math.log(2), places=3)
        self.assertAlmostEqual(info["d_chosen"], 0.0, places=2)

    def test_step_along_negative_gradient_raises_the_margin(self):
        torch.manual_seed(0)
        pm, rpo = self.build(False, False, False)
        randomize_lora_b(pm)
        pair = make_pair(random.Random(3))
        ref = (rpo.ref_logp(pair["chosen"]), rpo.ref_logp(pair["rejected"]))
        zero(pm)
        before = rpo.pair_backward(pair, ref, 1)
        g = flat_grads(pm).clone()
        params = [p for p in pm.parameters() if p.requires_grad]
        with torch.no_grad():
            off = 0
            for p in params:   # gradient descent on the loss, normalised step
                n = p.numel()
                p.sub_((0.05 * g[off:off + n] / g.norm()).view_as(p).to(p.dtype))
                off += n
        zero(pm)
        after = rpo.pair_backward(pair, ref, 1)
        self.assertGreater(after["margin"], before["margin"])
        self.assertLess(after["dpo_loss"], before["dpo_loss"])


class EndToEndTests(unittest.TestCase):
    def run_main(self, tmp, pairs, **env_extra):
        data = tmp / "pairs.jsonl"
        data.write_text("\n".join(json.dumps(p) for p in pairs), encoding="utf-8")
        base = dict(RPO_MODEL="x", RPO_DATA=str(data), RPO_OUT=str(tmp / "out"), RPO_TINY="1", RPO_ACC="2", RPO_LR="3e-3", RPO_EPOCHS="6", RPO_WARMUP="1", RPO_CKPT_EVERY="3")
        base.update({k: str(v) for k, v in env_extra.items()})
        env = dict(os.environ, **base)
        p = subprocess.run([sys.executable, str(HERE / "train_rpo.py")], env=env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:] + p.stdout[-1000:])
        return json.loads((tmp / "out" / "train_metrics.json").read_text())

    def test_overfit_run_raises_margin_and_saves_adapter(self):
        rng = random.Random(0)
        pairs = [make_pair(rng, task=f"t{i}") for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            m = self.run_main(tmp, pairs)
            self.assertEqual(m["stage"], "done")
            self.assertEqual(len(m["steps"]), m["optimizer_steps_planned"])
            self.assertEqual(m["optimizer_steps_planned"], 12)       # 4 pairs / accumulation 2 = 2 steps per epoch x 6 epochs
            margins = [s["margin"] for s in m["steps"]]
            self.assertAlmostEqual(margins[0], 0.0, places=2)         # first step: policy == reference
            self.assertGreater(sum(margins[-3:]) / 3, 0.5, margins)    # the preference margin grew
            self.assertTrue(all(math.isfinite(s["dpo_loss"]) and math.isfinite(s["grad_norm"]) for s in m["steps"]))
            ref = json.loads((tmp / "out" / "ref_logps.json").read_text())
            self.assertEqual(len(ref), 8)                              # one per side per pair, cached
            self.assertTrue((tmp / "out" / "adapter_full" / "adapter_model.safetensors").exists())
            self.assertTrue((tmp / "out" / "adapter_ckpt").exists())
            from safetensors.torch import load_file
            t = load_file(str(tmp / "out" / "adapter_full" / "adapter_model.safetensors"))
            self.assertGreater(max(v.float().abs().max().item() for k, v in t.items() if "lora_B" in k), 0.0)

    def test_kind_filter(self):
        rng = random.Random(1)
        pairs = [make_pair(rng, kind="s_vs_c", task="a"), make_pair(rng, kind="within_s", task="b"), make_pair(rng, kind="s_vs_f", task="c")]
        with tempfile.TemporaryDirectory() as d:
            m = self.run_main(Path(d), pairs, RPO_KINDS="s_vs_c,s_vs_f", RPO_EPOCHS=1)
            self.assertEqual(m["data"]["pairs"], 2)
            self.assertEqual(m["data"]["kinds"], {"s_vs_c": 1, "s_vs_f": 1})


if __name__ == "__main__":
    unittest.main()
