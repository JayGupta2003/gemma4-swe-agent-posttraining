"""Turn the raw PEFT output of train_g1.py into scorer-format adapter directories and verify them against the sample adapter format.

Variants written under <out>/:
  g1_full/    all trained LoRA matrices (attention + MLP)
  g1_attn/    attention-only subset (q,k,v,o): the safest set for vLLM LoRA support
  g1_canary/  attention-only with RANDOM lora_B (so the adapter visibly changes outputs even after only a few training steps):
              used only to prove the serving path really applies adapters ("silently zero" LoRA was reported by other teams)

Every variant: bf16 tensors, keys `base_model.model.model.language_model.layers.N.<module>.lora_{A,B}.weight`, adapter_config.json with
list-valued target_modules and base_model_name_or_path = the scorer's model id.

Usage: python make_adapters.py <adapter_full_dir> <out_dir> [sample_adapter_dir]
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SCORER_MODEL = "google/gemma-4-31b-it-qat-w4a16-ct"
KEY_RE = re.compile(r"^base_model\.model\.model\.language_model\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.lora_(A|B)\.weight$")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def write_variant(out: Path, tensors: dict[str, torch.Tensor], cfg: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mods = sorted({k.split(".")[-3] for k in tensors})
    cfg = dict(cfg, base_model_name_or_path=SCORER_MODEL, inference_mode=True, target_modules=mods)
    cfg.pop("layers_to_transform", None) if cfg.get("layers_to_transform") is None else None
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in tensors.items()}, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    size = (out / "adapter_model.safetensors").stat().st_size
    return {"dir": str(out), "tensors": len(tensors), "target_modules": mods, "bytes": size}


def main(src: Path, out: Path, sample: Path | None) -> dict:
    t = load_file(str(src / "adapter_model.safetensors"))
    cfg = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    bad = [k for k in t if not KEY_RE.match(k)]
    assert not bad, f"unexpected tensor names, e.g. {bad[:3]}"
    report: dict = {"n_tensors": len(t), "rank": cfg.get("r"), "alpha": cfg.get("lora_alpha"), "dtype_in": str(next(iter(t.values())).dtype)}
    if sample is not None and (sample / "adapter_model.safetensors").exists():
        sk = list(load_file(str(sample / "adapter_model.safetensors")))
        report["sample_key_example"] = sk[0]
        report["our_key_example"] = sorted(t)[0]
        # same naming scheme (prefix up to the layer index) as the scorer's sample adapter
        pre = lambda k: k.split(".layers.")[0]  # noqa: E731
        assert pre(sk[0]) == pre(sorted(t)[0]), (pre(sk[0]), pre(sorted(t)[0]))
        sample_cfg = json.loads((sample / "adapter_config.json").read_text(encoding="utf-8"))
        report["sample_cfg_keys_missing_in_ours"] = sorted(set(sample_cfg) - set(cfg))
    report["g1_full"] = write_variant(out / "g1_full", t, cfg)
    attn = {k: v for k, v in t.items() if k.split(".")[-3] in ATTN}
    report["g1_attn"] = write_variant(out / "g1_attn", attn, cfg)
    g = torch.Generator().manual_seed(0)
    canary = {k: (torch.randn(v.shape, generator=g) * 0.02 if k.endswith("lora_B.weight") else v.clone()) for k, v in attn.items()}
    report["g1_canary"] = write_variant(out / "g1_canary", canary, cfg)
    (out / "adapters_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
    print(json.dumps(r, indent=1))
