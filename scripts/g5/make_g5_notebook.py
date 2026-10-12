"""Generate kaggle_assets/g5/ : first GPU test of the smaller Gemma 4 variant `gemma-4-12b-it-qat-w4a16-ct` (variants allowed since 2026-10-12, forum #748287).

Derived from the G2 notebook (kaggle_assets/g2/g2.ipynb: capture proxy, smoke gate, full round) by exact-string edits, each asserted, so the harness path is
identical to the 31B rounds that we compare against:
  - arm: the B1 arm with every `gemma-4-31b-it-qat-w4a16-ct` replaced by the 12B name (swegemma requires all agents to use the same variant)
  - vLLM: 12B path, staff-documented `--hf-overrides` (Gemma4ForCausalLM / gemma4), LoRA off (this is a base-model test), and a guarded source patch that makes
    Gemma4ForCausalLM.load_weights skip the vision tensors the 12B checkpoint carries (users report a KeyError on `vision_embedder.patch_dense.weight` on wheelhouse v30)
  - a single-stream decode-speed probe (31B reference: 37.5 tok/s at concurrency 1) before the smoke round
  - one evaluation round `b12` over the same 75 tasks, same cap and concurrency as the 31B rounds r1/r2 (25/75 and 28/75 resolved, 44/39 cap hits)
Budget: 160-minute watchdog; expected ~1.5-1.8 h of L4x4 (~3-3.6 quota h).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "kaggle_assets" / "g2" / "g2.ipynb").read_text(encoding="utf-8"))
OUT = ROOT / "kaggle_assets" / "g5"
OUT.mkdir(parents=True, exist_ok=True)
M31, M12 = "gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct"


def sub(s, old, new, count=1):
    assert s.count(old) == count, f"expected {count} x {old[:70]!r}, found {s.count(old)}"
    return s.replace(old, new)


cells = SRC["cells"]
txt = {i: "".join(c["source"]) for i, c in enumerate(cells)}
assert [c["cell_type"] for c in cells] == ["markdown", "code", "markdown", "code", "markdown", "code", "markdown", "code"]

# ---------------------------------------------------------------- SETUP (cell 3)
s = txt[3]
s = sub(s, "G2 = WORKING_DIR / 'g2'", "G2 = WORKING_DIR / 'g5'")
s = sub(s, "'g2_summary.json'", "'g5_summary.json'")
s = sub(s, "BUDGET_MIN = 235", "BUDGET_MIN = 160")
s = sub(s, "shutil.copytree(arm_dirs[0], ARM)\n",
        "shutil.copytree(arm_dirs[0], ARM)\n"
        f"for _rel in ('agent.yaml', 'sub_agents/code_analyzer.yaml'):   # one variant for every agent and sub-agent\n"
        f"    _p = ARM / _rel; _t = _p.read_text(encoding='utf-8'); assert _t.count('{M31}') == 1, _rel; _p.write_text(_t.replace('{M31}', '{M12}'), encoding='utf-8', newline='\\n')\n"
        f"assert '{M31}' not in (ARM / 'agent.yaml').read_text()\n")
txt[3] = s

# ---------------------------------------------------------------- VLLM (cell 5)
s = txt[5]
s = sub(s, f"TARGET_MODEL_NAME = '{M31}'", f"TARGET_MODEL_NAME = '{M12}'")
s = sub(s, f"MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/{M31}/2')", f"MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/{M12}/2')\nassert MODEL_PATH.is_dir(), MODEL_PATH")
s = sub(s, "enable_lora=True, max_loras=8, max_lora_rank=128,",
        "enable_lora=False,\n    extra_args=('--hf-overrides', json.dumps({'architectures': ['Gemma4ForCausalLM'], 'model_type': 'gemma4'})),")
s = sub(s, "startup_timeout=60 * 20)", "startup_timeout=60 * 12)")
PATCH = r'''# --- guarded patch: Gemma4ForCausalLM.load_weights must skip vision tensors (12B checkpoint carries `vision_embedder.*`; see forum #748287)
import re as _re, vllm as _vllm
_g4 = Path(_vllm.__file__).parent / 'model_executor' / 'models' / 'gemma4.py'
_src = _g4.read_text(encoding='utf-8')
print('gemma4.py: mentions vision_embedder:', 'vision_embedder' in _src, '| lines', len(_src.splitlines()))
_marker = '                name = name.replace("language_model.", "")\n'
if 'vision_embedder' not in _src and _src.count(_marker) == 1:
    _src = _src.replace(_marker, _marker + '                if name.startswith(("model.vision_embedder.", "vision_embedder.", "embed_vision.", "model.embed_vision.")):\n                    continue   # patched by G5: text-only model\n')
    _g4.write_text(_src, encoding='utf-8'); print('patched', _g4)
else:
    print('NOT patching (already handles vision_embedder, or the insertion point is not unique):', _src.count(_marker))
'''
s = sub(s, "server_instance = VllmServer(vllm_cfg, adapter_manifest=adapters)", PATCH + "server_instance = VllmServer(vllm_cfg, adapter_manifest=adapters)")
PROBE = r'''
# --- single-stream decode speed (31B reference with no adapter: 37.5 tok/s at concurrency 1)
import requests as _rq
def _probe(n_tok=400):
    best = 0
    for _ in range(3):
        t0 = time.perf_counter()
        # vLLM serves the model under its PATH (captured 31B requests carry the path as `model`), not under the alias
        r = _rq.post(VLLM_URL + '/v1/chat/completions', json={'model': str(MODEL_PATH), 'max_tokens': n_tok, 'temperature': 0.7,
              'messages': [{'role': 'user', 'content': 'Write a long, detailed explanation of how Python decorators work, with several examples.'}],
              'chat_template_kwargs': {'enable_thinking': False}}, timeout=600).json()
        dt = time.perf_counter() - t0
        if 'usage' not in r: raise RuntimeError('probe response without usage: ' + json.dumps(r)[:400])
        best = max(best, r['usage']['completion_tokens'] / dt)
    return round(best, 1)
try:   # informational only: must never stop the run
    RESULTS['decode_tok_s_single_stream'] = _probe(); print('decode tok/s (single stream):', RESULTS['decode_tok_s_single_stream'], '| 31B reference 37.5')
except Exception as _e:
    RESULTS['decode_probe_error'] = repr(_e)[:600]; print('decode probe FAILED (non-fatal):', repr(_e)[:600])
save_results()
'''
s = s + "\n" + PROBE
txt[5] = s

# ---------------------------------------------------------------- RUN (cell 7): one round b12 instead of r1, r2
s = txt[7]
old_loop_start = s.index("# ------------------------------------------------------------------ full rounds")
old_loop_end = s.index("print('\\n===== G2 SUMMARY =====')")
s = s[:old_loop_start] + ("# ------------------------------------------------------------------ one full round over all 75 tasks with the 12B (31B reference r1/r2: 25/75 and 28/75)\n"
                          "ids = list(ALL_IDS); random.Random(101).shuffle(ids)\nrun_round('b12', ids)\n\n") + s[old_loop_end:]
s = sub(s, "print('\\n===== G2 SUMMARY =====')", "print('\\n===== G5 (12B) SUMMARY =====')")
s = sub(s, "for tag in ('smoke', 'r1', 'r2'):", "for tag in ('smoke', 'b12'):")
txt[7] = s

txt[0] = ("# G5: first GPU test of gemma-4-12b-it-qat-w4a16-ct (base model, no adapter): serving, decode speed, 4-task smoke, one 75-task round\n\n"
          "Compare with the 31B rounds r1/r2 (25/75, 28/75 resolved; 44/39 capped). Output: `g5/g5_summary.json`, `g5/capture.jsonl.gz`, `g5/b12/`.")
for i, c in enumerate(cells):
    c["source"] = txt[i].splitlines(True)
    if c["cell_type"] == "code":
        compile(txt[i], f"cell{i}", "exec")

(OUT / "g5.ipynb").write_text(json.dumps({"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
meta = json.loads((ROOT / "kaggle_assets" / "g2" / "kernel-metadata.json").read_text(encoding="utf-8"))
meta.update({"id": "noobhunterjay/gemma-4-g5-12b", "title": "Gemma 4 G5 12B", "code_file": "g5.ipynb",
             "model_sources": [f"google/gemma-4/Other/{M12}/2"]})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", OUT)
