"""Local dry run of the notebook's TRAIN + PACKAGE cells (exact cell source) against a tiny random model and synthetic sequences."""
import json, os, random, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "g1")); sys.path.insert(0, str(ROOT / "scripts" / "g3"))
sys.modules["torchao"] = None
nb = json.load(open(ROOT / "kaggle_assets" / "g3" / "g3.ipynb", encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
train = next(s for s in cells if s.startswith("import g1_util as U")); package = next(s for s in cells if s.startswith("import make_adapters"))
tmp = Path(tempfile.mkdtemp()); G3 = tmp / "g3"; (G3).mkdir(); CODE = ROOT / "scripts" / "g3"
rng = random.Random(1); rows = []
for i in range(5):
    n = rng.randint(2500, 4000); ids = [rng.randint(5, 990) for _ in range(n)]; spans = [[300, 600], [n - 250, n]]
    rows.append({"task": f"t{i}", "input_ids": ids, "loss_spans": spans, "n_loss_tokens": sum(e - s for s, e in spans)})
SFT = tmp / "sft.jsonl"; SFT.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
MODEL_BF16 = tmp / "model"; MODEL_BF16.mkdir(); (MODEL_BF16 / "config.json").write_text("{}")
R = {}; save_results = lambda: None
DATA_DIR = ROOT / "data"; WORK = tmp / "work"; WORK.mkdir()
B1 = ROOT / "submission_b1"
os.environ.update(G3_TINY="1")
train = train.replace("STAGED = Path('/tmp/ckpt_bf16')", f"STAGED = Path(r'{tmp / 'staged'}')").replace("'[g3] step 1/', 1800", "'[g3] step 1/', 300").replace("max_seconds=4.0 * 3600", "max_seconds=300")
train = train.replace("G3_CKPT_EVERY='10'", "G3_CKPT_EVERY='1'")
sys.path.insert(0, str(CODE))
STABLE = Path(r"C:/Users/jaygu/AppData/Local/Temp/g3_tiny_adapter")
MODE = sys.argv[1]
if MODE == "train":          # run in the venv that has peft (the exact TRAIN cell source)
    g = dict(globals())
    exec(train, g)
    print("TRAIN CELL OK:", {k: g["R"]["train"][k] for k in ("stage", "optimizer_steps_planned", "tokens_per_second")}, "| losses", g["R"]["train"]["losses"])
    shutil.rmtree(STABLE, ignore_errors=True); shutil.copytree(G3 / "train" / "adapter_full", STABLE)
else:                        # run in the env that has google-adk (the exact PACKAGE cell source, with the current wheelhouse packages)
    import g3_wiring
    (G3 / "train").mkdir(parents=True); shutil.copytree(STABLE, G3 / "train" / "adapter_full")
    g = dict(globals()); g["W"] = g3_wiring
    exec(package.replace("DATA_DIR / 'sample_submission' / 'adapters' / 'main_lora'", f"Path(r'{ROOT}/data/sample_submission/adapters/main_lora')"), g)
    print("PACKAGE CELL OK:", g["R"]["adapter"], "|", g["R"]["wiring_real"])
