"""Local dry run of the notebook's STAGE_B + summary cells against a tiny random model (no GPU quota). Executes the exact cell source."""
import json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "g1"))
nb = json.load(open(ROOT / "kaggle_assets" / "g1" / "g1.ipynb", encoding="utf-8"))
cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}
stage_b = next(s for s in cells.values() if s.startswith("import g1_util as U"))
summary = next(s for s in cells.values() if s.startswith("B = results.get('B_train')"))

tmp = Path(tempfile.mkdtemp())
WORK = tmp / "working"; G1 = WORK / "g1"; G1.mkdir(parents=True)
CODE = ROOT / "scripts" / "g1"
RES = G1 / "results.json"; results = {}
def save_results(): RES.write_text(json.dumps(results, indent=1, default=str))
TOK = Path(r"C:/Users/jaygu/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7")
MODEL_BF16 = TOK
G1_DATA = ROOT / "kaggle_assets" / "g1_data_dataset" / "g1_train.jsonl"
class _WD:  # watchdog stub
    def cancel(self): pass
_wd = _WD()
os.environ.update(G1_TINY="1", G1_TEST_MEMFRAC="0.25")
stage_b = stage_b.replace("STAGED = Path('/tmp/ckpt_bf16')", f"STAGED = Path(r'{tmp / 'ckpt'}')")
stage_b = stage_b.replace("G1_TRAIN_LEN='8192'", "G1_TRAIN_LEN='6500'").replace("G1_LADDER='10240,12288,14336,16384,20480'", "G1_LADDER='9000,16000'")
stage_b = stage_b.replace("G1_MICRO='6'", "G1_MICRO='4'")
g = dict(globals())
exec(stage_b, g)
exec(summary, g)
print("DRY RUN DONE; phases recorded:", [k for k in results if k.startswith("B_")])
shutil.rmtree(tmp, ignore_errors=True)
