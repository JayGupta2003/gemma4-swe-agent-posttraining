import sys, tempfile
from pathlib import Path
W = r"C:/Users/jaygu/AppData/Local/Temp/wh_new"
sys.path.insert(0, W + "/x_adk_submission-0.2.13"); sys.path.insert(0, W + "/x_swegemma-0.2.11")   # the current wheelhouse packages
sys.path.insert(0, str(Path(__file__).parent))
import g3_wiring as G
root = Path(__file__).resolve().parents[2]
arm = G.make_eval_arm(root / "submission_b1", root / "data" / "sample_submission" / "adapters" / "main_lora", Path(tempfile.mkdtemp()) / "arm")
print((arm / "agent.yaml").read_text().splitlines()[:5])
print((arm / "sub_agents" / "code_analyzer.yaml").read_text().splitlines()[:5])
print("WIRING OK:", G.check_wiring(arm))
