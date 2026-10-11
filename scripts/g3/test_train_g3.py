"""Local test of train_g3.py on a tiny random Gemma 4 with synthetic token sequences (run with the Kaggle-mirroring venv)."""
import json, os, random, subprocess, sys, tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
rng = random.Random(0)
rows = []
for i in range(7):
    n = rng.randint(2500, 5000)
    ids = [rng.randint(5, 990) for _ in range(n)]
    spans = [[400, 700], [1200, 1500], [n - 300, n]]
    rows.append({"task": f"t{i % 3}", "input_ids": ids, "loss_spans": spans, "n_loss_tokens": sum(e - s for s, e in spans)})
(tmp / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
env = dict(os.environ, G3_MODEL="x", G3_DATA=str(tmp / "d.jsonl"), G3_OUT=str(tmp / "out"), G3_TINY="1", G3_ACC="2", G3_CKPT_EVERY="2", **{k: v for k, v in [kv.split("=") for kv in sys.argv[1:]]})
p = subprocess.run([sys.executable, str(Path(__file__).parent / "train_g3.py")], env=env, capture_output=True, text=True)
print("\n".join(l for l in p.stdout.splitlines() if l.startswith("[g3]"))[:2500])
print("returncode", p.returncode)
if p.returncode:
    print(p.stderr[-1500:])
m = json.loads((tmp / "out" / "train_metrics.json").read_text())
print("stage:", m["stage"], "| steps ok:", len(m.get("steps", [])), "of", m.get("optimizer_steps_planned"), "| oom events:", len(m.get("oom_events", [])), "| leak GiB:", m.get("leak_gib_at_end"))
print("lr schedule:", [s["lr"] for s in m.get("steps", [])])
print("adapter files:", sorted(p.name for p in (tmp / "out" / "adapter_full").iterdir()) if (tmp / "out" / "adapter_full").exists() else None,
      "| ckpt:", (tmp / "out" / "adapter_ckpt").exists())
if m["stage"] == "done" and m.get("steps"):
    from safetensors.torch import load_file
    t = load_file(str(tmp / "out" / "adapter_full" / "adapter_model.safetensors"))
    b = [v.float().abs().mean().item() for k, v in t.items() if "lora_B" in k]
    print("adapter tensors:", len(t), "| mean |lora_B| =", f"{sum(b) / len(b):.2e}", "| example key:", sorted(t)[0])
