"""Chunked-layer-only local profile at long lengths with real 31B layer geometry (2 layers: sliding + global). Env CG / CS set chunk sizes."""
import os, sys
sys.path.insert(0, "scripts/g1")
from profile_real_dims import run
lens = [int(x) for x in os.environ.get("LENS", "12288,16384,20480,24576").split(",")]
for T in lens:
    print(f"CG={os.environ.get('CG','256')} CS={os.environ.get('CS','1024')} T={T:6d}  chunked layer: {run('chunk', T, ['sliding_attention', 'full_attention'])}", flush=True)
