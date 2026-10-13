"""Pick the 20 spike tasks from SWE-rebench's `filtered` split (cc-by-4.0, nebius/SWE-rebench).

Filters (all from the dataset's own metadata): not one of the competition repos, python in 3.9..3.12 (the Kaggle notebook image is 3.12; older pins will not install),
llm difficulty <= 1, test_score >= 2, issue_text_score >= 1, <= 2 modified files, 1..5 FAIL_TO_PASS tests, <= 150 PASS_TO_PASS tests, no apt/conda pre-install,
one task per repo, repos that need heavy native builds excluded by name. A fixed seed picks 20; the selection is NOT looked at outcome-wise (this is a yield measurement).

Usage: python scripts/g7/select_spike_tasks.py <filtered.parquet> <out_jsonl> [n=20] [seed=3]
"""

import json
import random
import sys

import pyarrow.parquet as pq

src, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 3
COMPETITION_REPOS = {"fastapi/fastapi", "Textualize/rich", "psf/requests", "encode/httpx", "tiangolo/fastapi", "Textualize/textual"}
HEAVY = ("pennylane", "dask", "ignite", "dvc", "torch", "tensorflow", "scipy", "pandas", "numpy", "astropy", "xarray", "zarr", "scikit", "sklearn", "opencv", "pyarrow",
         "jax", "numba", "matplotlib", "qiskit", "cirq", "pymc", "geopandas", "shapely", "rasterio", "gdal", "llvm", "cython")

rows = pq.read_table(src).to_pylist()
ok = []
for r in rows:
    ic = r["install_config"] or {}
    py = str(ic.get("python") or "")
    meta = r["meta"] or {}
    sc = (meta.get("llm_score") or {})
    f2p, p2p = r["FAIL_TO_PASS"] or [], r["PASS_TO_PASS"] or []
    pre = " ".join(ic.get("pre_install") or [])
    if r["repo"] in COMPETITION_REPOS or py not in ("3.9", "3.10", "3.11", "3.12"):
        continue
    if not (sc.get("difficulty_score", 9) <= 1 and sc.get("difficulty_score", -1) >= 0 and sc.get("test_score", 0) >= 2 and sc.get("issue_text_score", 0) >= 1):
        continue
    if (meta.get("num_modified_files") or 9) > 2 or not (1 <= len(f2p) <= 5) or len(p2p) > 150 or "apt" in pre or "conda" in pre:
        continue
    if any(h in r["repo"].lower() for h in HEAVY) or any(h in (ic.get("install") or "").lower() for h in ("conda", "apt-get")):
        continue
    if "pytest" not in (ic.get("test_cmd") or "pytest") or not r["test_patch"] or not r["patch"]:
        continue
    ok.append(r)
print(f"{len(ok)} candidate tasks after filters, {len({r['repo'] for r in ok})} repos")
random.Random(seed).shuffle(ok)
seen, pick = set(), []
for r in ok:
    if r["repo"] in seen:
        continue
    seen.add(r["repo"])
    pick.append(r)
    if len(pick) == n:
        break
with open(out, "w", encoding="utf-8") as fh:
    for r in pick:
        ic = r["install_config"]
        fh.write(json.dumps({
            "instance_id": r["instance_id"], "repo": r["repo"], "base_commit": r["base_commit"], "patch": r["patch"], "test_patch": r["test_patch"],
            "problem_statement": r["problem_statement"], "hints_text": r["hints_text"] or "", "created_at": r["created_at"],
            "FAIL_TO_PASS": list(r["FAIL_TO_PASS"]), "PASS_TO_PASS": list(r["PASS_TO_PASS"]), "environment_setup_commit": r["environment_setup_commit"],
            "version": r["version"], "license_name": r["license_name"],
            "install_config": {"python": ic.get("python"), "install": ic.get("install"), "pre_install": ic.get("pre_install"), "pip_packages": ic.get("pip_packages"),
                               "reqs_path": ic.get("reqs_path"), "test_cmd": ic.get("test_cmd"), "env_vars": ic.get("env_vars")},
            "meta": {"difficulty": r["meta"]["llm_score"]["difficulty_score"], "files": r["meta"]["num_modified_files"]}}) + "\n")
print("picked:")
for r in pick:
    ic = r["install_config"]
    print(f"  {r['instance_id']:55s} py {ic.get('python')} F2P {len(r['FAIL_TO_PASS'])} P2P {len(r['PASS_TO_PASS']):3d} | {ic.get('install')} | {r['license_name']}")
