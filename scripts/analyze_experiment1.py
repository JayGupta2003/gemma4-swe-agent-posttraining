"""Analyze Experiment 1 (see docs/STRATEGY.md): per-arm metrics, paired comparisons vs control,
mechanism diagnostics from traces, and the pre-registered decision rule applied mechanically.

Usage: python scripts/analyze_experiment1.py <dir containing exp1/<arm>/task_results.jsonl and traces/>
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ARMS = ["A0_control", "A1_edit_early", "A2_think_off", "A3_edit_think"]


def load_rows(arm_dir: Path) -> dict[str, dict]:
    return {r["instance_id"]: r for r in map(json.loads, (arm_dir / "task_results.jsonl").read_text(encoding="utf-8").splitlines())}


def trace_metrics(arm_dir: Path, ids: list[str]) -> dict:
    """First real edit (turn, elapsed seconds) and LLM-turn latency from ATIF traces."""
    first_turn, first_sec, never = [], [], 0
    lat = []
    for iid in ids:
        p = arm_dir / "traces" / f"trace_{iid}.json"
        if not p.exists():
            continue
        t = json.loads(p.read_text(encoding="utf-8"))
        turn, got, prev_obs = 0, False, None
        for s in t["steps"]:
            if s.get("source") != "agent":
                continue
            tcs = s.get("tool_calls") or []
            call_t = (tcs[0].get("extra") or {}).get("elapsed_s", (s.get("extra") or {}).get("elapsed_s")) if tcs else None
            if tcs:
                turn += 1
                if prev_obs is not None and call_t is not None and call_t >= prev_obs:
                    lat.append(call_t - prev_obs)
                for tc in tcs:
                    if not got and tc.get("function_name") in ("edit_file", "write_file") and "/tmp" not in json.dumps(tc.get("arguments", {})):
                        got = True
                        first_turn.append(turn)
                        first_sec.append(call_t if call_t is not None else float("nan"))
            obs_t = ((s.get("observation") or {}).get("extra") or {}).get("elapsed_s")
            prev_obs = obs_t if obs_t is not None else (call_t if call_t is not None else prev_obs)
        never += (not got)
    q = lambda a, f: (statistics.median(a) if a else float("nan")) if f == "med" else (statistics.mean(a) if a else float("nan"))
    p90 = lambda a: sorted(a)[int(0.9 * len(a))] if a else float("nan")
    return {"never_edit": never, "first_edit_turn_med": q(first_turn, "med"), "first_edit_s_med": q([x for x in first_sec if x == x], "med"),
            "turn_lat_med": q(lat, "med"), "turn_lat_mean": q(lat, "mean"), "turn_lat_p90": p90(lat), "n_turns": len(lat)}


def sign_p(b: int, c: int) -> float:
    """Two-sided exact sign test on discordant pairs (b gained, c lost)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main(root: Path) -> None:
    exp = root / "exp1"
    data = {a: load_rows(exp / a) for a in ARMS if (exp / a / "task_results.jsonl").exists()}
    ids = sorted(data["A0_control"])
    flag = lambda r, k: {"resolved": r["resolved"], "patch": r["agent_patch_size"] > 0, "cap": "session timeout" in (r["error"] or "")}[k]

    print(f"{'arm':14s} {'resolved':>8s} {'patch':>6s} {'capHit':>7s} {'mean_s':>7s} | {'neverEdit':>9s} {'1stEditTurn':>11s} {'1stEdit_s':>9s} | {'turnLat med/mean/p90':>22s}")
    tm = {}
    for a, rows in data.items():
        tm[a] = trace_metrics(exp / a, ids)
        m = tm[a]
        print(f"{a:14s} {sum(flag(r,'resolved') for r in rows.values()):>8d} {sum(flag(r,'patch') for r in rows.values()):>6d} "
              f"{sum(flag(r,'cap') for r in rows.values()):>7d} {statistics.mean(r['duration_seconds'] for r in rows.values()):>7.1f} | "
              f"{m['never_edit']:>9d} {m['first_edit_turn_med']:>11.1f} {m['first_edit_s_med']:>9.1f} | "
              f"{m['turn_lat_med']:>6.1f}/{m['turn_lat_mean']:>5.1f}/{m['turn_lat_p90']:>5.1f}")

    print("\nPaired vs control (per task; gained/lost = arm better/worse than control; two-sided exact sign test):")
    base = data["A0_control"]
    for a in data:
        if a == "A0_control":
            continue
        for k, better_when_true in (("resolved", True), ("patch", True), ("cap", False)):
            g = sum(1 for i in ids if (flag(data[a][i], k) != flag(base[i], k)) and (flag(data[a][i], k) == better_when_true))
            l = sum(1 for i in ids if (flag(data[a][i], k) != flag(base[i], k)) and (flag(data[a][i], k) != better_when_true))
            print(f"  {a:14s} {k:8s} better on {g:2d}, worse on {l:2d}  (sign-test p={sign_p(g, l):.2f})")

    print("\nBy repo, resolved (arm: fastapi/10 requests/3 rich/17):")
    for a, rows in data.items():
        by = {p: sum(rows[i]["resolved"] for i in ids if i.startswith(p)) for p in ("fastapi", "requests", "rich")}
        print(f"  {a:14s} {by['fastapi']}/10 {by['requests']}/3 {by['rich']}/17")

    print("\nPRE-REGISTERED RULE (STRATEGY.md): mechanism-confirmed = patch-written >= control+6; candidate = resolved >= control+3 AND patch not lower (and a confirming 2nd round).")
    c_res = sum(flag(r, 'resolved') for r in base.values()); c_patch = sum(flag(r, 'patch') for r in base.values())
    for a, rows in data.items():
        if a == "A0_control":
            continue
        res = sum(flag(r, 'resolved') for r in rows.values()); pat = sum(flag(r, 'patch') for r in rows.values())
        print(f"  {a:14s} patch {pat-c_patch:+d} (need +6: {'YES' if pat-c_patch>=6 else 'no'}) | resolved {res-c_res:+d} (need +3: {'YES' if res-c_res>=3 else 'no'}) "
              f"-> {'CANDIDATE (needs round 2)' if (res-c_res>=3 and pat>=c_patch) else 'no evidence'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
