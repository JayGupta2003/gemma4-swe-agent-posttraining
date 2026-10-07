"""Analyze Experiment 1 or 2 (see docs/STRATEGY.md): per-arm metrics, paired comparisons vs control,
mechanism diagnostics from traces, and the pre-registered decision rule applied mechanically.

Usage: python scripts/analyze_experiment1.py <dir containing <exp>/<arm>/task_results.jsonl and traces/> [exp1|exp2]
Arms are auto-discovered; the control is the alphabetically first arm (A0_control / B0_control).
Exp2 additionally compares each arm with the previous rung of the ladder and reports the tool-call mix.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path



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


def call_mix(arm_dir: Path, ids: list[str]) -> dict[str, float]:
    """Share of tool calls by kind, split by whether the attempt wrote a patch (rough, command-text heuristics)."""
    import collections
    mixes = {"patched": collections.Counter(), "no_patch": collections.Counter()}
    for iid in ids:
        p = arm_dir / "traces" / f"trace_{iid}.json"
        if not p.exists():
            continue
        t = json.loads(p.read_text(encoding="utf-8"))
        kinds = collections.Counter()
        for s in t["steps"]:
            for tc in (s.get("tool_calls") or []) if s.get("source") == "agent" else []:
                fn, args = tc.get("function_name"), json.dumps(tc.get("arguments", {}))
                if fn in ("edit_file", "write_file") and "/tmp" not in args:
                    kinds["edit"] += 1
                elif fn == "run_command" and ("/tmp/" in args and "python" in args or "repro" in args):
                    kinds["repro"] += 1
                elif fn == "run_command" and "pytest" in args:
                    kinds["test"] += 1
                elif fn == "run_command" and ("grep" in args or "find " in args):
                    kinds["grep/find"] += 1
                elif fn in ("read_file", "run_command"):
                    kinds["read/other cmd"] += 1
                elif fn not in ("get_status", "submit_patch"):
                    kinds["graph/other"] += 1
        rows_patch = (arm_dir / "task_results.jsonl").read_text(encoding="utf-8")
        got = any(json.loads(l)["instance_id"] == iid and json.loads(l)["agent_patch_size"] > 0 for l in rows_patch.splitlines())
        mixes["patched" if got else "no_patch"].update(kinds)
    out = {}
    for grp, c in mixes.items():
        tot = sum(c.values()) or 1
        out[grp] = {k: round(100 * v / tot) for k, v in c.most_common()}
    return out


def token_stats(arm_dir: Path, ids: list[str]) -> dict:
    """Context length per attempt = prompt + completion tokens of the last step (what the model actually saw), plus decoded tokens."""
    import statistics as st
    ctx, comp, steps = [], [], []
    for iid in ids:
        p = arm_dir / "traces" / f"trace_{iid}.json"
        if not p.exists():
            continue
        t = json.loads(p.read_text(encoding="utf-8"))
        ag = [x for x in t["steps"] if x.get("source") == "agent" and x.get("metrics")]
        if not ag:
            continue
        ctx.append(ag[-1]["metrics"]["prompt_tokens"] + ag[-1]["metrics"]["completion_tokens"])
        comp.append(sum(x["metrics"]["completion_tokens"] for x in ag))
        steps.append(len(ag))
    q = lambda a, f: round(f(a)) if a else float("nan")
    return {"ctx_median": q(ctx, st.median), "ctx_p75": q(sorted(ctx)[3 * len(ctx) // 4:][:1], st.median) if ctx else float("nan"),
            "decoded_median": q(comp, st.median), "steps_median": q(steps, st.median), "n": len(ctx)}


def main(root: Path, exp_name: str = "exp1") -> None:
    exp = root / exp_name
    arms = sorted(d.name for d in exp.iterdir() if (d / "task_results.jsonl").exists())
    ctrl = arms[0]
    data = {a: load_rows(exp / a) for a in arms}
    ids = sorted(data[ctrl])
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
    base = data[ctrl]
    for a in data:
        if a == ctrl:
            continue
        for k, better_when_true in (("resolved", True), ("patch", True), ("cap", False)):
            g = sum(1 for i in ids if (flag(data[a][i], k) != flag(base[i], k)) and (flag(data[a][i], k) == better_when_true))
            l = sum(1 for i in ids if (flag(data[a][i], k) != flag(base[i], k)) and (flag(data[a][i], k) != better_when_true))
            print(f"  {a:14s} {k:8s} better on {g:2d}, worse on {l:2d}  (sign-test p={sign_p(g, l):.2f})")

    repos = sorted({i.split("_")[0] for i in ids})
    print("\nBy repo, resolved (" + ", ".join(f"{rp}/{sum(i.startswith(rp + '_') for i in ids)}" for rp in repos) + "):")
    for a, rows in data.items():
        print(f"  {a:14s} " + " ".join(f"{sum(rows[i]['resolved'] for i in ids if i.startswith(rp + '_'))}" for rp in repos))

    c_res = sum(flag(r, "resolved") for r in base.values())
    c_patch = sum(flag(r, "patch") for r in base.values())
    if exp_name == "exp3":
        print("\nTOKENS per attempt (context = last step prompt+completion; decoded = sum of completion tokens):")
        ts = {a: token_stats(exp / a, ids) for a in arms}
        for a in arms:
            print(f"  {a:16s} {ts[a]}")
        print("\nPRE-REGISTERED RULE (STRATEGY.md Experiment 3): an arm is ACCEPTABLE iff resolved >= control-2 AND patch-written >= control-2 AND "
              "median context <= 0.80 x control; it is a WIN if additionally resolved >= control+3. Anything else: keep the control prompt.")
        for a in arms[1:]:
            res = sum(flag(r, "resolved") for r in data[a].values()); pat = sum(flag(r, "patch") for r in data[a].values())
            ratio = ts[a]["ctx_median"] / ts[ctx_arm := arms[0]]["ctx_median"] if ts[arms[0]]["ctx_median"] else float("nan")
            ok = res >= c_res - 2 and pat >= c_patch - 2 and ratio <= 0.80
            win = ok and res >= c_res + 3
            print(f"  {a:16s} resolved {res-c_res:+d} patch {pat-c_patch:+d} median-context ratio {ratio:.2f} -> {'WIN' if win else 'ACCEPTABLE (lean without loss)' if ok else 'keep control'}")
        return
    if exp_name == "exp2":
        print("\nLADDER (each arm vs the previous rung; two-sided exact sign test):")
        for prev, a in zip(arms, arms[1:]):
            for k in ("resolved", "patch"):
                g = sum(1 for i in ids if flag(data[a][i], k) and not flag(data[prev][i], k))
                l = sum(1 for i in ids if flag(data[prev][i], k) and not flag(data[a][i], k))
                print(f"  {prev} -> {a:14s} {k:8s} better on {g:2d}, worse on {l:2d} (p={sign_p(g, l):.2f})")
        print("\nTool-call mix, % of calls (patched attempts vs no-patch attempts):")
        for a in arms:
            print(f"  {a:14s} {call_mix(exp / a, ids)}")
        print("\nPRE-REGISTERED RULE (STRATEGY.md Experiment 2): candidate = resolved >= control+5 AND patch-written >= control "
              "AND no repo worse than control by more than 2; a candidate must replicate on the confirmation set before any submission change.")
        for a in arms[1:]:
            res = sum(flag(r, "resolved") for r in data[a].values())
            pat = sum(flag(r, "patch") for r in data[a].values())
            worst = min(sum(data[a][i]["resolved"] for i in ids if i.startswith(rp + "_")) - sum(base[i]["resolved"] for i in ids if i.startswith(rp + "_"))
                        for rp in repos)
            ok = res - c_res >= 5 and pat >= c_patch and worst >= -2
            print(f"  {a:14s} resolved {res-c_res:+d} patch {pat-c_patch:+d} worst-repo {worst:+d} -> {'CANDIDATE (confirm first)' if ok else 'no evidence'}")
        return

    print("\nPRE-REGISTERED RULE (STRATEGY.md): mechanism-confirmed = patch-written >= control+6; candidate = resolved >= control+3 AND patch not lower (and a confirming 2nd round).")
    for a, rows in data.items():
        if a == ctrl:
            continue
        res = sum(flag(r, "resolved") for r in rows.values())
        pat = sum(flag(r, "patch") for r in rows.values())
        print(f"  {a:14s} patch {pat-c_patch:+d} (need +6: {'YES' if pat-c_patch>=6 else 'no'}) | resolved {res-c_res:+d} (need +3: {'YES' if res-c_res>=3 else 'no'}) "
              f"-> {'CANDIDATE (needs round 2)' if (res-c_res>=3 and pat>=c_patch) else 'no evidence'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "exp1")
