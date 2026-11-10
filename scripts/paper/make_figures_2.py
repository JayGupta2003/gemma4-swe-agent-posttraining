"""Second batch of paper figures and tables (pool rounds, small model, temperature and pair tables), computed from the raw result files in data_extra/ (gitignored).
Run with the `mlai` conda python (matplotlib):  python scripts/paper/make_figures_2.py
Outputs: docs/paper/figures/fig5_pool_rounds.png, fig6_small_model.png, docs/paper/tables_2.md, docs/paper/figure_data_2.json (every plotted number).
Same design rules as make_figures.py: one measure per axis, direct labels, categorical slots 1-3 (blue, orange, aqua; aqua labelled directly), text in ink colours.
"""

import collections
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data_extra"
OUT = ROOT / "docs" / "paper" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b0"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NL = chr(10)


def R(p):
    return {json.loads(line)["instance_id"]: json.loads(line) for line in open(p, encoding="utf-8")}


def kind(r):
    return "S" if r["resolved"] else ("C" if "timeout" in (r.get("error") or "") else "F")


data = {}
runs = {"base r1": R(D / "g2_run1_2026-10-11/g2/r1/task_results.jsonl"), "base r2": R(D / "g2_run1_2026-10-11/g2/r2/task_results.jsonl"),
        "G3 adapter": R(D / "g3_run1_2026-10-12/g3/e1/task_results.jsonl"), "G11 adapter": R(D / "g11_run1_2026-10-20/g3/e1/task_results.jsonl")}

# ------------------------------------------------------------ Fig 5: three pool rounds (31B, 171 tasks)
pool = {"p1": R(D / "g10_run1_2026-10-19/g10/p1/task_results.jsonl"), "p2": R(D / "g15_run1_2026-10-26/g15/p2/task_results.jsonl"), "p3": R(D / "g16_run1_2026-10-27/g16/p3/task_results.jsonl")}
ptasks = sorted(pool["p1"])
assert all(set(r) == set(ptasks) for r in pool.values()) and len(ptasks) == 171
cls_counts = {n: collections.Counter(kind(pool[n][t]) for t in ptasks) for n in pool}
solved_rounds = collections.Counter(sum(pool[n][t]["resolved"] for n in pool) for t in ptasks)
data["fig5_pool_round_outcomes"] = {n: dict(c) for n, c in cls_counts.items()}
data["fig5_tasks_by_rounds_solved_of_3"] = {str(k): solved_rounds[k] for k in range(4)}
flips12 = [t for t in ptasks if pool["p1"][t]["resolved"] != pool["p2"][t]["resolved"]]
data["fig5_p1_vs_p2_flips"] = {"tasks": len(flips12), "solved_vs_capped": sum("C" in (kind(pool["p1"][t]), kind(pool["p2"][t])) for t in flips12)}
fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=200, gridspec_kw={"width_ratios": [1.25, 1]})
labels = ["p1" + NL + "T = 0.2", "p2" + NL + "T = 1.0", "p3" + NL + "T = 1.0"]
segs = [("S", "solved", BLUE, "white"), ("F", "finished," + NL + "wrong patch", ORANGE, INK), ("C", "hit the" + NL + "5-minute cap", AQUA, INK)]
bottom = [0, 0, 0]
for key, lab, col, txtcol in segs:
    vals = [cls_counts[n][key] for n in ("p1", "p2", "p3")]
    axa.bar(range(3), vals, bottom=bottom, color=col, width=0.62, edgecolor=SURFACE, linewidth=2)
    for i, (v, b0) in enumerate(zip(vals, bottom)):
        axa.text(i, b0 + v / 2, str(v), ha="center", va="center", color=txtcol, fontsize=9)
    axa.text(2.42, bottom[2] + vals[2] / 2, lab, ha="left", va="center", color=INK2, fontsize=8)
    bottom = [b0 + v for b0, v in zip(bottom, vals)]
axa.set_xticks(range(3), labels, fontsize=8)
axa.set_xlim(-0.5, 3.6)
axa.set_ylim(0, 178)
axa.set_ylabel("pool tasks (of 171)")
axa.set_title("Outcomes per round", loc="left", fontsize=10, color=INK)
xs5 = [0, 1, 2, 3]
ys5 = [solved_rounds[k] for k in xs5]
axb.bar(xs5, ys5, color=[GRAY, BLUE, BLUE, GRAY], width=0.62)
for x, y in zip(xs5, ys5):
    axb.text(x, y + 2, str(y), ha="center", va="bottom", color=INK, fontsize=10)
axb.set_xticks(xs5, ["never", "1 of 3", "2 of 3", "always"], fontsize=8)
axb.set_ylim(0, 118)
axb.set_title(f"Rounds in which a task is solved ({solved_rounds[1] + solved_rounds[2]} mixed)", loc="left", fontsize=10, color=INK)
axb.yaxis.grid(True, color=GRID, linewidth=0.6)
axb.set_axisbelow(True)
fig.suptitle("Temperature 1.0 changes who is solved, not how many; the contrast is mostly the clock", x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "fig5_pool_rounds.png")
plt.close(fig)

# ------------------------------------------------------------ Fig 6: small model
e4 = {"E4B base" + NL + "5-min prompt": R(D / "g14a_run1_2026-10-24/g14a/e4b/task_results.jsonl"),
      "E4B base" + NL + "B2 prompt": R(D / "g14c_run1_2026-10-25/g14c/e4b_b2/task_results.jsonl"),
      "E4B base" + NL + "verify prompt": R(D / "g14c_run1_2026-10-25/g14c/e4b_v/task_results.jsonl")}
e4_lora = R(D / "g14b_run1_2026-10-25/g14b/e1/task_results.jsonl")
bars = [("31B base" + NL + "round 1", runs["base r1"], GRAY), ("31B base" + NL + "round 2", runs["base r2"], GRAY)] + [(k, v, BLUE) for k, v in e4.items()] + [("E4B + LoRA" + NL + "(31B traces)", e4_lora, ORANGE)]
bars.append(("E4B + LoRA" + NL + "(Qwen3-Coder" + NL + "episodes)", R(D / "g18_run1/g18/e1/task_results.jsonl"), ORANGE))
_g21 = D / "g21_run1/g21/e1/task_results.jsonl"            # on-policy distillation; added once its results are downloaded
if _g21.exists():
    bars.append(("E4B + on-policy" + NL + "distillation" + NL + "(60 of 75 run)", R(_g21), AQUA))   # watchdog stopped the round at 60 tasks
summ = []
fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=200)
for i, (lab, rr, col) in enumerate(bars):
    rows = list(rr.values())
    res_n, pat, cap = sum(r["resolved"] for r in rows), sum(r.get("agent_patch_size", 0) > 0 for r in rows), sum(kind(r) == "C" for r in rows)
    summ.append((lab.replace(NL, " "), res_n, pat, cap, round(st.mean(r["duration_seconds"] for r in rows)), st.median(r["total_llm_calls"] for r in rows)))
    ax.bar(i, res_n, color=col, width=0.62)
    ax.text(i, res_n + 0.6, str(res_n), ha="center", va="bottom", color=INK, fontsize=10)
ax.set_xticks(range(len(bars)), [b[0] + NL + f"{x[2]} patches" + NL + f"{x[3]} capped" for b, x in zip(bars, summ)], fontsize=7.5)
ax.set_ylabel("tasks resolved (of 75)")
ax.set_ylim(0, 34)
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.set_title("E4B solves 7-10 tasks whatever the prompt; no distillation route raises the total", loc="left", fontsize=10.5, color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig6_small_model.png")
plt.close(fig)
data["fig6_small_model"] = [dict(zip(("round", "resolved", "patches_written", "capped", "mean_task_s", "median_calls"), x)) for x in summ]

# ------------------------------------------------------------ tables
tab = ["## Table 4. Small model (E4B) and E4B + LoRA on the 75 competition tasks (E4B rows: 8 min / 28 calls; 31B rows: 5-minute cap)", "",
       "| round | resolved | patches written | capped | mean task s | median model calls |", "|---|---|---|---|---|---|"]
tab += [f"| {a} | {b} | {c} | {d} | {e} | {f:g} |" for a, b, c, d, e, f in summ]
e4sets = [{t for t, r in v.items() if r["resolved"]} for v in e4.values()]
ever31 = {t for t in runs["base r1"] if any(runs[n][t]["resolved"] for n in runs)}
union = set().union(*e4sets)
tab += ["", f"E4B base rounds: union {len(union)} tasks; solved in at least 2 of the 3 rounds: {sum(sum(t in x for x in e4sets) >= 2 for t in runs['base r1'])}; "
        f"all {len(union)} are also solved by the 31B in some round: {union <= ever31}."]
g13 = json.load(open(D / "g13_run1_2026-10-23/g13/g13_summary.json"))["t4"]
sets75 = {"base r1 (T 0.2)": {t for t, r in runs["base r1"].items() if r["resolved"]}, "base r2 (T 0.2)": {t for t, r in runs["base r2"].items() if r["resolved"]},
          "G3 adapter (T 0.2)": {t for t, r in runs["G3 adapter"].items() if r["resolved"]}, "G11 adapter (T 0.2)": {t for t, r in runs["G11 adapter"].items() if r["resolved"]},
          "base T 1.0 (G13)": set(g13["resolved_ids"])}
ks = list(sets75)
tab += ["", "## Table 5. Tasks solved in exactly one of two rounds (31B, 75 competition tasks)", "", "| | " + " | ".join(k.split(" (")[0] for k in ks) + " |", "|---|" + "---|" * len(ks)]
for a in ks:
    tab.append(f"| {a.split(' (')[0]} | " + " | ".join(str(len(sets75[a] ^ sets75[b])) for b in ks) + " |")
data["table5_pairwise_flips"] = {a: {b: len(sets75[a] ^ sets75[b]) for b in ks} for a in ks}
ps = json.load(open(D / "pairs_p1p2p3/stats.json"))
tab += ["", "## Table 6. Preference pairs from the three pool rounds (31B, 171 tasks)", "", "| item | count |", "|---|---|",
        f"| attempts (solved / finished wrong / capped) | {ps['attempts']} ({ps['attempt_classes']['S']} / {ps['attempt_classes']['F']} / {ps['attempt_classes']['C']}) |",
        f"| final segments unusable (malformed history or over 16,384 tokens) | {sum(sum(v.values()) for v in ps['dropped_per_round'].values())} |",
        f"| solved-vs-unsolved pairs | {ps['selection']['main (S vs non-S) pairs']} ({ps['pairs_by_kind']['s_vs_c']} vs capped, {ps['pairs_by_kind']['s_vs_f']} vs wrong patch) |",
        f"| faster-vs-slower solved pairs | {ps['pairs_by_kind']['within_s']} |", f"| tasks with a pair | {ps['tasks_with_pairs']} |", f"| tokens, all pairs (M) | {ps['tokens_total_m']} |"]
(ROOT / "docs" / "paper" / "tables_2.md").write_text(NL.join(tab) + NL, encoding="utf-8")
(ROOT / "docs" / "paper" / "figure_data_2.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
print(NL.join(tab))
print(json.dumps({k: data[k] for k in ("fig5_pool_round_outcomes", "fig5_tasks_by_rounds_solved_of_3", "fig5_p1_vs_p2_flips")}))
