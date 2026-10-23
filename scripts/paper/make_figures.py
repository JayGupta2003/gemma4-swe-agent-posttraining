"""Figures and tables for the paper, computed from the raw result files in data_extra/ (gitignored) and docs/. Run with the `mlai` conda python (matplotlib).

Usage: python scripts/paper/make_figures.py
Outputs: docs/paper/figures/fig1_bimodal.png, fig2_clock.png, fig3_sft_nulls.png, fig4_task_funnel.png and docs/paper/tables.md (+ docs/paper/figure_data.json with every plotted number).

Design (dataviz skill): one measure per axis (no dual axes), thin marks, direct labels, categorical slots 1-3 of the validated palette (blue, orange, aqua; aqua is below 3:1 on the
light surface so every aqua mark is directly labelled), text in ink colours never in series colours, emphasis (one hue + gray) where one series is the point.
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
OUT.mkdir(parents=True, exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b0"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def R(p):
    return {json.loads(l)["instance_id"]: json.loads(l) for l in open(p, encoding="utf-8")}


def kind(r):
    return "S" if r["resolved"] else ("C" if "timeout" in (r.get("error") or "") else "F")


runs = {"base r1": R(D / "g2_run1_2026-10-11/g2/r1/task_results.jsonl"), "base r2": R(D / "g2_run1_2026-10-11/g2/r2/task_results.jsonl"),
        "G3 adapter": R(D / "g3_run1_2026-10-12/g3/e1/task_results.jsonl"), "G11 adapter": R(D / "g11_run1_2026-10-20/g3/e1/task_results.jsonl")}
b12 = R(D / "g5_run1_2026-10-13/g5/b12/task_results.jsonl")
p1 = R(D / "g10_run1_2026-10-19/g10/p1/task_results.jsonl")
data = {}

# ---------------------------------------------------------------- Fig 1: bimodal outcomes over 4 attempts
tasks = sorted(runs["base r1"])
frac = collections.Counter(sum(kind(runs[n][t]) == "S" for n in runs) for t in tasks)   # number of solved attempts out of 4
xs = [0, 1, 2, 3, 4]
ys = [frac.get(k, 0) for k in xs]
data["fig1_tasks_by_solved_attempts_of_4"] = dict(zip(map(str, xs), ys))
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
cols = [BLUE if k in (0, 4) else GRAY for k in xs]
ax.bar(xs, ys, color=cols, width=0.62)
for x, y in zip(xs, ys):
    ax.text(x, y + 0.8, str(y), ha="center", va="bottom", color=INK, fontsize=10)
ax.set_xticks(xs, ["never\n(0 of 4)", "1 of 4", "2 of 4", "3 of 4", "always\n(4 of 4)"])
ax.set_ylabel("competition tasks (of 75)")
ax.set_ylim(0, 52)
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.set_title("Outcomes are bimodal: only 13 of 75 tasks are mixed", loc="left", fontsize=11, color=INK)
ax.text(0.99, 0.93, "4 attempts per task: two base rounds and two adapter rounds\n(temperature 0.2, 5-minute cap)", transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig1_bimodal.png")
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: where the clock goes (G10 pool round p1): calls per attempt and model seconds per call, by outcome class
import gzip

cand = {json.loads(l)["instance_id"]: json.loads(l) for l in open(D / "ext_datasets/candidates.jsonl", encoding="utf-8")}
MAIN = "You are an expert autonomous software engineer"


def txt(m):
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c)


keys = collections.defaultdict(list)
ktask, firstu = {}, {}
with gzip.open(D / "g10_run1_2026-10-19/g10/capture.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        if rec["round"] != "p1":
            continue
        m = rec["request"]["messages"]
        if not next((txt(x) for x in m if x.get("role") == "system"), "").startswith(MAIN):
            continue
        k = rec["task_key"]
        if k not in firstu:
            firstu[k] = next((txt(x) for x in m if x.get("role") == "user"), "")
            hit = [i for i, t in cand.items() if t["problem_statement"][:300] and t["problem_statement"][:300] in firstu[k]]
            ktask[k] = hit[0] if len(hit) == 1 else None
        keys[k].append((rec["t0"], rec["t1"]))
by_task = collections.defaultdict(list)
for k, v in keys.items():
    if ktask[k]:
        by_task[ktask[k]] += v
agg = collections.defaultdict(lambda: {"calls": [], "llm": [], "n": 0})
for t, v in by_task.items():
    c = kind(p1[t])
    agg[c]["calls"].append(len(v))
    agg[c]["llm"].append(sum(b - a for a, b in v))
    agg[c]["n"] += 1
order = [("S", "resolved", BLUE), ("F", "finished,\nwrong patch", ORANGE), ("C", "hit the\n5-minute cap", AQUA)]
med_calls = [st.median(agg[k]["calls"]) for k, _, _ in order]
sec_call = [st.mean(agg[k]["llm"]) / st.mean(agg[k]["calls"]) for k, _, _ in order]
data["fig2_pool_round"] = {k: {"n": agg[k]["n"], "median_calls": st.median(agg[k]["calls"]), "model_seconds_per_call": round(st.mean(agg[k]["llm"]) / st.mean(agg[k]["calls"]), 1)} for k, _, _ in order}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=200)
for ax, vals, title, fmt in ((axes[0], med_calls, "median model calls per attempt", "{:.0f}"), (axes[1], sec_call, "model seconds per call", "{:.1f}")):
    ax.bar(range(3), vals, color=[c for _, _, c in order], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_xticks(range(3), [f"{lab}\n(n={agg[k]['n']})" for k, lab, _ in order], fontsize=8)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
fig.suptitle("Capped attempts make fewer calls, but each call is slower", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "fig2_clock.png")
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: SFT nulls (resolved of 75, same harness and cap)
names = ["base r1", "base r2", "G3 adapter", "G11 adapter"]
res = [sum(r["resolved"] for r in runs[n].values()) for n in names]
cap = [sum(kind(r) == "C" for r in runs[n].values()) for n in names]
data["fig3_resolved_of_75"] = dict(zip(names, res))
data["fig3_capped_of_75"] = dict(zip(names, cap))
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
cols = [GRAY, GRAY, BLUE, BLUE]
ax.bar(range(4), res, color=cols, width=0.6, zorder=2)
for i, (v, c) in enumerate(zip(res, cap)):
    ax.text(i, v + 0.6, str(v), ha="center", va="bottom", color=INK, fontsize=10)
    ax.text(i, 1.2, f"{c} capped", ha="center", va="bottom", color="white" if cols[i] == BLUE else INK, fontsize=8)
ax.axhspan(25, 28, color=GRAY, alpha=0.25, linewidth=0, zorder=0)
ax.text(3.45, 28.4, "base range 25-28", ha="right", va="bottom", color=INK2, fontsize=8)
ax.set_xticks(range(4), ["base\nround 1", "base\nround 2", "G3 adapter\n(30 trained tasks)", "G11 adapter\n(54 other repos)"], fontsize=8)
ax.set_ylabel("tasks resolved (of 75)")
ax.set_ylim(0, 36)
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.set_title("LoRA SFT on the model's own successes does not beat the base", loc="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig3_sft_nulls.png")
plt.close(fig)

# ---------------------------------------------------------------- Fig 4: task funnel (one hue, sequential)
res8 = json.load(open(D / "g9_run1_2026-10-13/gold_results.json"))
pool = json.load(open(ROOT / "docs" / "g9_rollout_pool.json"))
g8 = []
for s in (0, 1):
    g8 += json.load(open(D / f"g8_run1_2026-10-13/g8_{s}/scale_results_{s}.json", encoding="utf-8"))
valid = sum(r["verdict"] == "valid" for r in g8)
usable = sum(r["verdict"] in ("valid", "p2p_partial") for r in g8)
stages = [("candidates (SWE-rebench filtered split,\nquality and size filters)", len(g8)), ("validated in a clean Python 3.12 venv\n(valid + nearly valid)", usable),
          ("snapshotted in the competition layout", len(res8)), ("gold patch passes the harness check", sum(r["gold"]["resolved"] for r in res8.values())),
          ("... and an empty patch does not (rollout pool)", len(pool)), ("resolved by the unchanged 31B (5-min cap)", sum(r["resolved"] for r in p1.values()))]
data["fig4_funnel"] = {k.split("\n")[0]: v for k, v in stages}
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
vals = [v for _, v in stages]
ax.barh(range(len(stages)), vals, color=BLUE, height=0.58)
for i, v in enumerate(vals):
    ax.text(v + 10, i, f"{v}", va="center", color=INK, fontsize=10)
ax.set_yticks(range(len(stages)), [k for k, _ in stages], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, max(vals) * 1.12)
ax.xaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("tasks")
ax.set_title("From 898 candidates to 171 harness-verified tasks", loc="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig4_task_funnel.png")
plt.close(fig)

# ---------------------------------------------------------------- tables
subs = [("baseline (starter prompt)", "5 min cap", "0.12"), ("B1", "5 min cap", "0.13 and 0.15"), ("G3 adapter on B1", "5 min cap", "0.08"),
        ("B2", "8 min / 28 tool calls", "0.17"), ("B3 (B2 + thinking 2048)", "8 min / 28 tool calls", "pending")]
tab = ["## Table 1. Public leaderboard (58 tasks; one task = 0.017)", "", "| submission | budget | public score |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in subs]
tab += ["", "## Table 2. Local rounds on the 75 competition tasks (same harness, prompt, 5-minute cap)", "", "| round | resolved | patches written | capped | LLM calls / task |", "|---|---|---|---|---|"]
for n in names:
    rows = list(runs[n].values())
    tab.append(f"| {n} | {sum(r['resolved'] for r in rows)} | {sum(r.get('agent_patch_size', 0) > 0 for r in rows)} | {sum(kind(r) == 'C' for r in rows)} | {st.mean(r['total_llm_calls'] for r in rows):.1f} |")
rows = list(b12.values())
tab.append(f"| 12B base | {sum(r['resolved'] for r in rows)} | {sum(r.get('agent_patch_size', 0) > 0 for r in rows)} | {sum(kind(r) == 'C' for r in rows)} | {st.mean(r['total_llm_calls'] for r in rows):.1f} |")
tab += ["", "## Table 3. 31B vs 12B (4x L4; speeds measured single-stream in G5; the tokens-per-turn ratio is from the G5 profile)", "", "| | 31B | 12B |", "|---|---|---|",
        "| decode speed (tokens/s) | 37.5 | 79.2 |", "| completion tokens per call | 1x | 1.6-1.8x |", "| capped attempts (of 75) | 44 / 39 | 57 |", "| resolved (of 75) | 25 / 28 | 17 |"]
(ROOT / "docs" / "paper" / "tables.md").write_text("\n".join(tab) + "\n", encoding="utf-8")
(ROOT / "docs" / "paper" / "figure_data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
print(json.dumps(data, indent=1))
