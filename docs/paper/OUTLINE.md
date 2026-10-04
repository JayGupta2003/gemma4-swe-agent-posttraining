# Paper outline (draft 1, 2026-10-04)

Deadline: **2026-11-12 23:59 UTC** for a *submitted* (not draft) Kaggle Writeup. Plan to submit by **Nov 10**.
Rules read from the paper-track page on 2026-10-04 (re-check before submitting; the organizers may update them):
- Kaggle Writeup, **3,000 words maximum**, original and unpublished; non-archival (does not block other venues).
- Required: title and subtitle, abstract, introduction, methods and experiments, related work and citations.
- Optional: public notebook(s); an arXiv-style PDF in place of the Writeup text. Private resources attached to a public Writeup become public after the deadline.
- Scoring: five criteria, each 0-5, averaged: **Novelty, Quality (generality beyond the competition), Relevance, Verifiability, Clarity**.
- Prizes: Overall Best Paper $15k; Best New Resource $10k; Best New Application $10k (a paper competes for all three). Open question: whether judges will take one paper for several categories. Participation was thin on 2026-10-04 (113 teams, 116 submissions).
- To be eligible we must **join the paper-track competition** (the "Join Hackathon" button). That accepts rules, so it is the user's click.

## Working title
**Where the Clock Goes: Wall-Clock Budgets, Anytime Patches and Tool Interfaces for a 31B Local SWE Agent**
(alternate subtitle: *a measurement study on Gemma 4 under a 12-hour sequential budget*)

## Thesis (what a reader should believe after 3,000 words)
Under a hard sequential wall-clock budget, a mid-size quantized model does not fail mainly at localization or for lack of clever navigation tools. It fails because (1) almost all time is model latency, so the agent often never reaches an edit, and (2) when it does edit, the fix is in the right place but incomplete. Honest measurement of those two failure families, plus an audit of how many benchmark tasks are solvable at all in the scoring environment, changes which interventions are worth trying.

## Contributions (claim only what the evidence supports; status in the ledger below)
1. **A budget model and "anytime patch" framing.** The scorer grades whatever is in the working tree at the cap, so agents should be evaluated by *patch-by-time* and *resolved-at-cap* curves, not only final pass rate. A closed-form cap/12-hour budget analysis for a sequential scorer, with measured overheads.
2. **Where the time goes.** Per-turn latency decomposition from step-timed traces (model vs tools), the thinking-off result (turn latency about 2x lower, tool calls about 3x higher, wall time almost unchanged: latency is conserved, not removed), time-to-first-edit as the mechanism, and the tool-call mix of stuck vs resolved attempts.
3. **Two failure families**, via reference-patch comparison: never-edits (about 44% of attempts) vs right-region-incomplete-fix (81% of non-resolved patches hit the gold region; failures touch fewer files than the gold fix, 26% vs 4% for resolved).
4. **A benchmark-validity audit.** Only 75 of 129 public tasks pass their own reference patch in the scoring environment (52 locally); of the 36 fastapi failures, 33 are `ModuleNotFoundError` for a test dependency missing from the sandbox (27 `inline_snapshot`, 6 `dirty_equals`), so no agent can solve those tasks. Effective ceiling and the effect on how a team should choose dev tasks.
5. **Tool-interface failure modes for small-context agents** (the graph-tool and truncation hazards), with a graph-tools on/off ablation to test whether graph navigation helps at all in this regime.
6. **Methodological contribution:** pre-registered decision rules, paired designs, an estimate of the run-to-run noise floor at n of roughly 50, and negative results reported as such.
7. *Resource (if the data rules allow):* the Kaggle-environment solvability table, per-task outcomes, and the analysis toolkit.

## Prize strategy
Aim at **Best Paper** (all five criteria averaged) with the resource contribution (items 4, 7) as the hook for **Best New Resource**. Do not pitch an application paper: we have no novel use case. Not chasing the "graph reasoning" topic head-on unless the graph ablation (E3) shows something real.

## Section plan and word budget (sums to 3,000)
| § | Content | Words |
|---|---|---|
| Title, abstract | problem, 3 findings with numbers, what is released | 200 |
| 1 Introduction | local agents, the budget regime, why a measurement study; contribution list | 300 |
| 2 Setting and budget model | task, scorer (sequential, 12 h, graded-as-is), cap/12-h arithmetic, anytime framing | 300 |
| 3 Instruments | harness, ATIF traces with step timing, paired design, pre-registration, tasks (Kaggle-env-gold-verified pool) | 250 |
| 4 Where the clock goes | latency decomposition; thinking-off conservation; cap-sensitivity curve (Fig 1, Fig 2) | 450 |
| 5 Why attempts fail | two families; near-miss analysis; call-mix of stuck vs resolved | 350 |
| 6 Interventions | prompt ladder (edit-early, no-repro, completeness), graph on/off ablation, noise floor (Table 2) | 450 |
| 7 Benchmark validity | solvability audit, categories of failure, consequences | 250 |
| 8 Related work | see list | 200 |
| 9 Limitations and threats | one model, two dominant repos, small n, no trained result | 150 |
| 10 Conclusion and release | | 100 |

Figures and tables, kept to about five: **Fig 1** wall-time decomposition (model vs tool) per arm; **Fig 2** patch-by-time and resolved-at-cap curves with cap sensitivity; **Table 1** thinking-off latency/calls/wall time; **Table 2** prompt ladder results with paired counts and exact p-values; **Table 3** solvability audit by repo and failure category. Everything else goes in the public notebook and repository, since the word limit is tight.

## Evidence ledger (be strict: nothing goes in the abstract unless it is HAVE or DONE)
| # | Claim | Evidence today | Status | What is still needed |
|---|---|---|---|---|
| C1 | 96% of wall time is model latency, tools about 4% | 104 prior attempts + Exp1 traces | HAVE | recompute from Exp1+Exp2 traces with bootstrap CIs; pin the exact definition |
| C2 | Thinking-off: median turn latency 4.7 s -> 2.2 s, calls/task 18.7 -> 52.7, mean task time 288 s -> 263 s | Exp1, 30 paired tasks, 4 arms | HAVE (small n) | optional replication in Exp2 session; report that outcome effect is null |
| C3 | Two failure families (about 44% never edit; about 24% right place, incomplete) | 104 attempts | HAVE | re-estimate on Exp2 (about 190 more attempts) |
| C4 | 81% of non-resolved patches hit the gold region; 26% vs 4% touch fewer files than gold | near-miss script, n = 31 vs 26 | HAVE (small n) | pool with Exp2 attempts; report counts, not just percentages |
| C5 | Edit-early cuts median time to first edit 151 s -> 102 s | Exp1 A1 vs A0 | HAVE | **replicate** in Exp2 (B1) on fresh tasks |
| C6 | No-repro-first and completeness-grep prompts change patch rate or resolve rate | none | NEED | Exp2 B2/B3 (prepared, not run) |
| C7 | Solvability: 75/129 pass gold in Kaggle env, 52 locally; fastapi failures are 33/36 missing modules (27 `inline_snapshot`, 6 `dirty_equals`) | gold-check notebook, categorized from the stored output tails | HAVE | categorize the remaining 21 (requests fixture errors, rich/requests test failures) by reading full logs, and confirm the scorer env matches the notebook env |
| C8 | Tool hazards (uncapped `search_similar_code` can overflow 32k context and lose the patch; pytest verdict truncated; graph has no async nodes, one edge type) | engineering notes from runs 1-5 | HAVE as anecdote | turn into numbers: graph-tool call counts per attempt; **graph on/off ablation (E3)** |
| C9 | Cap sensitivity: share of resolved attempts whose last edit lands before cap t | traces exist | NEED, **free** | offline script over Exp1 traces + earlier attempts. Caveat: gives a *lower bound* on resolved-at-t, since an earlier snapshot could also pass |
| C10 | 12-hour budget model fits the real run | arithmetic + the 0.12 submission completing | PARTIAL | pull the real wall time of submission 56789021 if the API exposes it |
| C11 | Dev-set resolve rate vs hidden-test leaderboard (about 20% dev vs 0.12 public LB) | two numbers | WEAK | only as a generality datum; hidden repos differ, so say so explicitly |
| C12 | Run-to-run noise floor of an A/B at n of roughly 50 | none directly | NEED | confirmation round reruns B0 on the 27 Exp1 tasks, where A0 already ran: a free test-retest pair. Optionally repeat B0 on the 48 |

## Experiments still needed, in priority order
1. **Offline, no GPU (do first):** C9 cap-sensitivity curve; C7 finish categorizing the non-fastapi gold failures; C1/C3/C4 recomputed with bootstrap CIs on all traces; patch-by-time CDF by arm and outcome.
2. **E2 (prepared, about 3.5-4 h L4x4):** prompt ladder on the 48 fresh tasks. Then the 27-task confirmation round (B0 plus the best candidate), which also provides C12.
3. **E3 (to design, about 2 h L4x4):** graph tools on vs off, same 48 tasks, best prompt. This is the experiment that connects to the track's graph-reasoning topic. A null result is still a reportable result.
4. **Generality (optional):** run the best arm and control on SWE-Pallets-X tasks (other repos). Needs era-aware wheels for the no-internet Kaggle environment, which is real work; cut first if time is short.
5. **Not promised in the paper:** SFT/LoRA. Mention only if the feasibility spike and a real run happen in time.

## Related work to cover (200 words; **verify every citation before writing; I have not checked these**)
- Benchmarks and task validity: SWE-bench (Jimenez et al.), SWE-bench Verified curation, SWE-bench+ (solution leakage and weak tests). Our audit is the same kind of question asked of a sandboxed re-run.
- Scaffolds and agents: SWE-agent (agent-computer interface), Agentless, OpenHands.
- Training data and RL for SWE agents: SWE-Gym, R2E-Gym, SWE-smith, SWE-RL.
- Graph-based repository navigation: the "Codebase-Memory" paper the plan cites (arXiv:2603.27277, unverified), and the community **GraphLoc-129** dataset (must be differentiated; do not claim novelty over it).
- Anytime algorithms (Dean and Boddy; Zilberstein) for the budget/anytime framing.
- Latency and speculative/efficient inference only if a sentence is needed to justify the latency-conservation point.

## Criteria check (honest self-assessment, to revisit after Exp2)
- **Novelty:** moderate. Prompt tweaks alone are not novel; the budget/anytime framing, latency-conservation result and solvability audit are the defensible parts. Risk: reads as engineering notes.
- **Quality (generality):** the weakest criterion. One model, two repos make up 74 of 75 usable tasks. Mitigate by stating results as functions of latency and budget (transferable), reporting per-repo splits, and running the Pallets generality check if feasible.
- **Relevance:** good (offline agents on consumer hardware is the stated goal).
- **Verifiability:** our strongest. Pre-registered rules in git, hash-pinned artifacts, one-command scripts, public notebooks for gold check and experiments. Keep it that way: every number in the paper traceable to a script in the repository.
- **Clarity:** the 3,000-word limit forces a three-finding story. Resist adding sections.

## Risks and open items
- **Data redistribution:** raw traces contain task text and repository snippets from competition data. I have **not** read the data-use rules on sharing derived data; check before releasing traces. Release IDs, metrics and aggregate statistics regardless.
- **Quota:** L4x4 notebooks bill at about 2x; the weekly quota is unknown to me. E2 + confirmation + E3 is roughly 8-9 h of L4x4. Confirm remaining quota before committing.
- **Overfitting the story to a null:** if Exp2 shows no effect for any arm, the paper's contribution is the measurement (C1-C5, C7, C9, C12) and says so plainly. That is still a coherent paper; do not manufacture a positive result.
- **Main-track conflict:** the same GPU sessions serve both tracks, and anything that changes `submission/` must pass the confirmation rule first.

## Schedule (39 days from today)
| window | work |
|---|---|
| Oct 5-10 | offline analyses (C9, C7 categories, bootstrap CIs); join the paper track; read data-use rules |
| next L4x4 day | E2 discovery run; analyze same day |
| by Oct 20 | confirmation round (C12); decide E3 design |
| Oct 20-Nov 1 | E3 (and Pallets generality if feasible) |
| Nov 1-7 | write the 3,000-word draft; build figures from scripts |
| Nov 8-10 | polish, check word count and citations, attach public notebook, **submit** |
| Nov 11-12 | buffer only |
