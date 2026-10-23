# Paper outline (draft 2, restructured 2026-10-13 around post-training)

> **2026-10-24: the working text is now `docs/paper/DRAFT.md` (draft 1, ~2,000 words, real numbers, E4B/temperature/preference placeholders). This outline is kept for the evidence ledger and schedule; the thesis was rewritten as: noise + bimodality, the clock, pipeline, honest LoRA nulls, verified-task construction, [E4B].**

> Draft 1 (2026-10-04) was a pure measurement study. After the user's decision that post-training the 31B is the primary focus, and after G1-G3 (docs/STRATEGY.md), the paper is rebuilt around a **practical, reproducible post-training pipeline for a 4-bit-served 31B SWE agent under a sequential wall-clock budget**, reported with an honest null result and the measurements that explain it. Draft 1's measurement material is kept as supporting evidence (Section 5-7). Nothing here may be claimed beyond the ledger's HAVE/DONE rows.

Deadline: **2026-11-12 23:59 UTC** for a *submitted* (not draft) Kaggle Writeup. Plan to submit by **Nov 10**.
Rules read from the paper-track page on 2026-10-04 (re-check before submitting; the organizers may update them):
- Kaggle Writeup, **3,000 words maximum**, original and unpublished; non-archival.
- Required: title and subtitle, abstract, introduction, methods and experiments, related work and citations. Optional: public notebook(s); an arXiv-style PDF in place of the Writeup text. Private resources attached to a public Writeup become public after the deadline.
- Scoring: five criteria, each 0-5, averaged: **Novelty, Quality (generality beyond the competition), Relevance, Verifiability, Clarity**.
- Prizes: Overall Best Paper $15k; Best New Resource $10k; Best New Application $10k.
- To be eligible we must **join the paper-track competition** (the "Join Hackathon" button): the user's click, still pending.

## Working title
**Training a 4-bit-Served 31B SWE Agent on Four L4s: a Token-Exact Post-Training Pipeline, and Why the Wall Clock Decides**
(alternate subtitle: *self-generated LoRA SFT, an honest null result, and the budget analysis behind it*)

## Thesis (what a reader should believe after 3,000 words)
1. Post-training a 31B model on a 4x24 GB machine for long-context agent trajectories is practical, but only with specific engineering (exact sequence-chunked layers, chunked loss, token-exact capture of what the model actually generated, adapter serving checks). We give the recipe and the measured limits.
2. Applying it in the simplest form (one epoch of LoRA SFT on the model's own verified successes) did **not** measurably improve the resolve rate. We say so, with paired outcomes and the noise floor that bounds what such an experiment can detect.
3. The reason is visible in the measurements: under a sequential wall-clock budget, resolution is decided by whether the agent *finishes in time*, not by correctness once it finishes. Interventions must therefore buy time or raise the finish rate; imitation of already-easy successes does neither.

## Contributions (claim only what the evidence supports; status in the ledger below)
1. **A reproducible post-training pipeline** for the competition's fixed 4-bit checkpoint: training on the unquantized bf16 sibling, staging from NFS, exact sequence-chunked decoder layers and chunked lm_head loss (ceiling 18,432 tokens on 4x L4 against about 10k stock), a logging proxy that captures **token ids exactly as served**, a construction that turns multi-turn trajectories into one sequence per context segment with loss on every assistant span, scorer-format adapter export, and fail-fast wiring checks (an adapter that silently does nothing is the known failure mode).
2. **What the pipeline costs and does at serving time**: adapter decode about 0.89x and KV cache 0.79x of the base at the scorer's slot sizing, round time about +7%.
3. **A null result with the noise floor**: G3 (one epoch, 70 sequences, 30 tasks) 28/75 against base 25 and 28; the same-prompt rerun noise (7 of 75 tasks flip between base rounds; earlier harness 23 vs 18 at n=48), so the design can only see effects of roughly 5+ tasks at n=75.
4. **The finish-rate finding**: in all three rounds every resolved task finished before the cap and every capped task failed (25/31, 28/36, 28/31 of finishers resolved; 44/39/44 capped, 0 resolved). Combined with the budget model (cap, 12 h, sequential), this reframes what to optimise.
5. **Benchmark validity and measurement practice** (from draft 1): only 75 of 129 public tasks pass their own reference patch in the scoring environment; pre-registered decision rules; paired designs.
6. *Resource (only if the data rules allow):* per-task solvability table, analysis scripts, training code (needed anyway: winners must open-source under an OSI licence). Raw traces and task text must NOT be released.

## Prize strategy
Aim at **Best Paper** with Quality-through-honesty and Verifiability as strengths; a reusable pipeline and the solvability table are the hook for **Best New Resource**. Do not claim a performance gain we did not measure. If G4 produces a real, replicated gain, the paper changes to lead with it; the rest of the structure holds.

## Section plan and word budget (sums to 3,000)
| § | Content | Words |
|---|---|---|
| Title, abstract | problem, pipeline, null result with numbers, finish-rate finding, release | 200 |
| 1 Introduction | local post-training under a budget; contribution list | 250 |
| 2 Setting and budget model | task, scorer (sequential, 12 h, graded as-is), cap arithmetic, the fixed 4-bit checkpoint | 250 |
| 3 Pipeline | bf16 training on the unquantized sibling; memory engineering and exactness tests; capture design; sequence construction; adapter export and wiring checks (Fig 1: pipeline; Table 1: memory/throughput) | 650 |
| 4 Serving costs | adapter decode and KV overhead, round time | 150 |
| 5 Experiment G3 | data (70 sequences / 30 tasks), recipe, results with paired counts (Table 2), noise floor | 400 |
| 6 Why: the wall clock | finish-rate table (Table 3), latency decomposition, thinking-off conservation, failure families | 450 |
| 7 Benchmark validity | solvability audit | 150 |
| 8 Related work | see list | 200 |
| 9 Limitations and threats | one model, 2 repos dominate, one epoch, one round, no hidden-set gain shown, community numbers unverified | 150 |
| 10 Conclusion and release | | 100 |

Figures and tables, about five: **Fig 1** pipeline diagram; **Table 1** memory/throughput by sequence length (stock vs chunked); **Table 2** base r1/r2 vs G3 e1 paired outcomes by task group; **Table 3** finish-rate vs resolved by round; **Fig 2** wall-time decomposition and cap sensitivity. Everything else in the public notebook/repository.

## Evidence ledger (be strict: nothing goes in the abstract unless it is HAVE or DONE)
| # | Claim | Evidence today | Status | What is still needed |
|---|---|---|---|---|
| P1 | Training the real 31B (bf16, LoRA r16 attn+MLP) works on 4x L4 at about 80 tok/s, peak 21 GiB/GPU, 35 steps with no OOM | G1d/G3 logs | HAVE | one clean throughput table by length |
| P2 | Sequence-chunked layers are exact vs stock and lift the ceiling from about 10k to 18,432 tokens (20,480 OOM) | tests on real dimensions + G1d | HAVE | include the exactness test numbers |
| P3 | Token-exact capture: all 2,951 + 1,500 requests carry prompt/completion ids; ids equal `/tokenize`; each prompt extends the previous token-exactly; history re-render differs by at most 1 token in 99% of pairs | G2 analysis | HAVE | |
| P4 | Adapters apply in the scorer-format path (canary adapter changes outputs; G3 smoke gate: 100% of main-agent requests use the adapter) | G1 canary, G3 smoke | HAVE | |
| P5 | Serving cost: decode 33.5 vs 37.5 tok/s (0.89x), KV 36,400 vs 46,352 tokens (0.79x); e1 round +7% wall | G1 probe, G3 | HAVE | |
| P6 | G3 result: 28/75 vs base 25, 28; 26/30 on trained tasks (= base's solved union); 2/45 on never-solved (consistent with chance: 7 tasks flip between base rounds) | G2/G3 summaries | HAVE | exact paired test; per-task table |
| P7 | Behaviour is statistically indistinguishable: tool mix, completion tokens per turn, LLM calls per task equal; decode per request about 10% slower | compare_rounds.py | HAVE | |
| P8 | Finish-rate: every resolved attempt finished before the cap; every capped attempt failed | task_results of r1, r2, e1 | HAVE | check what "resolved" does for a capped attempt that wrote a patch (8-11 per round) in the scorer, not only in our harness |
| P9 | Hidden-set effect of the adapter | LB submission 57109472 pending | NEED | LB score(s); repeats; SE about 0.045 |
| P10 | Any real gain from post-training | none | NEED | G4 or a different design |
| C1 | 96% of wall time is model latency, tools about 4% | 104 prior attempts + Exp1 traces | HAVE | bootstrap CIs, exact definition |
| C2 | Thinking-off: turn latency 4.7 s -> 2.2 s, calls/task 18.7 -> 52.7, task time 288 -> 263 s | Exp1, 30 paired tasks | HAVE (small n); community report agrees that thinking-off is much worse (23 vs 48 of 129, forum #746250, unverified) | |
| C3 | Two failure families (about 44% never edit; about 24% right place, incomplete) | 104 attempts, old harness | HAVE | re-estimate on G2/G3 traces |
| C7 | Solvability: 75/129 pass gold in the Kaggle env; fastapi failures 33/36 missing modules | gold-check notebook | HAVE | community confirms the cause (typing_inspection missing, forum #747248); staff says the TEST set validates at 100% and the training-task set is being fixed |
| C12 | Noise floor | B1 reruns 23 vs 18 (old harness, n=48); base rounds 25 vs 28 (n=75) | HAVE | |
| C13 | Time/cap sweep (community): 6 min cap 28.7% vs 15 min 41.9% on 129 public tasks, stronger teacher models only +4-5 tasks | forum posts #746250, #747168 | EXTERNAL, unverified | cite as community reports with the caveat; do not rely on them |

## Experiments still needed, in priority order
1. **Done offline:** round comparison (P6-P8). Remaining offline: per-task paired table; resolve rule for capped attempts with patches; recompute C1/C3 on G2/G3 traces.
2. **Leaderboard repeats** for the adapter (P9) as submission slots allow (pending-submission limit means one at a time).
3. **G4 (design pending user decision):** better data (more tasks, harder tasks, on-policy mix including the adapter's own successes), or objective changes aimed at finishing faster; or **a smaller-variant experiment** (the organizers added 12B/E4B/E2B on 2026-10-12; all variants run under the same budgets, so faster decoding may convert directly into more finished tasks; LoRA training is also far cheaper). Not promised in the paper until run.
4. **Not promised in the paper:** RL, mining new tasks from other repos.

## Related work to cover (200 words; **verify every citation before writing; I have not checked these**)
- Benchmarks and task validity: SWE-bench (Jimenez et al.), SWE-bench Verified, SWE-bench+.
- Scaffolds and agents: SWE-agent, Agentless, OpenHands.
- Training data and RL for SWE agents: SWE-Gym, R2E-Gym, SWE-smith, SWE-RL; rejection-sampling / self-distillation (STaR, expert iteration, RFT) for the self-generated-successes recipe.
- Parameter-efficient training: LoRA (Hu et al.), QLoRA; QAT checkpoints and why we train on the unquantized sibling.
- Long-context memory-efficient training (activation checkpointing, chunked attention/loss) to place our exact chunked layers.
- Graph-based repository navigation: "Codebase-Memory" (arXiv:2603.27277, unverified), community GraphLoc-129 (do not claim novelty over it). Graph tools did not help in our removal and in a community ablation (forum #746250, unverified).
- Anytime algorithms (Dean and Boddy; Zilberstein) for the budget/anytime framing.

## Criteria check (honest self-assessment)
- **Novelty:** moderate at best. The pipeline pieces are engineering; the defensible parts are the token-exact capture construction, the exact chunked layers at this scale, and the finish-rate framing. A null result is rarely scored as novel; frame it as "what a practitioner needs to know before spending 10 GPU-hours".
- **Quality (generality):** weak: one model, rich/fastapi dominate, one epoch, one round. State as functions of budget, not as universal claims.
- **Relevance:** good (practical local post-training, which is the stated purpose).
- **Verifiability:** strongest: scripts in the repository, hash-pinned artifacts, pre-registered decision rule (STRATEGY.md), exactness tests. Keep every number traceable.
- **Clarity:** three findings, one pipeline figure.

## Risks and open items
- **Data redistribution:** raw traces/token data contain task text and repository snippets from competition data; the competition forbids redistribution. Release code, IDs, aggregate metrics; keep traces in `data_extra/` (gitignored).
- **No gain yet:** if G4 also shows none, the paper is the pipeline + null result + budget analysis, which is a coherent but modest paper; do not manufacture a positive result.
- **Forum numbers are community reports**, not verified by us; cite as such or omit.
- **Quota:** about 14 h left by my estimate (verify). One more training cycle costs about 10 h.
- **Wheelhouse drift:** the harness changed on 2026-10-09 (0.2.7 -> 0.2.11); say which versions every result used. Pre-Oct-9 results are from the old harness.
- **Open-source requirement for prizes:** training code is already in the repository; add an OSI licence before release.

## Schedule (30 days from 2026-10-13)
| window | work |
|---|---|
| Oct 13-18 | offline analyses (P6-P8 details); LB result for the adapter; decide G4 design (and whether to test a smaller variant); join the paper track |
| Oct 18-Nov 1 | G4 (about 10 quota-hours) and its evaluation; LB repeats |
| Nov 1-7 | write the 3,000-word draft; build figures from scripts; licence + repository cleanup |
| Nov 8-10 | polish, check word count and citations, attach public notebook, **submit** |
| Nov 11-12 | buffer only |
