# Preference run (design, 2026-10-20) — gated, because the data says the signal is thin

Status: **design only**; nothing launched. GPU quota refreshes 2026-10-24T00:00 (30 h); this week's remainder (~3.4 h after G12) is not enough.

## 1. Why a preference run at all
Three SFT runs on the model's own verified successes (G3 28/75 + leaderboard 0.08, G11 22/75) did not beat the base (25-28/75): imitation of typical successes teaches nothing the model does not already do (loss halves in epoch 2 = memorisation), and the LoRA decode tax (~10%) turns directly into more capped attempts. A preference objective adds what SFT lacks: a **contrast on the same task** (this attempt finished and resolved, that one ran out of clock or submitted a wrong patch) and a direct handle on the clock (prefer the *faster* success).

## 2. What the existing data says about feasibility (measured 2026-10-20)
* Outcomes are strongly bimodal. Competition tasks, 4 attempts each (r1, r2, G3, G11): **43 never solved, 19 always solved, 13 mixed** (17%). Pool, one attempt (p1): 55 resolved, 30 finished-but-wrong, 86 capped.
* Expected mixed tasks from the empirical per-task solve distribution: k=2 attempts 12 of 171 pool tasks, k=3 ~18, k=4 ~21. **So 3 extra pool rounds (~20 quota h) would give only ~20 matched pairs at temperature 0.2.** That is too few for DPO, and the gradient signal of any offline RL on own samples is limited to ~15-20% of the tasks (tasks with solve probability 0 or 1 carry no contrast).
* Where the clock goes (G10, 171 attempts): capped attempts make FEWER calls than successes (median 11 vs 14) but each call is slower (19 s vs 12.6 s per call): **cap hits are caused by slow turns (long thinking/outputs), not by many turns**; 34 of 86 capped attempts never made a single edit; tool execution is negligible (2-8 s per attempt); the >30 s gaps are context-compaction summary calls.
* Community: nobody has published an adapter that beats the base (icfoer: LoRA on 697 trajectories = wash; E4B SFT = nothing). The 0.22-0.24 leaderboard plateau is config/prompt based (8 min / 28-30 calls).

## 3. Design that respects those facts
**Algorithm**: RPO = DPO + length-normalised NLL on the chosen trajectory (Pang et al. style), LoRA r16/alpha 32 on attn+MLP, beta 0.1, NLL weight 1.0, LR 2e-5 cosine, 2 epochs, accumulation 2 pairs. Reference = the same HF bf16 model with the adapter disabled (NOT vLLM logprobs: the rollouts come from the w4a16 model, the trainer runs bf16; mixing them gives a non-zero initial margin). Per pair the cost is ~10 units vs 8 for two SFT sequences (two-pass trick: no-grad forward of policy and reference to get the DPO weight, then one forward+backward per trajectory with a scalar weight, so only one autograd graph is alive; peak memory stays at the SFT level of ~21 GiB per GPU). Estimated ~7-8 min per 14k-token pair, so **50 pairs x 2 epochs ~ 12-14 quota h**.

**Preference labels** (per task, final context segment of each attempt, both <= 16,384 tokens):
* classes: S = resolved and finished, F = finished but wrong, C = capped. Chosen = the *fastest* S (fewest seconds; if two S differ by < 20% either is fine); rejected = the F or C attempt with the most model time. Up to 2 pairs per task. Within-S pairs (faster over slower by >= 30%) only if S-vs-non-S pairs are fewer than 40.
* all pairs come from POOL tasks; the 75 competition tasks stay untouched as the cross-repo evaluation set (base 25/28, G3 28, G11 22).

**Making the contrast rich enough (the bimodality problem)**: roll out at temperature 1.0 (top_p 0.95), the community's best sampling and the setting that decorrelates attempts; keep the agent prompt/budget identical to the config we intend to deploy (B1 or B2, decided from the B2 leaderboard result and G12). Whether temperature 1.0 raises the mixed fraction is unknown; that is what the gate measures.

## 4. Gated plan (each stage has a stop rule written before it runs)
| stage | when | cost | what | gate to continue |
|---|---|---|---|---|
| G13a probe | Oct 24 | ~7.4 q h (B1) / ~8 (B2) | one pool round `p2` at temperature 1.0, capture on | tasks with S in one of (p1, p2) and non-S in the other: **>= 25** (else the preference idea is dead: too little contrast) |
| G13b | Oct 24-25 | ~7.4 q h | pool round `p3` at temperature 1.0 | **>= 40 pairs after the pair rules** |
| G14 train+eval | Oct 31 | ~13 + 4.5 q h | RPO on the pairs, evaluate on the 75 competition tasks | see 5 |
| fallback | any stage | 0 | stop; keep B1/B2; paper reports the pipeline + four nulls + the clock finding | |
Total ~33 quota h over two weekly quotas (of ~150 left until Nov 25), leaving room for evaluations, leaderboard checks and the paper.

## 5. Decision rule for the trained adapter (before seeing it)
Reference on the 75 competition tasks (same harness/prompt/cap): base 25 and 28 resolved, 44/39 capped; adapters 28 (G3) and 22 (G11). Interesting only if **resolved >= 34 or capped <= 35 with resolved not below 28**, and calls per task not higher; the LoRA tax (~10% slower decode) must be paid for by shorter episodes. Then sweep the inference-time LoRA scale (alpha/r in `adapter_config.json`: 1.0 and 0.5 are free restarts), then leaderboard with repeats. Anything inside 22-30 is a null.

## 6. Risks
* Too few pairs even at temperature 1.0 (the gate catches it after ~7 quota h).
* Trajectory-level DPO has weak credit assignment (the preferred and rejected trajectories differ in dozens of turns); known to give small gains (ETO-style).
* Length/behaviour drift (the adapter must not learn to stop early without a patch): the NLL term and the S-only chosen set limit this; check "patch written" and "submitted" rates, not only resolved.
* Compaction contexts: pairs are built from final segments whose contexts start with different summaries; fine for DPO (each log-prob uses its own context) but the builder must keep only segments with intact token-exact history.

## 7. Cheaper alternatives to weigh against it
1. **Config first (free)**: thinking budget/output cap (G12 running), the 8 min / 28 call budget (B2 on the leaderboard), sampling temperature. These change cap hits directly and carry no decode tax.
2. **Teacher distillation on never-solved tasks**: the only source of NEW capability (teachers solved 10 of 60 tasks Gemma failed twice; forum). Upper bound ~ +5% absolute, costly (a 27B teacher through our harness), conversion into Gemma's template needed.
3. **Budget-conditioned RFT with short thoughts**: roll out under the best thinking budget, SFT on the resolved ones: equals the config unless training generalises past truncation; do only if G12 shows that shorter thinking converts into finishers.
