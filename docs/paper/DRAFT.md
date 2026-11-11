# Where the Gains Aren't: Noise, Bimodality and the Limits of LoRA Post-Training for a 4-bit Gemma 4 SWE Agent

*Subtitle (Kaggle subtitle field): a reproducible post-training pipeline on four 24 GB GPUs, a harness-verified task pool, and what a 12-hour sequential budget does to evaluation*


## Abstract

Can a 4-bit Gemma 4 agent that fixes real GitHub issues offline be improved by post-training on four 24 GB GPUs? We study this under the Gemma 4 Developer Agent Competition's rules: a fixed checkpoint, a sequential scorer with a 12-hour budget, and hidden tasks. Evaluation is dominated by noise and task bimodality: one agent bundle scored 0.17 and 0.13 on the public leaderboard, and 62 of 75 local tasks have the same outcome in four attempts. Time binds through budgets, not speed: a 2.1x faster model solves fewer tasks, and in the scorer's regime the 28-tool-call cap ends a third of attempts, yet raising it overran the 12 hours. With a pipeline for token-exact capture, long-sequence LoRA training and verified adapter serving, SFT and preference training on the model's own trajectories do not beat the 31B base (22-28 vs 25-28 of 75). For the small E4B model, imitating 31B successes or rewritten Qwen3-Coder trajectories does not help either, the latter because a hindsight leak made 92% of its edits fail; on-policy distillation, where the 31B grades every token of E4B's own rollouts, cut the divergence by two thirds but imported the teacher's self-revising deliberation: 52 of 60 attempts hit the time cap, and capping the reasoning restored speed but not results. We release the code, a 171-task harness-verified pool and a pre-registered protocol.

## 1. Introduction

The competition asks for an autonomous coding agent that runs offline on everyday hardware: a fixed 4-bit Gemma 4 model, adaptable only through prompts, tools, sampling, per-task limits and LoRA adapters, scored sequentially on hidden issues from unseen repositories within 12 hours. Prompt-only agents cluster at 0.15-0.24 on the public leaderboard and nobody has reported a LoRA that beats the base model. We set out to find post-training that works and to explain why it is hard. Every evaluation had a decision rule written down before its result (the repository's commit dates show the order).

**Findings.**
1. *Evaluation is the bottleneck.* Identical files score 0.17 and 0.13; most tasks are always or never solved, which bounds anything learned from a model's own samples (Section 3).
2. *Budgets, not speed, decide outcomes.* Faster decoding and shorter thinking do not convert failures; in sequential scoring the tool-call cap binds and also governs total runtime (Section 4).
3. *LoRA imitation and preference training of the 31B model are nulls* even on 54 unseen repositories, and adapters cost ~10% decode speed (Sections 5-6).
4. *Harness-verified tasks* need the scorer's own sandbox conventions; we build 171 from SWE-rebench (Section 7).
5. *Distilling into the small model:* sequence imitation shifts behaviour without adding skill, trajectory rewriting must not leak the answer, and token-level on-policy distillation imports the teacher's deliberation style along with its token choices, which under a time budget is a loss even with reasoning capped (Section 8).

Code (Apache-2.0): https://github.com/JayGupta2003/gemma4-swe-agent-posttraining; competition data are not redistributed.

## 2. Setting

The agent is a Google ADK single agent with tools run_command, read_file, edit_file, write_file and submit_patch; the model gemma-4-31b-it-qat-w4a16-ct runs on vLLM (tensor parallel 4, 32,768-token context, event compaction at 14,336 tokens). A submission resolves a task if its patch makes the hidden tests pass; one submission per day. Our baseline uses temperature 0.2, thinking budget 4,096, output cap 8,192 and a 5-minute cap; the community recipe (B2) uses 8 minutes and 28 tool calls. Local evaluation runs the scorer's harness (swegemma 0.2.11) on the 75 competition tasks whose reference patch passes in our environment, four attempts concurrently, which is ~1.5x harsher per cap-minute than the sequential scorer (24 vs 37 tokens/s per request). Appendix A lists every run with data, hyperparameters, compute, rule and result.

## 3. Noise and bimodality

*Leaderboard.* Our submissions scored 0.12 (baseline), 0.13 and 0.15 (5-minute prompt), and 0.17 and 0.13 for two runs of byte-identical B2 files, on 58 public tasks (one task = 0.017). Community reports agree (7-13 of 58 for identical files; configurations tied locally scoring 0.17 and 0.10 hidden). Differences of 0.02-0.04 are indistinguishable.

*Local.* Two rounds of the same agent solve 25 and 28 of 75; 7 of the 30 tasks ever solved flip. With four attempts, 43 tasks are never solved, 19 always and 13 are mixed (Figure 1). Tasks with solve probability 0 or 1 carry no contrast for learning from the model's own samples. *Temperature 1.0* (top_p 0.95, top_k 64) resolves 29/75 and raises round-to-round disagreement by ~50% (9-13 vs 5-8 tasks) without raising the count. On the 171-task pool, rounds at 0.2, 1.0 and 1.0 resolve 55, 55 and 52; 100 tasks are never solved, 34 always, and 24 of the 26 tasks that differ between the first two rounds are *solved vs ran out of clock*, not *solved vs wrong patch* (Figure 2).

![Figure 1](figures/fig1_bimodal.png)
*Figure 1. Number of attempts (of four) that resolved each of the 75 competition tasks. Blue: the 62 tasks whose outcome never changed.*

| submission | per-task limits | public score |
|---|---|---|
| baseline | 5 min | 0.12 |
| 5-minute prompt (B1) | 5 min | 0.13, 0.15 |
| B1 + SFT adapter | 5 min | 0.08 |
| B2 | 8 min / 28 tool calls | 0.17, 0.13 (identical files) |
| B3 (B2 + thinking 2,048) | 8 min / 28 tool calls | 0.15 |
| B4 (B3 + 34 tool calls) | 8 min / 34 tool calls | exceeded 12 h, no score |
| E4B base, B2 prompt | 8 min / 28 tool calls | 0.06 |

*Table 1. Our public leaderboard scores (58 tasks; one task = 0.017).*

![Figure 2](figures/fig5_pool_rounds.png)
*Figure 2. The 171-task pool in three rounds of the 31B (temperature 0.2, 1.0, 1.0; 5-minute cap). Left: outcomes. Right: in how many rounds each task is solved.*

## 4. Where the clock goes

In 171 pool attempts (5-minute cap), all 55 resolved attempts finished before the cap and all 86 capped attempts failed, including 51 that had written a patch. Capped attempts make fewer calls than successes (median 11 vs 14) but each is slower (18 s vs 12 s of model time), and 34 of 86 never edited anything (Figure 3). Across 3,023 requests, time = 0.74 s + 0.042 s per completion token (R² 0.988): decoding is 98% of model time, prefill ~0 (prefix caching), and reasoning is 74% of generated text. Tool execution is negligible (2-8 s per attempt), and the code-graph tools the harness offers were called twice in 104 early attempts.

*Faster is not better.* The 12B variant decodes 2.1x faster but writes 1.6-1.8x more per call, is capped more often (57/75 vs 44 and 39) and resolves 17/75 vs 25 and 28. Thinking budgets of 2,048 and 1,024 make 31B requests ~20% faster without losing accuracy (12 and 11 of 32 vs 10 and 11) but leave the capped share unchanged.

*The scorer's regime.* Run sequentially like the scorer (B2, 24 tasks), the 31B resolves 9/24, in line with local rounds, but the binding limit changes: the 28-tool-call cap ends 9 attempts, more than the clock (5), 6 of them with an empty patch and 4 with 2.8-5.3 minutes left. Raising the cap to 34 (B4) exceeded the 12-hour total on the hidden set: the per-task call cap is also what keeps the whole run inside the budget, and local timing does not predict the hidden run's length.

![Figure 3](figures/fig2_clock.png)
*Figure 3. 171 pool attempts by outcome: median model calls per attempt (left) and mean model seconds per call (right).*

## 5. A pipeline for post-training the 31B model on four 24 GB GPUs

*Training model.* The scored checkpoint is 4-bit and cannot take LoRA; we train its bf16 QAT-unquantized sibling with LoRA (r=16, alpha=32, attention+MLP, 122M parameters), staging the 62.5 GB checkpoint from network storage to local disk (200 s). Stock layers fit ~10k tokens; exact sequence-chunked decoder layers with per-chunk checkpointing (verified equal to stock at the real dimensions) and a chunked lm-head loss raise the ceiling to 18,432 tokens at 81 tokens/s. A genuine out-of-memory error leaves tensors allocated under non-reentrant checkpointing, so we train first and probe limits afterwards.

*Token-exact capture.* A logging reverse proxy between the harness and vLLM records every request with exact prompt and completion token ids (identical to /tokenize). Each prompt extends the previous one token-exactly; re-rendered assistant turns differ from the generated tokens by at most one token in 99% of pairs. Training sequences are the final prompt plus completion, with loss on every assistant span.

*Serving.* Adapters are exported in the scorer's format and verified to be applied (a canary changes outputs; every smoke round requires >=95% of agent requests to use the adapter). With an adapter vLLM decodes 33.5 vs 37.5 tokens/s and the KV cache holds 21% fewer tokens.

## 6. LoRA post-training of the 31B does not help

*SFT on the base's own successes:* 70 sequences (0.83M tokens) from the 30 tasks the base solved, one epoch: 28/75, same as the base; leaderboard 0.08. *SFT across repositories:* 71 sequences from 54 tasks in 54 other repositories, two epochs, evaluated on the 75 competition tasks, all unseen: 22/75 (base 25 and 28), more capped attempts (50 vs 44 and 39), no task the base never solved; training loss fell from 0.076 to 0.038, i.e. the model memorised its own successes. Other teams report the same with 314-2,000 teacher trajectories. *Preference pairs (RPO [7]):* from the three pool rounds, chosen = fastest solved attempt and rejected = slowest unsolved one give 34 pairs, 30 of them against a capped attempt, so the signal mostly teaches finishing sooner. One epoch on the 28 cheapest pairs gives 26/75, 48 patches (base 41, 47), 45 capped (44, 39) and an unchanged median of 16 calls: a null. The implicit reward margin was positive on 19 of 28 steps, but the chosen sequences' log-probability fell by up to 226 nats, DPO's likelihood displacement (Figure 4).

![Figure 4](figures/fig3_sft_nulls.png)
*Figure 4. Tasks resolved of 75 (same harness, prompt and 5-minute cap) and capped attempts. Gray: base; blue: LoRA adapters.*

## 7. Building verified, harness-compatible tasks

From SWE-rebench [6] (6,542 filtered tasks) we selected 898 candidates (difficulty <= 2, <= 5 modified files, <= 4 per repository, no competition repositories). A CPU notebook validated each in a clean Python 3.12 environment (target tests fail with the test patch, pass with the gold patch): 318 valid + 43 nearly valid. The harness's own verification is stricter (all tests in touched files must pass; a skipped required test fails): 176 of 359 gold patches pass and 5 also pass with an empty patch (dropped), leaving **171 tasks from 142 repositories**; the unchanged 31B resolves 55, the E4B 25 (Figure 5). Pilots showed how the sandbox decides solvability: the subprocess sandbox never injects wheels, entries added through .pth files silently override system packages, and pytest auto-loads every package's pytest11 entry point, so one unrelated plugin crashed all 12 pilot tasks until the entry points were stripped.

![Figure 5](figures/fig4_task_funnel.png)
*Figure 5. From 898 SWE-rebench candidates to the 171-task pool.*

## 8. The small model: three routes to distil into E4B

E4B (8B parameters, 4-bit) is the model that fits on a laptop, and the competition rewards its best private score separately. Under the B2 budget the unmodified E4B resolves **10, 7 and 8 of 75** under three prompts, every solved task being one the 31B also solves; it is fast (114-125 s per task, at most one capped attempt) and scores 0.06 on the leaderboard. Its tokenizer and chat template reproduce real 31B prompts token for token (60/60 captured requests), so sequences and token-level scores transfer between the two models (Figure 6).

*Imitating 31B successes.* LoRA (34.9M parameters) on the 71 31B-resolved pool sequences: 9/75. Among finished attempts precision rises (29% vs 17%), but the adapter explores until the call budget is spent, then repeats `submit_patch` with an empty patch hundreds of times (the harness does not end the session), so only 30 attempts write a patch (base 68).

*Imitating a stronger teacher.* We rewrote resolved Qwen3-Coder-480B OpenHands trajectories (nebius/SWE-rebench-openhands-trajectories, CC BY 4.0; 63k tokens median, too long to imitate whole) into short episodes in our tool schema: search, read, one edit per hunk of the final patch, test, submit; 700 episodes from 597 repositories, rendered exactly as vLLM does. The adapter resolves **3/75**. The capture shows why: 522 of 568 edit calls fail because the old text is not in the file. Our converter chose each read window around the final hunk, i.e. from the answer; the student learned "read once, then edit" but at test time cannot know where to read, and writes edits for code it never saw. Hindsight pruning must only keep steps the student could have taken without the answer.

*On-policy distillation.* Following GKD [11] and MiniLLM [12], the student generates and the teacher grades: the unmodified E4B ran the 171 pool tasks (25 resolved; 244 context segments, 2.93M tokens), then the 31B scored every token of those contexts in one prefill each through vLLM's prompt log-probabilities (top 8; memory scales with chunk length x vocabulary, so 2,048-token chunks failed and 512 worked, 48 minutes). The 31B's top choice agrees with the student's token 82% of the time: a dense signal that needs no task success. The loss is the exact reverse KL between student and teacher coarse-grained to the teacher's top 8 tokens plus one "rest of vocabulary" bucket (known, because the top-8 log-probabilities are over the full vocabulary), on 712k student tokens. Rule written before the run: >= 15/75 with >= 55 patches is a candidate, <= 11 null, more than 20 capped attempts a failure.

Training ran 122 steps (2.6 h; loss 0.60 over the first 5 steps, 0.19 over the last 10) and the adapter passed the activity and smoke gates. The watchdog stopped the evaluation after 60 of 75 tasks, and the run is a **failure**: 52 attempts hit the 8-minute cap (base: at most 1), 9 patches (base 54-55), 6 resolved (base 5-8 on the same tasks). Reasoning per call grew from 1,405 to 3,289 characters, so a call took 26 s instead of 7.5 s. A pre-registered follow-up capped reasoning at 512 tokens for the adapter and, as a control, the base: time per call recovered (9.2 vs 6.4 s), the outcome did not (9 vs 11 of 75 resolved, 50 vs 1 capped, 41 vs 240 edits). Forced out of its reasoning, the adapter keeps deliberating in its visible output ("Wait, I think I misunderstood..." until the 4,096-token limit). The training data shows the source: on E4B's own contexts the 31B expects "Wait" 31 times as often as E4B writes it (63.5 vs 2.0 per 10k tokens), and gives E4B's decision to end its reasoning 0.49 probability against 0.79 for an average token. Reverse KL imported the teacher's self-revising style, and the student overshot it (57-72 "Wait" per 10k characters; the 31B writes 11) without the 31B's ability to resolve its doubts. Next steps: Appendix B.

![Figure 6](figures/fig6_small_model.png)
*Figure 6. Tasks resolved of 75 by the 31B and the E4B, with patches written and capped attempts under each bar.*

## 9. What transfers beyond this competition

Four results do not depend on Gemma or this harness. (i) *Protocol:* with outcomes this bimodal, single-run comparisons of agent variants are noise; pre-registered paired rounds and repeated identical submissions are the cheap instruments. (ii) *Hindsight leakage:* any conversion of successful trajectories into shorter training episodes must keep only steps whose inputs the student could have chosen without the answer; edit-failure rate is a quick detector. (iii) *Top-k distillation from a serving engine:* teacher top-k log-probabilities from any vLLM-compatible server give an exact coarse-grained KL with a known remainder bucket, so on-policy distillation needs no teacher logits in the training process; memory for prompt log-probabilities scales with chunk length times vocabulary. (iv) *Infrastructure:* the capture proxy works for any OpenAI-compatible agent, the chunked decoder layers make long-sequence LoRA fit on 24 GB GPUs, and the sandbox lessons apply to any pytest-based harness.

## 10. Related work

SWE-bench [1] and SWE-agent [2] define the task and interface. Training on verified trajectories is studied in SWE-Gym [3], SWE-smith [4] and R2E-Gym [5], mostly with full fine-tuning of 7-32B models and thousands of trajectories; SWE-rebench [6] is our task source, and FreshSWE-Py [13] is a community pool of 663 post-release tasks in the competition's layout. Preference optimisation with a likelihood term [7] and trial-and-error pairs [8] motivate our preference design; on-policy distillation [11, 12] our E4B method. LoRA [9] and DPO [10] are the underlying methods.

## 11. Limitations

One model family; three repositories dominate the 75 local tasks; single rounds with a paired noise of +-3 tasks; no hidden-set evidence of any adapter gain; E4B rounds of 7-10 tasks that a few flips can reorder; leaderboard numbers and community results are single runs or unverified forum reports; most local timing is concurrent while the scorer is sequential.

## 12. Conclusion and release

Under a sequential 12-hour budget, noise and task bimodality set the ceiling of evaluation, per-task budgets bind the whole run, and LoRA imitation of typical successes adds nothing while costing speed. For the small model, sequence imitation either shifts behaviour without adding skill or, if episodes leak the answer, teaches unrealisable shortcuts, and token-level distillation from a larger teacher imports its deliberation style, which the clock punishes even with reasoning capped. What transfers is the machinery and the protocol: exact capture, memory-exact training, adapter verification, a verified-task pipeline, token-level teacher scoring and pre-registered decisions. Code, task IDs (not task text), notebook generators and the full experiment log: https://github.com/JayGupta2003/gemma4-swe-agent-posttraining (Apache-2.0).

## References
[1] Jimenez et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024. arXiv:2310.06770.
[2] Yang et al. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS 2024. arXiv:2405.15793.
[3] Pan et al. Training Software Engineering Agents and Verifiers with SWE-Gym. ICML 2025. arXiv:2412.21139.
[4] Yang et al. SWE-smith: Scaling Data for Software Engineering Agents. arXiv:2504.21798.
[5] Jain et al. R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents. arXiv:2504.07164.
[6] Badertdinov et al. SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents. NeurIPS 2025 D&B. arXiv:2505.20411.
[7] Pang et al. Iterative Reasoning Preference Optimization. NeurIPS 2024. arXiv:2404.19733.
[8] Song et al. Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents. ACL 2024. arXiv:2403.02502.
[9] Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685.
[10] Rafailov et al. Direct Preference Optimization. arXiv:2305.18290.
[11] Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024. arXiv:2306.13649.
[12] Gu et al. MiniLLM: On-Policy Distillation of Large Language Models. arXiv:2306.08543.
[13] FreshSWE-Py. Community Kaggle dataset moshegolderson/freshswe-py, 2026.

## Appendix A. Every run (supplementary)

All GPU runs: Kaggle notebooks on 4x L4, offline; quota hours = 2x wall hours. Notebook generators are in the repository (`scripts/`), kernel ids are noobhunterjay/gemma-4-<id>; LoRA r=16, alpha=32 on attention and MLP projections throughout. Rules and full results: `docs/STRATEGY.md`.

| run | what | data | settings | quota h | rule (before the run) | result |
|---|---|---|---|---|---|---|
| Exp1-3 | prompt arms, 31B | 30 / 48 / 48 tasks | edit-early, no-repro, lean prompt, thinking 1,024 | - | +5 resolved and no repo worse | no arm passed; identical prompt 23 vs 18 of 48 |
| G1 | training/serving spike | 63 sequences | chunked layers, activation offload | ~7 | >= 60 tok/s, adapter active, decode >= 0.85x | passed; ceiling 10k -> 18.4k tokens |
| G2 | capture base rounds | 75 tasks x 2 | B1, 5 min | 6.2 | capture ids == /tokenize | 25, 28; ids identical |
| G3 | SFT on own successes | 70 seq, 30 tasks | lr 2e-4, 1 epoch, 35 steps | 9.4 | >= ~34/75 | 28/75; LB 0.08 |
| G5 | 12B base | 75 tasks | B1 | - | - | 17/75, 57 capped |
| G10 | pool round p1 | 171 pool tasks | B1, T 0.2 | 7.2 | - | 55/171 |
| G11 | SFT across repos | 71 seq, 54 repos | lr 2e-4, 2 epochs, 72 steps | 15.6 | >= ~34/75 or new tasks | 22/75 |
| G12 | thinking budgets | 32 tasks | 2,048/4,096 and 1,024/3,072 | 3.9 | long-turn time share halves | 12, 11 vs 10, 11; capped unchanged |
| G13 | temperature 1.0 | 75 tasks | T 1.0, top_p 0.95, top_k 64 | 3.8 | mixed tasks for pairs | 29/75; flips +50% |
| G14a/c | E4B base, 3 prompts | 75 tasks | B2 budget | 5.5 | >= 14/75 | 10, 7, 8 |
| G14b | E4B SFT on 31B successes | 71 seq | lr 2e-4, 2 epochs | 8.0 | >= 14/75, >= 60 patches, <= 5 capped | 9/75, 30 patches, 44 capped |
| G15/G16 | pool rounds p2, p3 | 171 tasks | T 1.0 | 14.5 | >= 25 flips, >= 40 pairs | 55, 52; 57 pairs |
| G17 | RPO on own pairs | 28 pairs | beta 0.1, NLL 1.0, lr 1e-4, 28 steps | 9.4 | >= 34/75 or <= 35 capped with >= 28 | 26/75, 45 capped |
| G18 | E4B SFT on rewritten Qwen3-Coder episodes | 700 episodes, 597 repos | lr 1e-4, 1 epoch, 350 steps | 9.0 | >= 16/75, >= 55 patches, <= 8 capped | 3/75; 92% edit failures |
| G19 | B2 sequential (scorer regime) | 24 tasks | concurrency 1 | 4.6 | - | 9/24; call cap ends 9 |
| G20/G20b | E4B pool rollouts + 31B scoring | 171 tasks; 244 segments | prompt_logprobs top-8, 512-token chunks | 4.8 | - | 25/171; 2.93M tokens scored |
| G21 | E4B on-policy distillation | 244 segments, 712k loss tokens | coarse top-8 reverse KL, lr 1e-4, 1 epoch, 122 steps | ~10 | >= 15/75 candidate, 12-14 weak, <= 11 null, > 20 capped failure | failure: 6/60 (watchdog at 60 tasks), 52 capped, 9 patches; KL 0.60 -> 0.19 |
| G22 | G21 adapter, reasoning capped; base control | 75 tasks x 2 arms | thinking budget 512, output cap 4096, no training | ~9 | >= 12/75, <= 8 capped and >= base@512 + 3 signal; <= 11 null; > 20 capped failure | failure: 9/75, 50 capped, 15 patches; base at the same cap 11/75, 1 capped, 65 patches |

## Appendix B. Next steps (supplementary)

Decision rules are committed to the repository before each result (`docs/STRATEGY.md`).

1. **Done: is there skill under the length? (G22.)** The unchanged G21 adapter with `thinking_budget` 512 (from 4,096) and `max_output_tokens` 4,096 (from 8,192), and the base E4B at the same cap, on the same 75 tasks in the same order. Rule: signal if >= 12/75 with <= 8 capped and >= 3 above the base at the cap; failure if > 20 capped even at the cap. Result: **failure** (9/75, 50 capped; base 11/75, 1 capped). Time per call recovered (9.2 vs 6.4 s), but the adapter made 2,996 main-agent calls against 1,074, of which 41 edits against 240, 98 calls ran into the output limit, and 9 attempts ended without submitting.
2. **Dropped: mask only the reasoning-close token.** Planned as step 2, but G22 shows the shift is not confined to when reasoning stops: the student also deliberates in its visible output. Tool choice is not where the models disagree either: when E4B chose `edit_file`, the 31B gave that choice 0.96 probability on average (`run_command` 0.99, `read_file` 0.72).
3. **Next: separate what to do from how to think.** 79% of the loss tokens are reasoning, where the 31B gives the student's token 0.76 probability on average against 0.94 on action tokens (tool names, arguments, edit text). Candidate objective: the teacher's KL on action tokens only, plus a KL anchor to the frozen base E4B on reasoning tokens so its own deliberation style is kept (as reference-model penalties do in RL fine-tuning). Rule to fix before running: >= 15/75 with >= 55 patches and <= 8 capped is a candidate; "Wait" per 10k characters must stay within 2x of the base. About 10 quota h. Alternative: a teacher closer in style (the 12B), or penalising self-revision markers directly.
4. **Then: confirm and iterate.** A candidate gets a second evaluation round (paired noise is +-3 tasks) before an E4B leaderboard submission, then on-policy iteration as in GKD [11]: rollouts from the improved adapter, re-scored by the 31B, retrained (~15 quota h per iteration).

Not planned: forward KL or more epochs (both pull the teacher's style harder), and reinforcement learning with verifiable rewards (one rollout pass over the 171-task pool takes ~3 h; RL needs dozens within a 30-hour weekly quota). The base E4B at the 512-token cap (11/75, 1 capped, 126 s per task) is within noise of the uncapped base (7-10) and is not evidence for changing the E4B submission.
