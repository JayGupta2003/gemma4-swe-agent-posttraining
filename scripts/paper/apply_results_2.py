"""One-off edit of docs/paper/DRAFT.md: fill the E4B section, the temperature/mixedness result, the preference gate, and fix the licence (MIT). Every replacement is asserted."""
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "docs" / "paper" / "DRAFT.md"
s = p.read_text(encoding="utf-8")


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


sub("> DRAFT 1 (2026-10-24). Items in [BRACKETS] are pending experiments (E4B distillation, temperature probe, preference gate, leaderboard repeats) and must be filled or deleted before submission.",
    "> DRAFT 2 (2026-11-02). E4B, temperature and preference-gate results are in; items in [BRACKETS] (the RPO result of run G17) are pending and must be filled or deleted before submission.")

sub("and document the sandbox conventions that silently decide whether a task is solvable. [E4B: ...]",
    "and document the sandbox conventions that silently decide whether a task is solvable. (5) The 4B-class model (E4B) solves 7-10 of 75 tasks, all of them tasks the 31B also solves, whatever the prompt (leaderboard 0.06); distilling 71 31B trajectories into it raises precision (29% vs 17% of finished attempts solved) but makes it explore until the budget ends without editing, so the total does not move. Temperature 1.0 raises round-to-round disagreement by ~50% but not the solve count, and 53 of the 57 resulting preference pairs only teach finishing sooner.")

sub("We computed that 3 additional rounds over the 171-task pool would yield only ~20 matched success/failure pairs at temperature 0.2. [TEMPERATURE PROBE: temperature 1.0 changes the mixed fraction from 17% to X%.]",
    "We predicted that 3 additional rounds over the 171-task pool would yield only ~20 matched success/failure pairs at temperature 0.2. *Temperature 1.0* (top_p 0.95, top_k 64) resolves 29/75 on the competition tasks (base 25 and 28) and, as a fifth attempt, lifts the mixed count from 13 to 18; tasks solved by exactly one of two rounds are 5-8 among the four temperature-0.2 rounds (two base, two adapter) and 9-13 for 1.0 vs 0.2. On the 171-task pool, rounds at 0.2, 1.0 and 1.0 resolve 55, 55 and 52 with 86, 93 and 90 capped; 100 tasks are never solved, 14 solved once, 23 twice, 34 always (37 mixed), and 24 of the 26 tasks that differ between the first two rounds are *solved vs ran out of clock*, not *solved vs wrong patch*.")

sub("We pre-registered the decision rule (>=34/75 or fewer capped without losses) before each evaluation. [PREFERENCE: gate result and, if run, RPO result.]",
    "We pre-registered the decision rule (>=34/75 or fewer capped without losses) before each evaluation. *Preference pairs:* from the three pool rounds the pair builder (chosen = fastest solved attempt, rejected = slowest unsolved one, final context segment, token-exact) yields 34 solved-vs-unsolved pairs (30 vs capped, 4 vs wrong patch) plus 23 faster-vs-slower solved pairs; at 24k tokens per pair one epoch of all 57 would take 14 h of wall time, more than a session allows, so the run uses the 28 cheapest pairs for one epoch. [RPO RESULT G17: resolved X/75, patches written Y, capped Z against base 25/28, 41/47, 44/39.]")

sub("## 8. [Small model] E4B distillation\n\n[PENDING: base E4B on 75 tasks; LoRA on the 31B's verified sequences (same tokenizer and chat template, so token-exact sequences transfer); results on held-out tasks; leaderboard.]",
    """## 8. A small model: E4B and distillation from the 31B

The competition's small-model bonus rewards the best private score with the 8B-parameter E4B (4-bit, LoRA allowed). With the same harness and an 8-minute / 28-call budget, the unmodified E4B resolves **10, 7 and 8 of 75** tasks under three prompts (the 5-minute prompt, the budget-aware prompt, and a reproduce-then-verify prompt); the three rounds differ on 5-8 tasks, their union is 14 tasks, and **every task any of them solves is a task the 31B solves at least once** (0 of the 43 the 31B never solves). It is fast (114-125 s per task, at most one capped attempt, median 12-13 calls, 68-69 patches written per round), so time is not its constraint; the verify prompt did not make it do more work (14.0 vs 13.1 calls). The unmodified E4B scores **0.06** on the public leaderboard.

We then trained a LoRA (34.9 M parameters, 3.7x faster per token than the 31B) on the 71 31B-resolved pool sequences of Section 6 (the tokenizer and chat template are shared, so token-exact sequences transfer) and verified that vLLM applies E4B adapters. On the 75 competition tasks it resolves 9/75, but writes a patch in only 30 attempts (base 68) and hits the cap in 44 (base 1). The capture shows why: the adapted model explores until the 28-call budget is spent without editing, then calls `submit_patch` with an empty patch, which the harness accepts without ending the session, and repeats the call hundreds of times (up to 440 model calls per attempt; 393 s per task vs 114 s, which would exceed the 12-hour limit on ~120 tasks). Among the 31 attempts that did finish, 9 resolve (29%) against 5-6 for the base on the same tasks (17%); it solves four tasks no base round ever solves and loses five of the 14 tasks the base ever solves. Precision rises, recall falls and the total does not move: as for the 31B, SFT shifts the behaviour distribution rather than adding capability, and any adapter needs a cap on model calls.""")

sub("Code, scripts, task IDs (not task text) and the analysis notebooks are released under Apache-2.0.",
    "Code, scripts, task IDs (not task text) and the analysis notebooks are released under the MIT licence.")
sub("no hidden-set evidence of any adapter gain,", "no hidden-set evidence of any adapter gain, E4B rounds of 7-10 tasks that a few flips can reorder,")
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
