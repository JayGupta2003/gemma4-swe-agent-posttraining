## Table 4. Small model (E4B) and E4B + LoRA on the 75 competition tasks (E4B rows: 8 min / 28 calls; 31B rows: 5-minute cap)

| round | resolved | patches written | capped | mean task s | median model calls |
|---|---|---|---|---|---|
| 31B base round 1 | 25 | 41 | 44 | 274 | 16 |
| 31B base round 2 | 28 | 47 | 39 | 269 | 16 |
| E4B base 5-min prompt | 10 | 68 | 1 | 114 | 13 |
| E4B base B2 prompt | 7 | 69 | 1 | 125 | 12 |
| E4B base verify prompt | 8 | 68 | 0 | 117 | 12 |
| E4B + LoRA (31B traces) | 9 | 30 | 44 | 393 | 33 |
| E4B + LoRA (Qwen3-Coder episodes) | 3 | 45 | 10 | 137 | 9 |
| E4B + on-policy distillation (60 of 75 run) | 6 | 9 | 52 | 478 | 15 |
| E4B + on-policy distillation, reasoning capped | 9 | 15 | 50 | 450 | 37 |
| E4B base, reasoning capped | 11 | 65 | 1 | 126 | 13 |

E4B base rounds: union 14 tasks; solved in at least 2 of the 3 rounds: 6; all 14 are also solved by the 31B in some round: True.

## Table 5. Tasks solved in exactly one of two rounds (31B, 75 competition tasks)

| | base r1 | base r2 | G3 adapter | G11 adapter | base T 1.0 |
|---|---|---|---|---|---|
| base r1 | 0 | 7 | 5 | 7 | 10 |
| base r2 | 7 | 0 | 6 | 8 | 9 |
| G3 adapter | 5 | 6 | 0 | 8 | 9 |
| G11 adapter | 7 | 8 | 8 | 0 | 13 |
| base T 1.0 | 10 | 9 | 9 | 13 | 0 |

## Table 6. Preference pairs from the three pool rounds (31B, 171 tasks)

| item | count |
|---|---|
| attempts (solved / finished wrong / capped) | 513 (162 / 82 / 269) |
| final segments unusable (malformed history or over 16,384 tokens) | 63 |
| solved-vs-unsolved pairs | 34 (30 vs capped, 4 vs wrong patch) |
| faster-vs-slower solved pairs | 23 |
| tasks with a pair | 51 |
| tokens, all pairs (M) | 1.35 |
