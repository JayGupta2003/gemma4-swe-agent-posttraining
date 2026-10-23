## Table 1. Public leaderboard (58 tasks; one task = 0.017)

| submission | budget | public score |
|---|---|---|
| baseline (starter prompt) | 5 min cap | 0.12 |
| B1 | 5 min cap | 0.13 and 0.15 |
| G3 adapter on B1 | 5 min cap | 0.08 |
| B2 | 8 min / 28 tool calls | 0.17 |
| B3 (B2 + thinking 2048) | 8 min / 28 tool calls | pending |

## Table 2. Local rounds on the 75 competition tasks (same harness, prompt, 5-minute cap)

| round | resolved | patches written | capped | LLM calls / task |
|---|---|---|---|---|
| base r1 | 25 | 41 | 44 | 18.0 |
| base r2 | 28 | 47 | 39 | 17.9 |
| G3 adapter | 28 | 39 | 44 | 17.8 |
| G11 adapter | 22 | 32 | 50 | 17.5 |
| 12B base | 17 | 37 | 57 | 23.1 |

## Table 3. 31B vs 12B (4x L4; speeds measured single-stream in G5; the tokens-per-turn ratio is from the G5 profile)

| | 31B | 12B |
|---|---|---|
| decode speed (tokens/s) | 37.5 | 79.2 |
| completion tokens per call | 1x | 1.6-1.8x |
| capped attempts (of 75) | 44 / 39 | 57 |
| resolved (of 75) | 25 / 28 | 17 |
