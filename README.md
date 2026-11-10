# Post-training a 4-bit Gemma 4 SWE agent on four 24 GB GPUs

Code for our entry to the Kaggle [Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent) and its paper track
("Where the Gains Aren't: Noise, Bimodality and the Limits of LoRA Post-Training for a 4-bit Gemma 4 SWE Agent"). The paper draft is in
[`docs/paper/DRAFT.md`](docs/paper/DRAFT.md); every experiment, with its pre-registered decision rule and result, is logged in [`docs/STRATEGY.md`](docs/STRATEGY.md).

Everything runs on Kaggle notebooks (4x NVIDIA L4, offline) with the competition's own harness (`swegemma`, `adk_submission`, vLLM) from the official wheelhouse.

## What is here

| Path | What it does |
|---|---|
| `submission*/` | The agent bundles we evaluated and submitted (prompts, sampling, budget). `submission_b2/` (8 min / 28 tool calls) is the main 31B recipe; `submission_e4b/` the E4B one. |
| `scripts/build_submission_zip.py` | Builds `submission.zip` with POSIX entry names and verifies the raw bytes (a Windows-built zip with backslash names crashes the Linux scorer). |
| `scripts/g2/capture_proxy.py` | Logging reverse proxy between the harness and vLLM: exact requests, responses, prompt and completion token ids. |
| `scripts/g1/gemma4_memfix.py` | Sequence-chunked Gemma 4 decoder layers (exact) that lift the LoRA training length on 4x L4 from ~10k to ~18k tokens for the 31B. |
| `scripts/g3/train_g3.py` | LoRA SFT trainer on token-exact captured sequences (31B and E4B), chunked lm-head loss. |
| `scripts/g1/make_adapters.py`, `scripts/g3/g3_wiring.py` | Export to the scorer's adapter format and check that the harness really routes requests to the adapter. |
| `scripts/g7`, `scripts/g9` | Building verified, harness-compatible tasks from SWE-rebench (candidate selection, validation at scale, offline wheels, gold-patch checks in the harness): 171-task rollout pool. |
| `scripts/g15/` | Preference-pair builder and RPO (DPO + NLL) trainer. |
| `scripts/g18/` | Converter from public OpenHands trajectories to short episodes in our tool schema, and a token-exact chat-template renderer. |
| `scripts/g20/`, `scripts/g21/` | On-policy distillation 31B -> E4B: teacher scoring with vLLM `prompt_logprobs`, coarse top-k KL loss, trainer, tests. |
| `scripts/make_*_notebook.py`, `scripts/g*/make_*_notebook.py`, `kaggle_assets/` | Generators for every Kaggle notebook we ran, and the generated notebooks. |
| `scripts/paper/` | Figures, tables and word count for the paper. |

Each experiment G1-G21 has its own `scripts/gN/` folder; `docs/STRATEGY.md` says what it tested, how, and what came out.

## Data

Competition data is **not** included (competition rules): download it with the Kaggle API into `data/`. We publish task IDs and per-task outcomes only
(`docs/kaggle_env_gold_results.json`, `docs/exp2_task_sets.json`, `docs/g6_screen_tasks.json`, `docs/g9_rollout_pool.json`).
The rollout pool is built from [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) and the teacher episodes from
[nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) (both CC BY 4.0).
Captures and trained adapters are not included (they contain competition task text).

## Reproducing

1. Join the competition, attach the wheelhouse dataset `metric/gemma-4-developer-agent-wheelhouse` and the Gemma 4 models, and use an L4x4 session attached to the competition (L4s are only granted to notebooks with the competition as a source).
2. Generate a notebook, e.g. `python scripts/g21/make_g21_notebook.py`, upload its dataset, then `kaggle kernels push -p kaggle_assets/g21`.
3. Local tests (no GPU needed for most): `python scripts/g20/test_opd_score.py`, `python scripts/g20/test_train_opd.py` (needs transformers 5.x + peft), `python scripts/g15/test_build_pairs.py`.

## Licence

MIT, see [LICENSE](LICENSE).
