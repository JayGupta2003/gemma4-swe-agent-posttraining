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

**Compute.** Every GPU run is a Kaggle notebook on 4x NVIDIA L4 (24 GB each), offline. L4s are only granted to notebooks that list the competition as a data source (`competition_sources` in each `kernel-metadata.json`); otherwise Kaggle silently assigns a T4x2 and vLLM fails on bf16. Kaggle bills L4x4 at 2x, and the weekly quota is 30 h, so one 31B training run plus evaluation (~5 h wall) uses about a third of a week.

**Environment.** The notebooks install the competition harness from the dataset `metric/gemma-4-developer-agent-wheelhouse` (swegemma 0.2.11, adk_submission 0.2.13, vLLM 0.19.1; Kaggle image pinned in each `kernel-metadata.json`). Models: `google/gemma-4` variants `gemma-4-31b-it-qat-w4a16-ct` and `gemma-4-e4b-it-qat-w4a16-ct` (served, scored) and their `*-qat-q4_0-unquantized` bf16 siblings (trained). Local tooling: Python 3.11+, `kaggle` CLI 2.2+; local trainer tests need transformers 5.13 + peft 0.19 (a venv that mirrors Kaggle's versions).

**Running an experiment.**
1. Look up the experiment in `docs/STRATEGY.md` (or the paper's appendix table): it names the generator script, the Kaggle kernel id, the input datasets, the decision rule and the result.
2. Build its inputs. Example (on-policy distillation): `python scripts/g20/build_opd_dataset.py <capture.jsonl.gz> <teacher_scores.npz> <teacher_meta.json> <out_dir>` from the outputs of the G20 and G20b kernels, then upload `<out_dir>` as a Kaggle dataset.
3. Generate and push the notebook: `python scripts/g21/make_g21_notebook.py && kaggle kernels push -p kaggle_assets/g21`. Each notebook checks it got 4x L4, gates on a 2-4 task smoke round before the long round, and writes a `*_summary.json` with the numbers the paper reports.
4. Fetch results: `kaggle kernels output <kernel id> -p <dir> --file-pattern "summary|task_results"` (the output listing pages 20 files, so use a pattern).

**Submissions.** Build `submission.zip` only with `python scripts/build_submission_zip.py <out.zip> <submission_dir>` (POSIX entry names), check it on Linux with `scripts/make_zip_check_notebook.py`, then submit with `scripts/submit_pinned_zip.py <submission_dir> <sha256 prefix> "<message>"`.

**Tests (no GPU).** `python scripts/g20/test_opd_score.py`, `python scripts/g20/test_train_opd.py` (tiny random Gemma 4; checks that a teacher equal to the student gives loss 0), `python scripts/g15/test_build_pairs.py`, `python scripts/g15/test_train_rpo.py`, `python scripts/g18/test_teacher_convert.py`, `python scripts/g2/test_capture_proxy.py`.

**Commit hashes.** `docs/STRATEGY.md` cites hashes of the original local history; `docs/commit_map.tsv` maps them to the published ones (the history was rewritten once before publication, see its header). Commit dates are unchanged and show that each decision rule was recorded before its result.

## Licence

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The agent prompts in `submission*/prompts/` are adapted from the competition's sample submission.
