# Gemma 4 Developer Agent Competition

Kaggle: https://www.kaggle.com/competitions/gemma-4-developer-agent

Full strategy: [`docs/STRATEGY.md`](docs/STRATEGY.md)

## Environment

Local work uses the `mlai` conda env (never `base`):

```
source /j/Anaconda/etc/profile.d/conda.sh && conda activate mlai
```

Kaggle CLI is installed there; credentials at `~/.kaggle/kaggle.json`.

## Layout

- `data/` — competition dataset (gitignored, downloaded via Kaggle API)
- `docs/` — strategy doc + paper track drafts (`docs/paper/`)
- `notebooks/` — Kaggle training/eval notebooks
- `src/` — reusable Python (agent scaffold, data pipeline, reward functions)
- `scripts/` — one-off CLI scripts (data mining, eval runs)
- `adapters/` — trained LoRA adapters (gitignored, weights only)
- `submission_drafts/` — built `submission.zip` candidates (gitignored)
- `logs/` — training/eval run logs (gitignored)

## Related

- `J:\KAAM\Kaggle Competition\Google Tunix` — prior hackathon with a working GRPO+LoRA+Tunix pipeline on Gemma 2B, being adapted here for Gemma 4 + code-agent tasks.
