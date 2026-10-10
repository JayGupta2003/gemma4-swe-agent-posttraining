"""Show the submission allowance and the latest submissions with scores and the raw error text (the CLI hides _error_description)."""
from kaggle.api.kaggle_api_extended import KaggleApi

C = "gemma-4-developer-agent"
a = KaggleApi()
a.authenticate()
print(a.competition_get_submission_limits(C))
for s in a.competition_submissions(C)[:6]:
    print(s.ref, s.date, s.status, "score=", s.public_score, "| err=", getattr(s, "_error_description", None))
