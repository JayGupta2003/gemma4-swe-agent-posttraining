"""Word count of the paper draft (Kaggle Writeup limit: 3,000 words). Per the host (paper-track forum thread 749546), references do not count and an
appendix is supplementary, so the counted body stops at "## References"; the editorial banner (lines starting with "> DRAFT") is excluded. Markdown syntax
is stripped; bracketed placeholders count. Totals for references and the appendix are printed for information.
Usage: python scripts/paper/count_words.py [docs/paper/DRAFT.md]"""
import re, sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/paper/DRAFT.md")
t = "\n".join(l for l in p.read_text(encoding="utf-8").splitlines() if not l.startswith("> DRAFT"))
body, _, tail = t.partition("\n## References")
refs, _, appendix = tail.partition("\n## Appendix")


def n(s):
    return len(re.findall(r"\S+", re.sub(r"[#*_`>|]", " ", s)))


print(f"counted words (body): {n(body)} (limit 3000) | references (not counted): {n(refs)} | appendix (supplementary): {n(appendix)}")
for s in re.split(r"\n(?=## )", body):
    print(f"  {n(s):5d}  {s.strip().splitlines()[0][:60]}")
