"""Word count of the paper draft (Kaggle Writeup limit: 3,000 words). Counts everything below the banner line, markdown syntax stripped, bracketed TODO items included.
Usage: python scripts/paper/count_words.py [docs/paper/DRAFT.md]"""
import re, sys
from pathlib import Path
p = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/paper/DRAFT.md")
t = p.read_text(encoding="utf-8")
body = "\n".join(l for l in t.splitlines() if not l.startswith("> DRAFT"))
body = re.sub(r"[#*_`>|]", " ", body)
words = re.findall(r"\S+", body)
secs = re.split(r"\n(?=## )", t)
print(f"total words: {len(words)} (limit 3000)")
for s in secs:
    head = s.splitlines()[0][:60]
    n = len(re.findall(r"\S+", re.sub(r"[#*_`>|]", " ", s)))
    print(f"  {n:5d}  {head}")
