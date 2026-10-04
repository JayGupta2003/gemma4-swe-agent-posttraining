"""Build the G1 spike training set from ATIF traces of RESOLVED attempts (Exp1 + Exp2 outputs).

For G1 the content only has to be realistic (throughput/memory/serving are what is measured); G2 will replace this with the exact
vLLM request/response capture. Output: kaggle_assets/g1_data_dataset/g1_train.jsonl, one JSON per line:
    {"instance_id", "arm", "messages": [system, user, assistant(tool_calls), tool, ...]}   (OpenAI-style, dict arguments)

Usage: python scripts/g1/make_g1_data.py <exp1_out_dir> <exp2_out_dir>
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kaggle_assets" / "g1_data_dataset"
MAX_PER_TASK = 2


def trace_to_messages(trace: dict) -> list[dict] | None:
    msgs: list[dict] = []
    for s in trace["steps"]:
        src = s.get("source")
        if src == "system":
            msgs.append({"role": "system", "content": s["message"]})
        elif src == "user":
            msgs.append({"role": "user", "content": s["message"]})
        elif src == "agent":
            tcs = s.get("tool_calls") or []
            if not tcs:
                msgs.append({"role": "assistant", "content": s.get("message") or ""})
                continue
            msgs.append({"role": "assistant", "content": s.get("message") or "",
                         "tool_calls": [{"type": "function", "function": {"name": tc["function_name"], "arguments": tc.get("arguments", {})}}
                                        for tc in tcs]})
            obs = s.get("observation") or {}
            results = obs.get("results")
            if results:
                for tc, r in zip(tcs, results):
                    msgs.append({"role": "tool", "name": tc["function_name"], "content": r.get("content", "") if isinstance(r, dict) else str(r)})
            elif "content" in obs:
                msgs.append({"role": "tool", "name": tcs[0]["function_name"], "content": obs["content"]})
            else:
                return None  # tool call without an observation: not a clean example
    merged: list[dict] = []
    for m in msgs:  # the chat template merges adjacent same-role messages, so merge them here to keep rendering prefix-stable
        if merged and m["role"] == "assistant" and merged[-1]["role"] == "assistant" and not m.get("tool_calls") and not merged[-1].get("tool_calls"):
            merged[-1] = {"role": "assistant", "content": (merged[-1].get("content") or "") + "\n\n" + (m.get("content") or "")}
        else:
            merged.append(m)
    msgs = merged
    if not msgs or msgs[0]["role"] != "system" or sum(m["role"] == "assistant" for m in msgs) < 3:
        return None
    return msgs


def main(exp_dirs: list[Path]) -> None:
    rows = []
    for exp in exp_dirs:
        for arm_dir in sorted(p for p in exp.iterdir() if (p / "task_results.jsonl").exists()):
            for line in (arm_dir / "task_results.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if not r["resolved"]:
                    continue
                tp = arm_dir / "traces" / f"trace_{r['instance_id']}.json"
                if not tp.exists():
                    continue
                msgs = trace_to_messages(json.loads(tp.read_text(encoding="utf-8")))
                if msgs:
                    rows.append({"instance_id": r["instance_id"], "arm": arm_dir.name, "messages": msgs,
                                 "n_chars": sum(len(m.get("content") or "") for m in msgs)})
    # at most MAX_PER_TASK per task, prefer mid-length trajectories (not the shortest trivial ones, not the huge ones)
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["instance_id"], []).append(r)
    chosen = []
    for iid, rs in sorted(by_task.items()):
        rs.sort(key=lambda r: abs(r["n_chars"] - 40000))
        chosen += rs[:MAX_PER_TASK]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "g1_train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in chosen:
            f.write(json.dumps({k: r[k] for k in ("instance_id", "arm", "messages")}, ensure_ascii=False) + "\n")
    (OUT / "dataset-metadata.json").write_text(json.dumps({"title": "gemma4-g1-data", "id": "noobhunterjay/gemma4-g1-data",
                                                           "licenses": [{"name": "CC0-1.0"}]}, indent=2), encoding="utf-8")
    print(f"{len(rows)} resolved attempts with clean traces -> {len(chosen)} examples over {len(by_task)} tasks; wrote {OUT}")
    import statistics
    print("chars per example: median", int(statistics.median(r["n_chars"] for r in chosen)), "max", max(r["n_chars"] for r in chosen))


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
