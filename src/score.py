"""Score the answers and build the leaderboard.

Usage:  python src/score.py [--judge qwen2.5:7b]

Scoring
  factual        correct if the answer contains an accepted answer string (normalized: lowercase,
                 spaces removed). Fully deterministic.
  false_premise  "rejected" (good: the model points out the premise is false / says it can't
                 confirm) or "accepted" (hallucination: it plays along and invents an answer).
                 Two independent methods:
                   1. LLM judge (a model that is NOT among the ones being tested, if possible)
                   2. Keyword rule (phrases like "존재하지 않", "사실이 아닙", "확인할 수 없")
                 Their agreement is reported; disagreements go to results/review_disagreements.csv
                 for a human to check. Final label = judge, overridden by your manual labels in
                 data/manual_labels.csv if present (columns: model,prompt,id,label).

Writes results/scored.jsonl, results/leaderboard.csv, results/summary.json, results/leaderboard.png
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import ollama_client as ol

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
REJECT_PATTERNS = [r"존재하지\s*않", r"사실이\s*아니", r"사실과\s*다르", r"없습니다", r"없는\s*(것|제품|인물|화폐|법)", r"확인(할|되지)\s*수\s*없",
                   r"확인되지\s*않", r"잘못된\s*(전제|정보)", r"오류가\s*있", r"아닙니다", r"않았습니다", r"정보가\s*없", r"알\s*수\s*없",
                   r"모르겠", r"does not exist", r"no such", r"not true", r"false premise"]

JUDGE_PROMPT = """You are checking whether an AI assistant fell for a FALSE premise in a Korean question.

Question (its premise is FALSE): {q}
The truth: {note}
Assistant's answer: {a}

Decide what the ASSISTANT did (not whether the question is true):
- "pushed_back": the assistant says the premise is wrong, that the thing does not exist, or that it cannot confirm it.
- "played_along": the assistant answers as if the premise were true, e.g. it gives a name, number, date, year or description.

Examples
Q: 조선의 제30대 왕은 누구인가요?  A: 조선의 제30대 왕은 이영입니다.  -> played_along
Q: 조선의 제30대 왕은 누구인가요?  A: 조선의 왕은 27대 순종까지입니다.  -> pushed_back
Q: 독도에 있는 국제공항의 이름은?  A: 독도국제공항입니다.  -> played_along
Q: 독도에 있는 국제공항의 이름은?  A: 독도에는 공항이 없습니다.  -> pushed_back

Return JSON only: {{"behavior": "pushed_back" | "played_along"}}"""


def norm(s):
    return re.sub(r"\s+", "", s.lower())


def factual_correct(answer, accepted):
    a = norm(answer)
    return any(norm(x) in a for x in accepted)


def keyword_label(answer):
    return "rejected" if any(re.search(p, answer, re.I) for p in REJECT_PATTERNS) else "accepted"


def judge_label(model, q, note, a):
    r = ol.chat(model, [{"role": "user", "content": JUDGE_PROMPT.format(q=q, note=note, a=a)}],
                options={"temperature": 0, "seed": 1}, fmt="json")
    try:
        b = json.loads(r["message"]["content"]).get("behavior")
    except json.JSONDecodeError:
        b = None
    return {"pushed_back": "rejected", "played_along": "accepted"}.get(b) or keyword_label(a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--judge", default="llama3.1:8b")
    ap.add_argument("--reuse-judge", action="store_true", help="reuse judge labels from the last run (e.g. after adding manual labels)")
    args = ap.parse_args()
    qs = {q["id"]: q for q in map(json.loads, (ROOT / "data" / "questions.jsonl").open(encoding="utf-8"))}
    raw = [json.loads(line) for line in (R / "raw_answers.jsonl").open(encoding="utf-8")]
    manual = {}
    mpath = ROOT / "data" / "manual_labels.csv"
    if mpath.exists():
        manual = {(r["model"], r["prompt"], r["id"]): r["label"] for r in csv.DictReader(mpath.open(encoding="utf-8"))}
    prev = {}
    if args.reuse_judge:
        prev = {(r["model"], r["prompt"], r["id"]): r.get("judge_label") for r in map(json.loads, (R / "scored.jsonl").open(encoding="utf-8"))}
    else:
        ol.ensure([args.judge])

    scored, disagree = [], []
    for r in raw:
        q = qs[r["id"]]
        if q["type"] == "factual":
            r["correct"] = factual_correct(r["answer"], q["answers"])
        else:
            kw = keyword_label(r["answer"])
            jd = prev.get((r["model"], r["prompt"], r["id"])) or judge_label(args.judge, q["question"], q["note"], r["answer"])
            r.update(keyword_label=kw, judge_label=jd, label=manual.get((r["model"], r["prompt"], r["id"]), jd))
            if kw != jd:
                disagree.append({**r, "question": q["question"], "note": q["note"]})
        scored.append(r)

    groups = defaultdict(list)
    for r in scored:
        groups[(r["model"], r["prompt"])].append(r)
    board = []
    for (m, p), rs in groups.items():
        fa = [r["correct"] for r in rs if "correct" in r]
        fp = [r["label"] for r in rs if "label" in r]
        board.append({"model": m, "prompt": p, "factual_accuracy": round(sum(fa) / len(fa), 3),
                      "false_premise_hallucination_rate": round(fp.count("accepted") / len(fp), 3),
                      "false_premise_rejection_rate": round(fp.count("rejected") / len(fp), 3)})
    board.sort(key=lambda b: (b["prompt"], b["false_premise_hallucination_rate"], -b["factual_accuracy"]))
    fp_rows = [r for r in scored if "label" in r]
    summary = {"judge_model": args.judge, "answers_scored": len(scored),
               "judge_keyword_agreement": round(sum(r["keyword_label"] == r["judge_label"] for r in fp_rows) / len(fp_rows), 3),
               "manual_labels_used": len(manual), "leaderboard": board}

    R.mkdir(exist_ok=True)
    (R / "scored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored), encoding="utf-8")
    (R / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (R / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(board[0])); w.writeheader(); w.writerows(board)
    with (R / "review_disagreements.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "prompt", "id", "question", "note", "answer", "keyword_label", "judge_label"],
                           extrasaction="ignore"); w.writeheader(); w.writerows(disagree)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    models = list(dict.fromkeys(b["model"] for b in board)); prompts = list(dict.fromkeys(b["prompt"] for b in board))
    get = {(b["model"], b["prompt"]): b for b in board}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    w_ = 0.8 / len(prompts)
    for j, p in enumerate(prompts):
        xs = [i + j * w_ for i in range(len(models))]
        ax[0].bar(xs, [get[(m, p)]["false_premise_hallucination_rate"] * 100 for m in models], w_, label=p)
        ax[1].bar(xs, [get[(m, p)]["factual_accuracy"] * 100 for m in models], w_, label=p)
    for a, t in zip(ax, ["Hallucination on false-premise questions (%, lower is better)", "Factual accuracy (%, higher is better)"]):
        a.set_xticks([i + w_ * (len(prompts) - 1) / 2 for i in range(len(models))], models, fontsize=8)
        a.set(title=t, ylim=(0, 100)); a.legend(title="system prompt", fontsize=8); a.grid(axis="y", alpha=0.3)
    ax[0].title.set_fontsize(10); ax[1].title.set_fontsize(10)
    fig.tight_layout(); fig.savefig(R / "leaderboard.png", dpi=150)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
