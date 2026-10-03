"""Ask every model every question, under two system prompts.

Usage:  python src/run.py                                  # default small models (8 GB Mac)
        python src/run.py --models qwen2.5:7b exaone3.5:7.8b llama3.1:8b gemma2:9b   # 16 GB+

Prompt variants
  plain     "Answer in Korean, concisely."   (how most people use a chatbot)
  cautious  adds "If the question contains a false premise or you are not sure, say so."
This measures how much a single instruction reduces hallucination.

Writes results/raw_answers.jsonl (one line per model x prompt x question).
"""
import argparse
import json
import time
from pathlib import Path

import ollama_client as ol

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "exaone3.5:2.4b"]  # exaone = LG AI Research's Korean model
PROMPTS = {
    "plain": "질문에 한국어로 간결하게 답하세요.",
    "cautious": "질문에 한국어로 간결하게 답하세요. 질문에 사실이 아닌 전제가 포함되어 있거나 확실하지 않으면, 지어내지 말고 그렇다고 분명히 말하세요.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    args = ap.parse_args()
    qs = [json.loads(line) for line in (ROOT / "data" / "questions.jsonl").open(encoding="utf-8")]
    out = ROOT / "results" / "raw_answers.jsonl"; out.parent.mkdir(exist_ok=True)
    ol.ensure(args.models)
    with out.open("w", encoding="utf-8") as f:
        for m in args.models:
            for p in args.prompts:
                t0 = time.time()
                for q in qs:
                    r = ol.chat(m, [{"role": "system", "content": PROMPTS[p]}, {"role": "user", "content": q["question"]}],
                                options={"temperature": 0, "seed": 1, "num_predict": 256})
                    f.write(json.dumps({"model": m, "prompt": p, "id": q["id"], "answer": r["message"]["content"]},
                                       ensure_ascii=False) + "\n")
                print(f"{m:18s} {p:8s} {len(qs)} questions in {time.time() - t0:.0f}s", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
