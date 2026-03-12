#!/usr/bin/env python3
"""
evaluate_gemini.py — GPTea LLM-as-Judge using Google Gemini (Free Tier)

Usage:
    python evaluate_gemini.py
    python evaluate_gemini.py --kb data/domain.csv
    python evaluate_gemini.py --save-report results.json

Requirements:
    pip install google-genai sentence-transformers scikit-learn rapidfuzz symspellpy numpy pandas python-dotenv

Your .env must contain:
    GOOGLE_API_KEY=your_key_here
"""

import argparse, time, json, os
import numpy as np
from sklearn.preprocessing import normalize as sk_normalize

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    raise ImportError("Run: pip install google-genai")

KB_PATH = "data/domain.csv"
DEFAULT_JUDGE_MODEL = "gemini-2.0-flash"

UNSEEN_TESTS = [
    ("What pH level is best for growing tea?",               "Soil & pH"),
    ("My soil is too alkaline, what should I do?",           "Soil & pH"),
    ("How acidic should the soil be for tea?",               "Soil & pH"),
    ("What happens if soil pH is too high for tea?",         "Soil & pH"),
    ("How do I improve the health of my soil?",              "Soil & pH"),
    ("What nutrients does a tea plant need?",                "Fertilizer"),
    ("My plants are not growing fast, what fertilizer?",     "Fertilizer"),
    ("Can I use natural compost on tea?",                    "Fertilizer"),
    ("How many times a year should I fertilize?",            "Fertilizer"),
    ("What does potassium do for tea plants?",               "Fertilizer"),
    ("There are tiny bugs under my leaves, what is it?",     "Pests"),
    ("My young shoots are turning brown and dying",          "Pests"),
    ("White spots appearing on my tea leaves",               "Pests"),
    ("How do I get rid of insects without chemicals?",       "Pests"),
    ("Leaves are falling off my tea bushes",                 "Pests"),
    ("How much water do tea plants need in dry weather?",    "Irrigation"),
    ("Is drip watering good for tea farms?",                 "Irrigation"),
    ("How do I stop my field from flooding?",                "Irrigation"),
    ("Does the pH of my water matter for tea?",              "Irrigation"),
    ("How do I know when to water my tea plants?",           "Irrigation"),
    ("Why am I getting less tea than last year?",            "Yield"),
    ("How can I get more leaves per bush?",                  "Yield"),
    ("Does the age of the bush affect how much I harvest?",  "Yield"),
    ("Will cutting my bushes reduce my harvest?",            "Yield"),
    ("How much tea can I expect per acre?",                  "Yield"),
    ("Is hot weather bad for my tea plants?",                "Climate"),
    ("How does rain affect tea production?",                 "Climate"),
    ("My area has low rainfall, will tea survive?",          "Climate"),
    ("Does growing tea at high altitude change the taste?",  "Climate"),
    ("How does moisture in the air affect disease?",         "Climate"),
]

_df = _model = _kb_embeddings = None


def init_rag(kb_path=KB_PATH):
    global _df, _model, _kb_embeddings
    print("  [RAG] Loading knowledge base and embedding model...")
    t0 = time.perf_counter()
    from rag_retrieval import load_knowledge_base, get_model_and_embeddings
    _df = load_knowledge_base(kb_path)
    _model, _kb_embeddings = get_model_and_embeddings(_df)
    print(f"  [RAG] Ready in {time.perf_counter() - t0:.2f}s\n")


def fast_retrieve(query, k=3):
    from rag_retrieval import correct_text, fuzzy_fallback
    qc = correct_text(query)
    if len(qc) < 2:
        return []
    q_emb = sk_normalize(_model.encode([qc], convert_to_numpy=True), axis=1)
    sims = np.dot(_kb_embeddings, q_emb.T).squeeze()
    if np.isscalar(sims):
        sims = np.array([sims])
    top_idx = sims.argsort()[-min(k, len(sims)):][::-1]
    results = []
    for idx in top_idx:
        score = float(sims[idx])
        row = _df.iloc[idx]
        if score < 0.45:
            fb = fuzzy_fallback(qc, _df["question_norm"].tolist())
            if fb:
                fbl = _df.index[_df["question_norm"] == fb].tolist()
                if fbl:
                    row, score = _df.loc[fbl[0]], 0.5
                else:
                    continue
            else:
                continue
        results.append({
            "question": row["question"],
            "answer":   row["answer"],
            "score":    round(score, 3)
        })
    return results


PROMPT = """You are evaluating GPTea, an AI advisor for Sri Lankan tea farmers.
Judge if the retrieved answers address the farmer's question.

FARMER_QUESTION: {question}

RETRIEVED_ANSWERS:
{answers}

Score on TWO criteria:
1. RELEVANCE (0-5): Does the retrieved content relate to the topic?
2. CORRECTNESS (0-5): Does it contain enough info to answer the question?

VERDICT: PASS if BOTH relevance >= 3 AND correctness >= 3, otherwise FAIL.

Reply ONLY with a JSON object. No markdown, no extra text, just raw JSON:
{{"relevance": <0-5>, "correctness": <0-5>, "verdict": "PASS" or "FAIL", "reasoning": "<one sentence>"}}"""


def judge(client, question, docs, model_name):
    if not docs:
        return {
            "relevance": 0, "correctness": 0,
            "verdict": "FAIL", "reasoning": "No documents retrieved."
        }

    answers = ""
    for i, d in enumerate(docs, 1):
        answers += f"\n[{i}] Matched Q: {d['question']}\nAnswer: {d['answer']}\nScore: {d['score']}\n"

    prompt = PROMPT.format(question=question, answers=answers)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        raw = response.text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        return {"relevance": 0, "correctness": 0, "verdict": "FAIL",
                "reasoning": f"Unparseable response: {str(e)}"}
    except Exception as e:
        return {"relevance": 0, "correctness": 0, "verdict": "FAIL",
                "reasoning": str(e)}


def run_eval(judge_model=DEFAULT_JUDGE_MODEL, save_path=None):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Add to your .env file:\n"
            "GOOGLE_API_KEY=your_google_api_key_here"
        )

    client = genai.Client(api_key=api_key)

    print("=" * 65)
    print(f"  GPTea — LLM-as-Judge (Google Gemini: {judge_model})")
    print(f"  RAG Model : all-MiniLM-L6-v2  (sentence-transformers)")
    print("=" * 65)
    print(f"\n  {'#':<4} {'STATUS':<6} {'REL':<5} {'COR':<5} {'RAG_T':<8} {'JDG_T':<8} CATEGORY")
    print(f"  {'-'*65}")

    results = []
    passed = failed = no_ret = 0
    rag_times, judge_times = [], []

    for i, (query, cat) in enumerate(UNSEEN_TESTS, 1):
        t0 = time.perf_counter()
        docs = fast_retrieve(query)
        rt = time.perf_counter() - t0
        rag_times.append(rt)
        if not docs:
            no_ret += 1

        t0 = time.perf_counter()
        j = judge(client, query, docs, judge_model)
        jt = time.perf_counter() - t0
        judge_times.append(jt)

        v = j.get("verdict", "FAIL")
        r = j.get("relevance", 0)
        c = j.get("correctness", 0)

        if v == "PASS":
            passed += 1
        else:
            failed += 1

        results.append({
            "n": i, "query": query, "category": cat,
            "relevance": r, "correctness": c, "verdict": v,
            "reasoning": j.get("reasoning", ""),
            "rag_time": round(rt, 3), "judge_time": round(jt, 3)
        })

        print(f"  {i:<4} {v:<6} {r:<5} {c:<5} {rt:.3f}s   {jt:.3f}s   {cat}")

    # Summary
    total = passed + failed
    acc = passed / total * 100 if total else 0
    all_rel = [r["relevance"] for r in results]
    all_cor = [r["correctness"] for r in results]

    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  Passed        : {passed}/{total} ({acc:.1f}%)")
    print(f"  Failed        : {failed}/{total}")
    print(f"  No Retrieval  : {no_ret}/{total}")
    print(f"\n  Avg Relevance    : {np.mean(all_rel):.2f}/5")
    print(f"  Avg Correctness  : {np.mean(all_cor):.2f}/5")
    print(f"\n  RAG avg latency   : {np.mean(rag_times):.3f}s")
    print(f"  Judge avg latency : {np.mean(judge_times):.3f}s")
    print(f"  Total avg latency : {np.mean([r+j for r,j in zip(rag_times, judge_times)]):.3f}s")

    print(f"\n  Category Breakdown:")
    for cat in ["Soil & pH", "Fertilizer", "Pests", "Irrigation", "Yield", "Climate"]:
        cr = [r for r in results if r["category"] == cat]
        cp = sum(1 for r in cr if r["verdict"] == "PASS")
        bar = "█" * cp + "░" * (len(cr) - cp)
        avg_r = np.mean([r["relevance"] for r in cr])
        avg_c = np.mean([r["correctness"] for r in cr])
        print(f"    {cat:<14}: {cp}/{len(cr)} [{bar}]  Rel:{avg_r:.1f} Cor:{avg_c:.1f}")

    failures = [r for r in results if r["verdict"] == "FAIL"]
    if failures:
        print(f"\n  Failed Cases ({len(failures)}):")
        for f in failures:
            print(f"\n    [{f['n']}] {f['category']} — Rel:{f['relevance']} Cor:{f['correctness']}")
            print(f"    Q: {f['query']}")
            print(f"    ↳ {f['reasoning']}")
    else:
        print(f"\n  ✓ PERFECT SCORE — All 30 questions passed!")

    print(f"\n{'='*65}")
    print(f"  Judge: Google Gemini {judge_model} (free tier)")
    print(f"  Evaluates semantic correctness, not keyword matching.")
    print(f"{'='*65}\n")

    if save_path:
        report = {
            "judge_model": judge_model,
            "total": total, "passed": passed, "failed": failed,
            "accuracy_pct": round(acc, 2),
            "avg_relevance": round(float(np.mean(all_rel)), 3),
            "avg_correctness": round(float(np.mean(all_cor)), 3),
            "avg_rag_latency": round(float(np.mean(rag_times)), 3),
            "avg_judge_latency": round(float(np.mean(judge_times)), 3),
            "results": results,
        }
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {save_path}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="GPTea LLM-as-Judge (Gemini)")
    parser.add_argument("--kb",          default=KB_PATH,            help="Path to domain.csv")
    parser.add_argument("--model",       default=DEFAULT_JUDGE_MODEL, help="Gemini model ID")
    parser.add_argument("--save-report", default=None,               help="Save JSON report to path")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  GPTea — LLM-as-Judge (Google Gemini, Free Tier)        ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    init_rag(args.kb)
    run_eval(judge_model=args.model, save_path=args.save_report)


if __name__ == "__main__":
    main()