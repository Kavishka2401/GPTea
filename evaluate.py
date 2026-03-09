#!/usr/bin/env python3
"""
evaluate.py

Comprehensive evaluation suite for the GPTea system covering:
  1. Classification metrics  — precision, recall, F1, accuracy
  2. RAG retrieval quality   — hit rate, score thresholds, spell correction, edge cases
  3. Chatbot response quality — groundedness, relevance, hallucination checks

Usage:
    # Run everything
    python evaluate.py

    # Run only one section
    python evaluate.py --section classification
    python evaluate.py --section rag
    python evaluate.py --section chatbot
"""

import argparse
import json
import sys
import os

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ============================================================
# SECTION 1: CLASSIFICATION METRICS
# ============================================================

def evaluate_classification(y_true, y_pred, average="binary"):
    """
    Evaluate classification performance.

    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        average (str): 'binary', 'macro', 'micro', or 'weighted'
                       - binary:   for 2-class problems
                       - macro:    average across classes equally
                       - micro:    global TP/FP/FN
                       - weighted: weighted by class frequency

    Returns:
        dict: accuracy, precision, recall, f1, n_samples
    """
    metrics = {
        "accuracy":   accuracy_score(y_true, y_pred),
        "precision":  precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall":     recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1":         f1_score(y_true, y_pred, average=average, zero_division=0),
        "n_samples":  len(y_true),
    }
    return metrics


def run_classification_eval():
    print("=" * 60)
    print("SECTION 1: CLASSIFICATION METRICS")
    print("=" * 60)

    # --- Test 1: Binary classification ---
    y_true_bin = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred_bin = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

    print("\n[1a] Binary Classification")
    metrics_bin = evaluate_classification(y_true_bin, y_pred_bin, average="binary")
    for k, v in metrics_bin.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Test 2: Perfect predictions ---
    y_perfect = [1, 0, 1, 0, 1]
    print("\n[1b] Perfect Binary Predictions (all metrics should be 1.0)")
    metrics_perfect = evaluate_classification(y_perfect, y_perfect, average="binary")
    for k, v in metrics_perfect.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    assert metrics_perfect["f1"] == 1.0, "FAIL: Perfect predictions should give F1=1.0"
    print("  PASS: F1 = 1.0 as expected")

    # --- Test 3: All-wrong predictions ---
    y_true_wrong = [1, 1, 1, 1, 1]
    y_pred_wrong = [0, 0, 0, 0, 0]
    print("\n[1c] All-Wrong Predictions (F1 should be 0.0)")
    metrics_wrong = evaluate_classification(y_true_wrong, y_pred_wrong, average="binary")
    for k, v in metrics_wrong.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    assert metrics_wrong["f1"] == 0.0, "FAIL: All-wrong predictions should give F1=0.0"
    print("  PASS: F1 = 0.0 as expected")

    # --- Test 4: Multi-class macro ---
    y_true_multi = ["fertilizer", "disease", "fertilizer", "irrigation", "disease"]
    y_pred_multi = ["fertilizer", "fertilizer", "fertilizer", "irrigation", "disease"]

    print("\n[1d] Multi-class Classification (macro average)")
    metrics_macro = evaluate_classification(y_true_multi, y_pred_multi, average="macro")
    for k, v in metrics_macro.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Test 5: Multi-class weighted ---
    print("\n[1e] Multi-class Classification (weighted average)")
    metrics_weighted = evaluate_classification(y_true_multi, y_pred_multi, average="weighted")
    for k, v in metrics_weighted.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Test 6: Single sample ---
    print("\n[1f] Single Sample Edge Case")
    metrics_single = evaluate_classification([1], [1], average="binary")
    for k, v in metrics_single.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    assert metrics_single["accuracy"] == 1.0, "FAIL: Single correct sample should give accuracy=1.0"
    print("  PASS: Single sample handled correctly")

    print("\n[Classification] All tests passed.\n")


# ============================================================
# SECTION 2: RAG RETRIEVAL EVALUATION
# ============================================================

# Ground-truth test set: (query, expected_keyword_in_answer)
RAG_TEST_CASES = [
    # Format: (query, keyword that must appear in the top answer, description)
    ("What is the ideal soil pH for tea?",          "4.5",              "Exact KB match — soil pH"),
    ("best fertilizer for tea plants",              "NPK",              "Fertilizer recommendation"),
    ("how to identify red spider mite",             "mite",             "Pest identification"),
    ("blister blight symptoms",                     "blight",           "Disease symptoms"),
    ("how to prune tea bushes",                     "prune",            "Pruning advice"),
    ("weed control without chemicals",              "weed",             "Organic weed management"),
    ("signs of nitrogen deficiency",                "nitrogen",         "Nutrient deficiency"),
    ("irrigation timing for tea",                   "irrigat",          "Irrigation guidance"),
    ("how does humidity affect tea",                "humid",            "Climate/humidity"),
    ("storage of harvested tea leaves",             "stor",             "Post-harvest storage"),
]

# Typo/spell-correction test cases: queries with deliberate misspellings
SPELL_TEST_CASES = [
    ("how to manage humidte in tea",    "humid",   "SymSpell: humidte → humidity"),
    ("fetilizer for tee plants",        "NPK",     "SymSpell: fetilizer/tee → fertilizer/tea"),
    ("pruning tee bushs",               "prune",   "SymSpell: tee/bushs → tea/bushes"),
]


def run_rag_eval(kb_path="data/domain.csv"):
    print("=" * 60)
    print("SECTION 2: RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    try:
        from rag_retrieval import retrieve_docs
    except ImportError:
        print("  SKIP: rag_retrieval.py not found. Place it in the same directory.")
        return

    passed = 0
    failed = 0
    results_log = []

    # --- Test 2a: Standard retrieval hit rate ---
    print(f"\n[2a] Standard Retrieval — {len(RAG_TEST_CASES)} test queries")
    for query, keyword, desc in RAG_TEST_CASES:
        docs = retrieve_docs(query, k=3, path=kb_path)
        if isinstance(docs, dict):
            status = "FAIL"
            detail = f"No match returned: {docs.get('message', '')}"
            failed += 1
        else:
            combined_answers = " ".join(d["answer"].lower() for d in docs)
            if keyword.lower() in combined_answers:
                status = "PASS"
                detail = f"score={docs[0]['score']:.3f}  matched_q='{docs[0]['question'][:50]}'"
                passed += 1
            else:
                status = "FAIL"
                detail = f"Keyword '{keyword}' not found in top-3 answers"
                failed += 1
        print(f"  [{status}] {desc}")
        if status == "FAIL":
            print(f"         → {detail}")
        results_log.append({"query": query, "status": status, "detail": detail})

    # --- Test 2b: Spell-corrected queries ---
    print(f"\n[2b] Spell Correction Retrieval — {len(SPELL_TEST_CASES)} typo queries")
    for query, keyword, desc in SPELL_TEST_CASES:
        docs = retrieve_docs(query, k=3, path=kb_path)
        if isinstance(docs, dict):
            status = "FAIL"
            detail = f"No match after spell correction: {docs.get('message', '')}"
            failed += 1
        else:
            combined_answers = " ".join(d["answer"].lower() for d in docs)
            if keyword.lower() in combined_answers:
                status = "PASS"
                detail = f"score={docs[0]['score']:.3f}"
                passed += 1
            else:
                status = "FAIL"
                detail = f"Keyword '{keyword}' not found after spell correction"
                failed += 1
        print(f"  [{status}] {desc}")
        if status == "FAIL":
            print(f"         → {detail}")

    # --- Test 2c: Score threshold check ---
    print("\n[2c] Score Threshold Check (all results must be >= 0.45)")
    threshold_pass = True
    for query, _, _ in RAG_TEST_CASES[:5]:
        docs = retrieve_docs(query, k=3, path=kb_path)
        if isinstance(docs, list):
            for doc in docs:
                if doc["score"] < 0.45:
                    print(f"  FAIL: score {doc['score']} below threshold for query: '{query}'")
                    threshold_pass = False
    if threshold_pass:
        print("  PASS: All returned scores meet the 0.45 minimum threshold")

    # --- Test 2d: Edge cases ---
    print("\n[2d] Edge Cases")

    # Too short query
    result_short = retrieve_docs("hi", k=3, path=kb_path)
    if isinstance(result_short, dict) and ("error" in result_short or "message" in result_short):
        print("  PASS: Short/vague query handled gracefully")
    else:
        print("  NOTE: Short query returned results — may want to review")

    # Gibberish query
    result_gibberish = retrieve_docs("xzqwerty blahhhhh foobar", k=3, path=kb_path)
    if isinstance(result_gibberish, dict):
        print("  PASS: Gibberish query returned no-match dict (not a crash)")
    else:
        top_score = result_gibberish[0]["score"] if result_gibberish else 0
        if top_score < 0.55:
            print(f"  PASS: Gibberish query returned low-confidence result (score={top_score:.3f})")
        else:
            print(f"  WARN: Gibberish query returned high score={top_score:.3f} — check min_score threshold")

    # Empty string
    result_empty = retrieve_docs("", k=3, path=kb_path)
    if isinstance(result_empty, dict):
        print("  PASS: Empty query handled gracefully")
    else:
        print("  WARN: Empty query returned results — unexpected")

    # --- Summary ---
    total = passed + failed
    hit_rate = passed / total * 100 if total > 0 else 0
    print(f"\n[RAG Summary] Hit Rate: {passed}/{total} ({hit_rate:.1f}%) | Failed: {failed}")
    print()


# ============================================================
# SECTION 3: CHATBOT RESPONSE QUALITY EVALUATION
# ============================================================

# Test cases: (query, must_contain, must_not_contain, description)
CHATBOT_TEST_CASES = [
    (
        "What is the ideal soil pH for tea?",
        ["4.5", "5.5"],
        ["visit", "come to your farm"],
        "pH answer grounded in KB"
    ),
    (
        "How do I identify red spider mite?",
        ["mite", "leaf"],
        ["visit", "I will come"],
        "Pest ID grounded in KB"
    ),
    (
        "What fertilizer should I use?",
        ["fertilizer", "NPK"],
        ["visit your estate", "I can visit"],
        "Fertilizer advice grounded in KB"
    ),
    (
        "How do I prune tea bushes?",
        ["prune", "bush"],
        ["visit", "come to"],
        "Pruning advice grounded in KB"
    ),
    (
        "How can I control weeds without chemicals?",
        ["weed", "mulch"],
        ["visit", "personally assess"],
        "Organic weed management"
    ),
]

def score_response(response: str, must_contain: list, must_not_contain: list):
    """
    Simple keyword-based scoring of a chatbot response.
    Returns (passed: bool, contain_hits: list, contain_misses: list, hallucination_hits: list)
    """
    resp_lower = response.lower()
    contain_hits   = [kw for kw in must_contain    if kw.lower() in resp_lower]
    contain_misses = [kw for kw in must_contain    if kw.lower() not in resp_lower]
    halluc_hits    = [kw for kw in must_not_contain if kw.lower() in resp_lower]
    passed = len(contain_misses) == 0 and len(halluc_hits) == 0
    return passed, contain_hits, contain_misses, halluc_hits


def run_chatbot_eval():
    print("=" * 60)
    print("SECTION 3: CHATBOT RESPONSE QUALITY EVALUATION")
    print("=" * 60)

    try:
        from chatbot_openai import tea_chain
    except ImportError:
        print("  SKIP: chatbot_openai.py not found or GROQ_API_KEY not set.")
        print("  To run chatbot eval, ensure chatbot_openai.py and .env are present.\n")
        return
    except Exception as e:
        print(f"  SKIP: Could not load chatbot — {e}\n")
        return

    passed = 0
    failed = 0

    print(f"\n[3a] Response Quality — {len(CHATBOT_TEST_CASES)} test queries")
    for query, must_contain, must_not_contain, desc in CHATBOT_TEST_CASES:
        try:
            response = tea_chain.invoke(query)
            ok, hits, misses, halluc = score_response(response, must_contain, must_not_contain)
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"\n  [{status}] {desc}")
            print(f"  Query   : {query}")
            print(f"  Keywords found    : {hits}")
            if misses:
                print(f"  Keywords missing  : {misses}")
            if halluc:
                print(f"  Hallucination flags: {halluc}")
            print(f"  Response: {response[:200].strip()}{'...' if len(response) > 200 else ''}")
        except Exception as e:
            print(f"\n  [ERROR] {desc} — {e}")
            failed += 1

    # --- Test 3b: Response is a non-empty string ---
    print("\n[3b] Response Format Check")
    try:
        response = tea_chain.invoke("What is tea?")
        assert isinstance(response, str), "Response should be a string"
        assert len(response.strip()) > 10, "Response should not be near-empty"
        print("  PASS: Response is a non-empty string")
    except Exception as e:
        print(f"  FAIL: {e}")

    # --- Test 3c: Out-of-domain query ---
    print("\n[3c] Out-of-Domain Query Handling")
    try:
        response = tea_chain.invoke("What is the capital of France?")
        resp_lower = response.lower()
        if any(w in resp_lower for w in ["not", "context", "unable", "outside", "don't", "cannot"]):
            print("  PASS: Model correctly declined or flagged out-of-domain query")
        else:
            print(f"  WARN: Model answered out-of-domain query without disclaimer")
            print(f"  Response: {response[:150].strip()}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Summary ---
    total = passed + failed
    print(f"\n[Chatbot Summary] Passed: {passed}/{total} | Failed: {failed}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GPTea Evaluation Suite")
    parser.add_argument(
        "--section",
        choices=["classification", "rag", "chatbot", "all"],
        default="all",
        help="Which section to run (default: all)"
    )
    parser.add_argument(
        "--kb",
        type=str,
        default="data/domain.csv",
        help="Path to knowledge base CSV (default: data/domain.csv)"
    )
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║              GPTea — Full Evaluation Suite               ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    if args.section in ("classification", "all"):
        run_classification_eval()

    if args.section in ("rag", "all"):
        run_rag_eval(kb_path=args.kb)

    if args.section in ("chatbot", "all"):
        run_chatbot_eval()

    print("=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()