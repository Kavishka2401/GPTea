

import argparse
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


SAMPLE_CASES = [
    (
        "My soil is too alkaline, what should I do?",
        ["acid", "pH"],
        "Soil & pH", True
    ),
    (
        "What pH level is best for growing tea?",
        ["4.5", "5.5"],
        "Soil & pH", True
    ),
    (
        "How do I improve the organic content of my soil?",
        ["compost", "mulch"],
        "Soil & pH", True
    ),

    # --- Fertilizer ---
    (
        "My plants are not growing fast, what fertilizer should I use?",
        ["nitrogen", "fertiliz"],
        "Fertilizer", True
    ),
    (
        "Can I use natural compost instead of chemical fertilizer?",
        ["compost", "nitrogen"],
        "Fertilizer", True
    ),
    (
        "When is the worst time to apply fertilizer?",
        ["rain", "drought"],
        "Fertilizer", True
    ),

    # --- Pests & Disease ---
    (
        "There are tiny bugs under my leaves, what could it be?",
        ["mite", "pest"],
        "Pests & Disease", True
    ),
    (
        "White spots are appearing on my tea leaves",
        ["blight", "fungal"],
        "Pests & Disease", True
    ),
    (
        "How do I get rid of insects without using chemicals?",
        ["neem", "organic"],
        "Pests & Disease", True
    ),
    (
        "My tea shoots are drying and turning brown at the tips",
        ["mosquito", "pest"],
        "Pests & Disease", True
    ),

    # --- Irrigation ---
    (
        "How much water do tea plants need in dry weather?",
        ["irrigat", "water"],
        "Irrigation", True
    ),
    (
        "Is drip watering good for tea farms?",
        ["drip", "water"],
        "Irrigation", True
    ),
    (
        "How do I stop my field from flooding after rain?",
        ["drain", "water"],
        "Irrigation", True
    ),

    # --- Yield ---
    (
        "Why am I getting less tea than last year?",
        ["yield", "prune"],
        "Yield", True
    ),
    (
        "How can I get more leaves per bush?",
        ["pluck", "prune"],
        "Yield", True
    ),

    # --- Climate ---
    (
        "Is hot weather bad for my tea plants?",
        ["temperature", "heat"],
        "Climate", True
    ),
    (
        "How does rain affect tea production?",
        ["rain", "yield"],
        "Climate", True
    ),

    # --- Out of Scope ---
    (
        "What is the best recipe for milk tea?",
        [],
        "Out of Scope", False
    ),
    (
        "How do I repair my irrigation pump motor?",
        [],
        "Out of Scope", False
    ),
    (
        "What is the weather forecast for this week?",
        [],
        "Out of Scope", False
    ),
    (
        "How do I get a government subsidy for my farm?",
        [],
        "Out of Scope", False
    ),
    (
        "Can you teach me how to cook rice?",
        [],
        "Out of Scope", False
    ),
]


DECLINE_KEYWORDS = [
    "not enough information in the context",
    "context provided does not contain",
    "context does not contain",
    "there is not enough information",
    "there is no information",
    "does not contain enough information",
    "not contain enough",
    "only discusses",
    "context only discusses",
    "outside the scope",
    "not related to tea",
    "not relevant",
    "not covered",
    "not able",
    "cannot answer",
    "unable to",
    "no information provided",
    "not enough information",
    "does not contain",
    "outside",
]


def score_response(response: str, must_contain: list, should_answer: bool) -> dict:
    resp_lower = response.lower()

    if not should_answer:
        declined = any(kw in resp_lower for kw in DECLINE_KEYWORDS)
        return {
            "passed":         not declined,  #
            "contain_hits":   [],
            "contain_misses": [],
            "declined":       declined,
        }

    contain_hits   = [kw for kw in must_contain if kw.lower() in resp_lower]
    contain_misses = [kw for kw in must_contain if kw.lower() not in resp_lower]
    passed = len(contain_misses) == 0

    return {
        "passed":         passed,
        "contain_hits":   contain_hits,
        "contain_misses": contain_misses,
        "declined":       False,
    }



def calculate_metrics(results: list) -> dict:
    TP = sum(1 for r in results if r["verdict"] == "PASS" and     r["should_answer"])
    FP = sum(1 for r in results if r["verdict"] == "PASS" and not r["should_answer"])
    FN = sum(1 for r in results if r["verdict"] == "FAIL" and     r["should_answer"])
    TN = sum(1 for r in results if r["verdict"] == "FAIL" and not r["should_answer"])

    total     = len(results)
    accuracy  = (TP + TN) / total                             if total > 0            else 0
    precision = TP / (TP + FP)                                if (TP + FP) > 0        else 0
    recall    = TP / (TP + FN)                                if (TP + FN) > 0        else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }



def run_eval(save_path=None):
    try:
        from tea_chain import tea_chain
    except ImportError:
        try:
            from chatbot_openai import tea_chain
        except Exception as e:
            print(f"  ERROR: Could not load chatbot — {e}")
            return

    in_scope  = sum(1 for _, _, _, s in SAMPLE_CASES if s)
    out_scope = sum(1 for _, _, _, s in SAMPLE_CASES if not s)

    print("=" * 68)
    print(f"  GPTea — Sample Question Evaluation")
    print(f"  Tests : {len(SAMPLE_CASES)} questions ({in_scope} in-scope, {out_scope} out-of-scope)")
    print(f"  Note  : Questions are rephrased — NOT copied from domain.csv")
    print("=" * 68)
    print(f"\n  {'#':<4} {'RESULT':<7} {'OUTCOME':<8} CATEGORY / QUERY")
    print(f"  {'-'*68}")

    results = []

    for i, (query, must_contain, category, should_answer) in enumerate(SAMPLE_CASES, 1):
        try:
            response = tea_chain.invoke(query)
            score    = score_response(response, must_contain, should_answer)
            verdict  = "PASS" if score["passed"] else "FAIL"

            if   verdict == "PASS" and     should_answer: outcome = "TP"
            elif verdict == "PASS" and not should_answer: outcome = "FP"
            elif verdict == "FAIL" and     should_answer: outcome = "FN"
            else:                                          outcome = "TN"

            results.append({
                "n": i, "query": query, "category": category,
                "should_answer": should_answer,
                "verdict": verdict, "outcome": outcome,
                "response": response,
                "keywords_checked": must_contain,
                **score,
            })

            icon = "✓" if verdict == "PASS" else "✗"
            print(f"  {i:<4} {icon} {verdict:<6} {outcome:<8} {category} — {query[:52]}")

            if verdict == "FAIL":
                if score.get("contain_misses"):
                    print(f"       Missing keywords : {score['contain_misses']}")
                if not should_answer and not score["declined"]:
                    print(f"       Answered when it should have declined")

        except Exception as e:
            print(f"  {i:<4} ✗ ERROR          {category} — {e}")
            results.append({
                "n": i, "query": query, "category": category,
                "should_answer": should_answer,
                "verdict": "FAIL", "outcome": "FN" if should_answer else "TN",
                "error": str(e),
            })

    m = calculate_metrics(results)

    print(f"\n{'='*68}")
    print(f"  CONFUSION MATRIX")
    print(f"{'='*68}")
    print(f"                        Predicted PASS    Predicted FAIL")
    print(f"  Actually SHOULD     |  TP = {m['TP']:<14}  FN = {m['FN']}")
    print(f"  Actually SKIP       |  FP = {m['FP']:<14}  TN = {m['TN']}")

    print(f"\n{'='*68}")
    print(f"  METRICS")
    print(f"{'='*68}")

    def grade(v):
        if v >= 0.85:   return "✓ Excellent"
        elif v >= 0.70: return "~ Good"
        else:           return "✗ Needs improvement"

    print(f"  Accuracy   : {m['accuracy']*100:.1f}%  {grade(m['accuracy'])}")
    print(f"  Precision  : {m['precision']*100:.1f}%  {grade(m['precision'])}")
    print(f"  Recall     : {m['recall']*100:.1f}%  {grade(m['recall'])}")
    print(f"  F1 Score   : {m['f1']*100:.1f}%  {grade(m['f1'])}")

    print(f"\n{'='*68}")
    print(f"  CATEGORY BREAKDOWN")
    print(f"{'='*68}")
    categories = list(dict.fromkeys(cat for _, _, cat, _ in SAMPLE_CASES))
    for cat in categories:
        cr      = [r for r in results if r["category"] == cat]
        correct = sum(1 for r in cr if r["outcome"] in ("TP", "TN"))
        tp = sum(1 for r in cr if r["outcome"] == "TP")
        fp = sum(1 for r in cr if r["outcome"] == "FP")
        fn = sum(1 for r in cr if r["outcome"] == "FN")
        tn = sum(1 for r in cr if r["outcome"] == "TN")
        bar = "█" * correct + "░" * (len(cr) - correct)
        print(f"  {cat:<16}: {correct}/{len(cr)} [{bar}]  TP:{tp} FP:{fp} FN:{fn} TN:{tn}")

    misses = [r for r in results if r["outcome"] in ("FP", "FN")]
    if misses:
        print(f"\n{'='*68}")
        print(f"  MISCLASSIFIED CASES ({len(misses)})")
        print(f"{'='*68}")
        for r in misses:
            label = "FALSE POSITIVE — answered when it should have declined" \
                    if r["outcome"] == "FP" \
                    else "FALSE NEGATIVE — failed to answer a valid question"
            print(f"\n  [{r['n']}] {label}")
            print(f"  Q        : {r['query']}")
            if r.get("keywords_checked"):
                print(f"  Keywords : {r['keywords_checked']}")
            if r.get("contain_misses"):
                print(f"  Missing  : {r['contain_misses']}")
            if r.get("response"):
                print(f"  Response : {r['response'][:150].strip()}...")
    else:
        print(f"\n  ✓ PERFECT — No misclassifications!")

    print(f"\n{'='*68}\n")

    if save_path:
        report = {"metrics": m, "results": results}
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {save_path}\n")

    return m, results


def main():
    parser = argparse.ArgumentParser(description="GPTea Sample Question Evaluator")
    parser.add_argument("--save-report", default=None, help="Save JSON report to path")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       GPTea — Sample Question Evaluation                ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    run_eval(save_path=args.save_report)


if __name__ == "__main__":
    main()