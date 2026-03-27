import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)
import argparse
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


SAMPLE_CASES = [
    # --- Soil & pH (6 questions) ---
    ("My soil is too alkaline what should I do?",               ["acid", "pH"],              "Soil & pH",       True),
    ("What pH level is best for growing tea?",                  ["5.5", "acidic"],           "Soil & pH",       True),
    ("How do I improve the organic content of my soil?",        ["compost", "mulch"],        "Soil & pH",       True),
    ("How do I test my soil before planting tea?",              ["soil", "test"],            "Soil & pH",       True),
    ("My soil is very sandy is that bad for tea?",              ["drain", "nutrient"],       "Soil & pH",       True),
    ("How does soil compaction affect my tea plants?",          ["root", "soil"],            "Soil & pH",       True),

    # --- Fertilizer (8 questions) ---
    ("My plants are not growing fast what fertilizer should I use?", ["fertiliz", "NPK"],   "Fertilizer",      True),
    ("Can I use natural compost instead of fertilizer?",        ["compost", "organic"],      "Fertilizer",      True),
    ("When is the worst time to apply fertilizer?",             ["rain", "waterlog"],        "Fertilizer",      True),
    ("How do I avoid burning my tea leaves with fertilizer?",   ["fertiliz", "burn"],        "Fertilizer",      True),
    ("What does potassium do for my tea plants?",               ["potassium", "quality"],    "Fertilizer",      True),
    ("How often should I fertilize my mature tea bushes?",      ["fertiliz", "year"],        "Fertilizer",      True),
    ("Can I use urea on my tea plants?",                        ["urea", "nitrogen"],        "Fertilizer",      True),
    ("What is split application of fertilizer?",                ["split", "smaller"],        "Fertilizer",      True),

    # --- Pests & Disease (10 questions) ---
    ("There are tiny bugs under my leaves what could it be?",   ["pest", "insect"],          "Pests & Disease", True),
    ("White spots are appearing on my tea leaves",              ["fungal", "disease"],       "Pests & Disease", True),
    ("How do I get rid of insects without using chemicals?",    ["neem", "organic"],         "Pests & Disease", True),
    ("My tea shoots are drying and turning brown at the tips",  ["pest", "brown"],           "Pests & Disease", True),
    ("How do I prevent fungal diseases after heavy rain?",      ["fungal", "drain"],         "Pests & Disease", True),
    ("What are the signs of red spider mite on tea?",           ["mite", "yellow"],          "Pests & Disease", True),
    ("My leaves are curling and turning dry what is wrong?",    ["mite", "stress"],          "Pests & Disease", True),
    ("How do I control aphids on my tea plants?",               ["neem", "aphid"],           "Pests & Disease", True),
    ("What is blister blight and how do I treat it?",           ["blight", "fungal"],        "Pests & Disease", True),
    ("Can I mix two pesticides together?",                      ["label", "mix"],            "Pests & Disease", True),

    # --- Irrigation (6 questions) ---
    ("How much water do tea plants need in dry weather?",       ["irrigat", "water"],        "Irrigation",      True),
    ("Is drip watering good for tea farms?",                    ["drip", "water"],           "Irrigation",      True),
    ("How do I stop my field from flooding after rain?",        ["drain", "water"],          "Irrigation",      True),
    ("How do I know when my tea plants need water?",            ["moisture", "soil"],        "Irrigation",      True),
    ("Can rainwater be collected and used for tea irrigation?", ["rain", "water"],           "Irrigation",      True),
    ("How do I irrigate tea plants on a slope?",                ["slope", "erosion"],        "Irrigation",      True),

    # --- Yield (6 questions) ---
    ("Why am I getting less tea than last year?",               ["yield", "factor"],         "Yield",           True),
    ("How can I get more leaves per bush?",                     ["fertiliz", "nutrient"],    "Yield",           True),
    ("Does shade help increase my tea yield?",                  ["shade", "yield"],          "Yield",           True),
    ("How long after pruning will my yield recover?",           ["prune", "recover"],        "Yield",           True),
    ("What is the average tea yield per hectare?",              ["yield", "hectare"],        "Yield",           True),
    ("Why are my tea bushes producing small leaves?",           ["nutrient", "deficien"],    "Yield",           True),

    # --- Climate (6 questions) ---
    ("Is hot weather bad for my tea plants?",                   ["temperature", "hot"],      "Climate",         True),
    ("How does rain affect tea production?",                    ["rain", "rainfall"],        "Climate",         True),
    ("Can tea plants survive a drought?",                       ["drought", "yield"],        "Climate",         True),
    ("What temperature is best for tea to grow?",               ["temperature", "growth"],   "Climate",         True),
    ("Does altitude affect the quality of my tea?",             ["altitude", "quality"],     "Climate",         True),
    ("How does humidity affect disease in my tea field?",       ["humid", "disease"],        "Climate",         True),

    # --- Out of Scope (8 questions) ---
    ("What is the best recipe for milk tea?",                   [],                          "Out of Scope",    False),
    ("How do I repair my irrigation pump motor?",               [],                          "Out of Scope",    False),
    ("What is the weather forecast for this week?",             [],                          "Out of Scope",    False),
    ("How do I get a government subsidy for my farm?",          [],                          "Out of Scope",    False),
    ("Can you teach me how to cook rice?",                      [],                          "Out of Scope",    False),
    ("What is the stock price of tea companies today?",         [],                          "Out of Scope",    False),
    ("How do I fix my tractor engine?",                         [],                          "Out of Scope",    False),
    ("What is the capital city of Sri Lanka?",                  [],                          "Out of Scope",    False),
]


DECLINE_KEYWORDS = [
    "outside the scope",
    "not related to tea",
    "not relevant",
    "not covered",
    "cannot answer",
    "unable to",
    "no information provided",
    "not enough information",
    "does not contain",
    "outside",
    "scope of tea estate",
]


def score_response(response: str, must_contain: list, should_answer: bool) -> dict:
    resp_lower = response.lower()
    if not should_answer:
        # PASS = correctly declined (said outside scope)
        declined = any(kw in resp_lower for kw in DECLINE_KEYWORDS)
        return {"passed": declined, "contain_hits": [], "contain_misses": [], "declined": declined}
    contain_hits   = [kw for kw in must_contain if kw.lower() in resp_lower]
    contain_misses = [kw for kw in must_contain if kw.lower() not in resp_lower]
    passed = len(contain_misses) == 0
    return {"passed": passed, "contain_hits": contain_hits, "contain_misses": contain_misses, "declined": False}


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
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy": round(accuracy, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def run_eval(save_path=None):
    try:
        from chatbot_openai import tea_chain
    except Exception as e:
        print(f"  ERROR: Could not load chatbot — {e}")
        return

    in_scope  = sum(1 for _, _, _, s in SAMPLE_CASES if s)
    out_scope = sum(1 for _, _, _, s in SAMPLE_CASES if not s)

    print("=" * 68)
    print(f"  GPTea — Chatbot Response Evaluation")
    print(f"  Tests : {len(SAMPLE_CASES)} questions ({in_scope} in-scope  {out_scope} out-of-scope)")
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
            elif verdict == "PASS" and not should_answer: outcome = "TN"
            elif verdict == "FAIL" and     should_answer: outcome = "FN"
            else:                                          outcome = "FP"

            results.append({"n": i, "query": query, "category": category,
                             "should_answer": should_answer, "verdict": verdict,
                             "outcome": outcome, "response": response,
                             "keywords_checked": must_contain, **score})

            icon = "✓" if verdict == "PASS" else "✗"
            print(f"  {i:<4} {icon} {verdict:<6} {outcome:<8} {category} — {query[:50]}")
            if verdict == "FAIL":
                if score.get("contain_misses"):
                    print(f"       Missing : {score['contain_misses']}")
                if not should_answer and not score["declined"]:
                    print(f"       Answered when it should have declined")

        except Exception as e:
            print(f"  {i:<4} ✗ ERROR          {category} — {e}")
            results.append({"n": i, "query": query, "category": category,
                             "should_answer": should_answer, "verdict": "FAIL",
                             "outcome": "FN" if should_answer else "FP", "error": str(e)})

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
    print(f"  Accuracy   : {m['accuracy']*100:.1f}%")
    print(f"  Precision  : {m['precision']*100:.1f}%")
    print(f"  Recall     : {m['recall']*100:.1f}%")
    print(f"  F1 Score   : {m['f1']*100:.1f}%")

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
            print(f"  Q       : {r['query']}")
            if r.get("contain_misses"):
                print(f"  Missing : {r['contain_misses']}")
            if r.get("response"):
                print(f"  Response: {r['response'][:150].strip()}...")
    else:
        print(f"\n  PERFECT — No misclassifications!")

    print(f"\n{'='*68}\n")

    if save_path:
        with open(save_path, "w") as f:
            json.dump({"metrics": m, "results": results}, f, indent=2)
        print(f"  Report saved to: {save_path}\n")

    return m, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-report", default=None)
    args = parser.parse_args()
    print("\nGPTea — Chatbot Response Evaluation\n")
    run_eval(save_path=args.save_report)


if __name__ == "__main__":
    main()