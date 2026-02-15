#!/usr/bin/env python3
"""
metrics_eval.py

Compute precision, recall, and F1 score for binary or multi-class classification.

Usage:
    python metrics_eval.py
"""

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_classification(y_true, y_pred, average="binary"):
    """
    Evaluate classification performance.

    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        average (str): 'binary', 'macro', 'micro', or 'weighted'
                       - binary: for 2-class problems
                       - macro: average across classes equally
                       - micro: global TP/FP/FN
                       - weighted: weighted by class frequency

    Returns:
        dict: accuracy, precision, recall, f1
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "n_samples": len(y_true)
    }
    return metrics

if __name__ == "__main__":
    # Example 1: Binary classification
    y_true_bin = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred_bin = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

    print("=== Binary Classification ===")
    metrics_bin = evaluate_classification(y_true_bin, y_pred_bin, average="binary")
    for k, v in metrics_bin.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    # Example 2: Multi-class classification
    y_true_multi = ["fertilizer", "disease", "fertilizer", "irrigation", "disease"]
    y_pred_multi = ["fertilizer", "fertilizer", "fertilizer", "irrigation", "disease"]

    print("\n=== Multi-class Classification (macro average) ===")
    metrics_macro = evaluate_classification(y_true_multi, y_pred_multi, average="macro")
    for k, v in metrics_macro.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== Multi-class Classification (weighted average) ===")
    metrics_weighted = evaluate_classification(y_true_multi, y_pred_multi, average="weighted")
    for k, v in metrics_weighted.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
