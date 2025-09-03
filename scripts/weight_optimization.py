import itertools
import json
from pathlib import Path
from joblib import load, dump
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from utils import load_splits, plot_roc_pr

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

def get_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }

def find_best_threshold(y_true, y_proba):
    best = {"th": 0.5, "f1": 0, "acc": 0}
    for th in np.arange(0.1, 0.9, 0.01):
        preds = (y_proba >= th).astype(int)
        f1 = f1_score(y_true, preds)
        acc = accuracy_score(y_true, preds)
        if f1 > best["f1"]:
            best = {"th": th, "f1": f1, "acc": acc}
    return best

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # Load top models
    rf = load(MODELS / "rf_tuned.joblib")
    svm = load(MODELS / "svm_tuned.joblib")
    xgb = load(MODELS / "xgb_smote.joblib")

    weight_combinations = [(a, b, c) for a, b, c in itertools.product([1, 2, 3, 4, 5], repeat=3)]

    best_score = 0
    best_combo = None
    best_model = None
    best_threshold = None

    print("=== OPTIMIZING WEIGHTS ===")
    for w in weight_combinations:
        model = VotingClassifier(
            estimators=[("rf", rf), ("svm", svm), ("xgb", xgb)],
            voting="soft",
            weights=w
        )
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        th_info = find_best_threshold(y_val, val_proba)
        if th_info["f1"] > best_score:
            best_score = th_info["f1"]
            best_combo = w
            best_threshold = th_info
            best_model = model

    print(f"Best weights: {best_combo}, Best threshold: {best_threshold}")

    # Save best model
    dump(best_model, MODELS / "ultimate_weighted_model.joblib")

    # Evaluate on test set
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= best_threshold["th"]).astype(int)
    metrics = get_metrics(y_test, test_preds, test_proba)

    print("[FINAL TEST METRICS]", metrics)

    # Save report
    report = {
        "best_weights": best_combo,
        "best_threshold": best_threshold,
        "test_metrics": metrics
    }
    with open(OUTPUTS / "ultimate_weighted_model_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Plots
    plot_roc_pr(y_test, test_proba, "ultimate_weighted_model")

    print(f"[MODEL] {MODELS / 'ultimate_weighted_model.joblib'}")
    print(f"[REPORT] {OUTPUTS / 'ultimate_weighted_model_report.json'}")

if __name__ == "__main__":
    main()
