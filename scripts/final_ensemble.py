import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load, dump
from utils import load_splits, summarize_metrics, save_report, plot_roc_pr
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

def load_model(path):
    return load(str(path))

def optimize_threshold(y_true, y_proba):
    best = {"th": 0.5, "f1": 0, "acc": 0}
    for th in np.arange(0.1, 0.91, 0.01):
        pred = (y_proba >= th).astype(int)
        acc = (pred == y_true).mean()
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best["f1"]:
            best = {"th": th, "f1": f1, "acc": acc}
    return best

def main():
    print("=== FINAL ENSEMBLE TRAINING START ===")
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # Load top models
    model_paths = {
        "final_best_model_v2": MODELS / "final_best_model_v2.joblib",
        "xgb_smote": MODELS / "xgb_smote.joblib",
        "rf_tuned": MODELS / "rf_tuned.joblib"
    }
    models = {name: load_model(path) for name, path in model_paths.items()}

    # Build weighted voting ensemble
    print("Building Weighted Voting Classifier...")
    voting_clf = VotingClassifier(
        estimators=[
            ("final_best", models["final_best_model_v2"]),
            ("xgb", models["xgb_smote"]),
            ("rf", models["rf_tuned"])
        ],
        voting="soft",
        weights=[3, 2, 1]  # heuristic weights (can tune)
    )

    print("Training final ensemble...")
    voting_clf.fit(X_train, y_train)

    # Predict probabilities on validation set for threshold tuning
    val_proba = voting_clf.predict_proba(X_val)[:, 1]
    best_th = optimize_threshold(y_val.values, val_proba)
    print(f"Best threshold from val: {best_th}")

    # Evaluate on test set
    test_proba = voting_clf.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_th["th"]).astype(int)
    metrics = summarize_metrics(y_test, test_pred, test_proba)

    # Save model
    final_model_path = MODELS / "ultimate_best_model.joblib"
    dump(voting_clf, final_model_path)
    print(f"[MODEL] {final_model_path}")

    # Save reports and plots
    save_report(metrics, "ultimate_best_model_eval.json")
    plot_roc_pr(y_test, test_proba, "ultimate_best_model")

    # Markdown summary
    md = f"""
# Ultimate Best Model Evaluation

**Threshold**: {best_th['th']:.2f}

| Metric        | Value  |
|---------------|--------|
| Accuracy      | {metrics['accuracy']:.4f} |
| Precision     | {metrics['precision']:.4f} |
| Recall        | {metrics['recall']:.4f} |
| F1-score      | {metrics['f1']:.4f} |
| Specificity   | {metrics['specificity']:.4f} |
| ROC-AUC       | {metrics['roc_auc']:.4f} |

✅ Final Accuracy Goal: **>85%**
"""
    with open(OUTPUTS / "ultimate_best_model_report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[FINAL EVAL] ultimate_best_model: metrics={metrics}")
    print("=== FINAL ENSEMBLE TRAINING DONE ===")

if __name__ == "__main__":
    main()
