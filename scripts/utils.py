# scripts/utils.py
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend (no GUI issues on Windows)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve, auc
)

# Directories
ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
for d in [OUTPUTS, MODELS]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load Train/Val/Test Splits
# -----------------------------
def load_splits():
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_val   = pd.read_csv(DATA_PROCESSED / "X_val.csv")
    X_test  = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze("columns")
    y_val   = pd.read_csv(DATA_PROCESSED / "y_val.csv").squeeze("columns")
    y_test  = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze("columns")

    # Ensure Series format
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, name="target")
    y_train.name = "target"; y_val.name = "target"; y_test.name = "target"
    return X_train, X_val, X_test, y_train, y_val, y_test

# -----------------------------
# Metrics Summary
# -----------------------------
def summarize_metrics(y_true, y_pred, y_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    return metrics

# -----------------------------
# Save JSON Report
# -----------------------------
def save_report(obj: dict, fname: str):
    path = OUTPUTS / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[REPORT] {path}")

# -----------------------------
# Save Model
# -----------------------------
def save_model(model, fname: str):
    path = MODELS / fname
    joblib.dump(model, path)
    print(f"[MODEL] {path}")

# -----------------------------
# Plot ROC & Precision-Recall
# -----------------------------
def plot_roc_pr(y_true, y_proba, tag: str):
    try:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {tag}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUTS / f"roc_{tag}.png", dpi=150)
        plt.close()

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}", color="green")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {tag}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUTS / f"pr_{tag}.png", dpi=150)
        plt.close()

        print(f"[PLOT] {OUTPUTS / f'roc_{tag}.png'}\n[PLOT] {OUTPUTS / f'pr_{tag}.png'}")
    except Exception as e:
        print(f"Failed to plot curves for {tag}: {e}")

# -----------------------------
# Confusion Matrix Plot
# -----------------------------
def plot_confusion_matrix(y_true, y_pred, tag: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {tag}")
    plt.tight_layout()
    path = OUTPUTS / f"confusion_{tag}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] {path}")
