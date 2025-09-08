"""
Final validation for Risk Categorization System using rf_tuned model.
Generates:
- Accuracy & Performance Report
- Risk Category Distribution
- Histogram of Probabilities with Thresholds
- Markdown Summary Report for submission
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATA = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

MODEL_PATH = MODELS / "rf_tuned.joblib"
DATA_PATH = DATA / "heart_features.csv"

THRESHOLDS = {"low": 0.0, "moderate": 0.3, "high": 0.7}


def classify_risk(prob):
    if prob < THRESHOLDS["moderate"]:
        return "Low"
    if prob < THRESHOLDS["high"]:
        return "Moderate"
    return "High"


def main():
    print("=== FINAL VALIDATION START ===")

    # Load model and data
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    if "target" not in df.columns:
        raise RuntimeError("Expected 'target' column in heart_features.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    # Predictions
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("rf_tuned model does not support predict_proba")

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    roc_auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds)

    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)

    # Risk categorization
    risk_categories = [classify_risk(p) for p in probs]
    category_counts = pd.Series(risk_categories).value_counts()

    # Plot histogram of probabilities
    plt.figure(figsize=(8, 6))
    plt.hist(probs, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    plt.axvline(THRESHOLDS["moderate"], color="orange", linestyle="--", label="Moderate Threshold (0.3)")
    plt.axvline(THRESHOLDS["high"], color="red", linestyle="--", label="High Threshold (0.7)")
    plt.title("Risk Probability Distribution with Thresholds")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    hist_path = OUTPUTS / "risk_distribution.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()

    print(f"[SAVED] Probability distribution plot -> {hist_path}")

    # Classification report
    cls_report = classification_report(y, preds, digits=4)

    # Save markdown report
    report_path = OUTPUTS / "final_validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Final Validation Report\n\n")
        f.write("## Model Performance\n")
        f.write(f"- Accuracy: {acc:.4f}\n")
        f.write(f"- F1 Score: {f1:.4f}\n")
        f.write(f"- ROC-AUC: {roc_auc:.4f}\n\n")
        f.write("### Confusion Matrix\n")
        f.write(f"```\n{cm}\n```\n\n")
        f.write("### Classification Report\n")
        f.write(f"```\n{cls_report}\n```\n\n")

        f.write("## Risk Categorization\n")
        f.write(f"- Thresholds: Low < 0.3 | Moderate < 0.7 | High ≥ 0.7\n")
        f.write("### Category Counts\n")
        f.write(f"```\n{category_counts.to_string()}\n```\n\n")

        f.write("![Risk Distribution](risk_distribution.png)\n\n")

        if acc >= 0.85:
            f.write("✅ **Accuracy target of 85% achieved.**\n")
        else:
            f.write("⚠ **Accuracy target not achieved. Further tuning required.**\n")

    print(f"[SAVED] Final report -> {report_path}")
    print("=== FINAL VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()
