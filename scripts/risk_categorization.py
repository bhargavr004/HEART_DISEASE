# scripts/risk_categorization.py
"""
Prediction + Risk Categorization CLI focused on rf_tuned.

Usage:
    python scripts\risk_categorization.py --input data/processed/heart_features.csv --out outputs/predictions.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
from typing import Dict, Any

# local utils
from utils import MODELS, OUTPUTS

# defaults
DEFAULT_CAL_MODEL = MODELS / "rf_tuned_calibrated.joblib"
DEFAULT_MODEL = MODELS / "rf_tuned.joblib"
OUTPUTS.mkdir(parents=True, exist_ok=True)

def load_model_auto():
    """Prefer calibrated model if exists."""
    if DEFAULT_CAL_MODEL.exists():
        print(f"[INFO] Loading calibrated model: {DEFAULT_CAL_MODEL}")
        return joblib.load(DEFAULT_CAL_MODEL), str(DEFAULT_CAL_MODEL)
    if DEFAULT_MODEL.exists():
        print(f"[INFO] Loading base model: {DEFAULT_MODEL}")
        return joblib.load(DEFAULT_MODEL), str(DEFAULT_MODEL)
    raise FileNotFoundError("No rf_tuned model found. Train / place rf_tuned.joblib in models/")

def classify_risk(prob: float, thresholds: Dict[str, float]) -> str:
    if prob < thresholds["moderate"]:
        return "Low"
    if prob < thresholds["high"]:
        return "Moderate"
    return "High"

def per_sample_parametric_ci(model, X_single: np.ndarray, n_bootstrap=200, alpha=0.05, random_state=42):
    """
    For each single sample add small Gaussian noise to approximate CI.
    X_single shape: (n_samples, n_features)
    Returns lower, upper arrays (length n_samples)
    """
    rng = np.random.RandomState(random_state)
    n_samples = X_single.shape[0]
    probs_boot = np.zeros((n_bootstrap, n_samples), dtype=float)

    # scale per feature from sample itself (conservative)
    scale = np.maximum(np.std(X_single, axis=0), 1e-6)

    for i in range(n_bootstrap):
        noise = rng.normal(loc=0.0, scale=0.01 * scale, size=X_single.shape)
        Xs = X_single + noise
        try:
            p = model.predict_proba(Xs)[:, 1]
        except Exception:
            dec = model.decision_function(Xs)
            p = (dec - dec.min())/(dec.max()-dec.min()+1e-8)
        probs_boot[i, :] = p

    lower = np.percentile(probs_boot, 100 * (alpha/2), axis=0)
    upper = np.percentile(probs_boot, 100 * (1 - alpha/2), axis=0)
    return lower, upper

def composite_risk_score(row: pd.Series, prob: float) -> float:
    """
    Combine model probability with a simple clinical adjustment:
    - If age present: add normalized age contribution (0..1) * 0.12
    - If resting_bp or systolic present: add scaled bp contribution * 0.08
    - If cholesterol present: add scaled chol contribution * 0.05
    Weighted so that model probability remains dominant.
    """
    score = float(prob) * 0.75  # base weight
    # age (0..1)
    if "age" in row.index and not pd.isna(row["age"]):
        age_norm = min(max((row["age"] - 30) / 50.0, 0.0), 1.0)  # rough 30->80 map
        score += age_norm * 0.12
    # bp
    if "resting_bp_s" in row.index and not pd.isna(row["resting_bp_s"]):
        bp = row["resting_bp_s"]
        bp_norm = min(max((bp - 110) / 60.0, 0.0), 1.0)  # 110->170
        score += bp_norm * 0.08
    elif "trestbps" in row.index and not pd.isna(row["trestbps"]):
        bp = row["trestbps"]
        bp_norm = min(max((bp - 110) / 60.0, 0.0), 1.0)
        score += bp_norm * 0.08
    # cholesterol
    if "cholesterol" in row.index and not pd.isna(row["cholesterol"]):
        chol = row["cholesterol"]
        chol_norm = min(max((chol - 150) / 200.0, 0.0), 1.0)  # 150->350
        score += chol_norm * 0.05
    return float(min(score, 1.0))

def shap_explanations(model, X, feature_names, top_k=5):
    """Return top_k SHAP contributions per sample (best-effort)."""
    try:
        import shap
        base = model
        try:
            if hasattr(model, "named_steps"):
                steps = model.named_steps
                if "clf" in steps:
                    base = steps["clf"]
                elif "model" in steps:
                    base = steps["model"]
                else:
                    base = list(steps.values())[-1]
        except Exception:
            base = model

        explainer = shap.TreeExplainer(base)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        arr = np.array(shap_vals)
        explanations = []
        for i in range(arr.shape[0]):
            v = arr[i]
            idx = np.argsort(np.abs(v))[::-1][:top_k]
            ex = [{"feature": feature_names[j], "shap_value": float(v[j])} for j in idx]
            explanations.append(ex)
        return explanations
    except Exception as e:
        warnings.warn(f"SHAP explanations not generated: {e}")
        return None

def predict_file(model_path: Path, input_csv: Path, output_json: Path,
                 th_low: float = 0.3, th_high: float = 0.7, n_boot: int = 200):
    df = pd.read_csv(input_csv)
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    model, model_loaded_path = load_model_auto() if model_path is None else (joblib.load(model_path), str(model_path))

    # ✅ Align columns with training features
    if hasattr(model, "feature_names_in_"):
        missing = [col for col in model.feature_names_in_ if col not in df.columns]
        if missing:
            raise ValueError(f"Input CSV is missing required features: {missing}")
        df = df[model.feature_names_in_]

    X = df.values

    # predict probabilities
    try:
        proba = model.predict_proba(df)[:, 1]  # use DataFrame to keep feature names
    except Exception as e:
        raise RuntimeError(f"Model predict_proba failed: {e}")

    # CI
    try:
        lower, upper = per_sample_parametric_ci(model, X, n_bootstrap=n_boot)
    except Exception:
        lower = np.clip(proba - 0.1, 0.0, 1.0)
        upper = np.clip(proba + 0.1, 0.0, 1.0)

    thresholds = {"low": 0.0, "moderate": th_low, "high": th_high}
    categories = [classify_risk(float(p), thresholds) for p in proba]

    comp_scores = [composite_risk_score(df.iloc[i], float(p)) for i, p in enumerate(proba)]

    explanations = shap_explanations(model, df, df.columns.tolist(), top_k=5)

    rows = []
    for i in range(len(proba)):
        rows.append({
            "index": int(i),
            "probability": float(proba[i]),
            "ci_lower": float(lower[i]),
            "ci_upper": float(upper[i]),
            "risk_category": categories[i],
            "composite_score": float(comp_scores[i]),
            "explanation": explanations[i] if explanations is not None else None
        })

    out = {
        "model_used": model_loaded_path if model_path is None else str(model_path),
        "n_input": len(proba),
        "thresholds": thresholds,
        "results": rows
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {output_json}")
    return out

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="path to model (.joblib). If not set will use rf_tuned_calibrated or rf_tuned")
    ap.add_argument("--input", type=str, required=True, help="CSV file with processed features")
    ap.add_argument("--out", type=str, default=str(OUTPUTS / "predictions.json"))
    ap.add_argument("--th_low", type=float, default=0.3)
    ap.add_argument("--th_high", type=float, default=0.7)
    ap.add_argument("--n_boot", type=int, default=200)
    args = ap.parse_args()
    res = predict_file(Path(args.model) if args.model else None, Path(args.input), Path(args.out),
                       th_low=args.th_low, th_high=args.th_high, n_boot=args.n_boot)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    cli()
