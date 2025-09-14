# scripts/evaluation.py
import json
from pathlib import Path
from joblib import load
import numpy as np
import pandas as pd
from utils import load_splits, summarize_metrics, save_report, plot_roc_pr, OUTPUTS
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import ttest_rel
import matplotlib

# Force non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

def load_model_any(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in [".h5", ".keras"]:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(path))
        return ("keras", model)
    else:
        m = load(str(path))
        return ("sk", m)

def _get_last_estimator_if_pipeline(model):
    # If it's a sklearn Pipeline, try to use the last step for feature_names_in_
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            if hasattr(model, "named_steps") and model.named_steps:
                return list(model.named_steps.values())[-1]
    except Exception:
        pass
    return model

def align_features_to_model(X_df: pd.DataFrame, model):
    """
    If the model (or its last step) has feature_names_in_, align X_df to match:
      - add missing columns filled with 0.0
      - drop extra columns
      - order columns identically
    If not available, fall back to raw numpy values.
    """
    if not isinstance(X_df, pd.DataFrame):
        return X_df  # already ndarray

    candidate_objs = [model, _get_last_estimator_if_pipeline(model)]
    expected = None
    for obj in candidate_objs:
        cols = getattr(obj, "feature_names_in_", None)
        if cols is not None:
            expected = list(cols)
            break

    if expected is None:
        # No feature metadata -> use raw values to avoid name checks downstream
        return X_df.values

    X = X_df.copy()
    # add missing as zeros
    for c in expected:
        if c not in X.columns:
            X[c] = 0.0
    # drop extras
    X = X[expected]
    return X

def predict_model(kind_model, X_df):
    kind, model = kind_model

    # Align features for sklearn models
    if kind == "sk":
        X = align_features_to_model(X_df, model)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            dec = model.decision_function(X)
            proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-8)
        else:
            proba = model.predict(X)
        pred = (proba >= 0.5).astype(int)
        return pred, proba

    # Keras model: best effort — use raw values, shapes must match
    X = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
    proba = kind_model[1].predict(X).ravel()
    pred = (proba >= 0.5).astype(int)
    return pred, proba

def eval_one(model_path, tag, X_test, y_test):
    try:
        kind_model = load_model_any(model_path)
    except Exception as e:
        print(f"Skipping {tag}, load failed: {e}")
        return None

    try:
        pred, proba = predict_model(kind_model, X_test)
    except Exception as e:
        print(f"Error evaluating {tag} {e}")
        return None

    metrics = summarize_metrics(y_test, pred, proba)
    save_report({"model": tag, "test_metrics": metrics}, f"test_{tag}.json")
    plot_roc_pr(y_test, proba, tag)
    plt.close("all")
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"{tag} Confusion: TN {tn} FP {fp} FN {fn} TP {tp}")
    return {"tag": tag, "pred": pred, "proba": proba, "metrics": metrics}

def statistical_tests(results, y_test):
    tests = {}
    n = len(results)
    for i in range(n):
        for j in range(i + 1, n):
            a = results[i]; b = results[j]
            tag = f"{a['tag']}__vs__{b['tag']}"
            a_corr = (a["pred"] == y_test.values)
            b_corr = (b["pred"] == y_test.values)
            both = np.sum(a_corr & b_corr)
            a_only = np.sum(a_corr & (~b_corr))
            b_only = np.sum((~a_corr) & b_corr)
            neither = np.sum((~a_corr) & (~b_corr))
            try:
                res = mcnemar(table=[[a_only, b_only], [b_only, neither]], exact=False)
            except Exception:
                res = mcnemar(table=[[both, a_only], [b_only, neither]])
            try:
                tstat, pval = ttest_rel(a["proba"], b["proba"])
            except Exception:
                tstat, pval = float("nan"), float("nan")
            tests[tag] = {
                "mcnemar_statistic": float(getattr(res, "statistic", float("nan"))),
                "mcnemar_pvalue": float(getattr(res, "pvalue", float("nan"))),
                "paired_t_stat": float(tstat),
                "paired_t_pvalue": float(pval),
                "a_only": int(a_only),
                "b_only": int(b_only),
            }
    save_report(tests, "statistical_tests.json")
    return tests

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # IMPORTANT: we no longer apply ColumnTransformer here.
    # All models are expected to consume the same engineered feature schema as the splits.
    candidate_paths = [
        (MODELS / "rf.joblib", "rf"),
        (MODELS / "rf_tuned.joblib", "rf_tuned"),
        (MODELS / "svm_rbf.joblib", "svm_rbf"),
        (MODELS / "svm_tuned.joblib", "svm_tuned"),
        (MODELS / "mlp.joblib", "mlp"),
        (MODELS / "mlp_tuned.joblib", "mlp_tuned"),
        (MODELS / "keras_mlp.h5", "keras_mlp"),
    ]

    results = []
    for p, tag in candidate_paths:
        out = eval_one(str(p), tag, X_test, y_test)
        if out:
            results.append(out)

    if len(results) >= 2:
        statistical_tests(results, y_test)

    # Ranking
    if results:
        df = pd.DataFrame([{**r["metrics"], "model": r["tag"]} for r in results])
        df_sorted = df.sort_values(by="accuracy", ascending=False)
        print("\n=== Model Performance Ranking (by Accuracy) ===")
        print(df_sorted[["model", "accuracy", "f1", "roc_auc"]])
        best = df_sorted.iloc[0]
        print(f"\nBest Model: {best['model']} | Accuracy: {best['accuracy']:.4f} | F1: {best['f1']:.4f}")
        if best["accuracy"] < 0.85:
            print("⚠ Accuracy below 85%. Consider further tuning or advanced ensemble.")
        report_md = df_sorted.to_markdown(index=False)
        with open(OUTPUTS / "evaluation_report.md", "w", encoding="utf-8") as f:
            f.write("# Model Evaluation Summary\n\n")
            f.write(report_md)
        print(f"[REPORT] {OUTPUTS / 'evaluation_report.md'}")
    else:
        print("No models evaluated successfully.")

if __name__ == "__main__":
    main()
