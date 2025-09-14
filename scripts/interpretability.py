import shap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import load
from pathlib import Path
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
FEATURES_FPATH = ROOT / "data" / "processed" / "heart_features.csv"

def main():
    print("=== INTERPRETABILITY ANALYSIS START ===")

    # Load processed features if available
    if FEATURES_FPATH.exists():
        print(f"[INFO] Loading processed features from {FEATURES_FPATH}")
        df = pd.read_csv(FEATURES_FPATH)
        if "target" not in df.columns:
            raise RuntimeError("heart_features.csv must contain a 'target' column")
        X_all = df.drop("target", axis=1)
        y_all = df["target"]

        # Use last 15% as test set (or consistent split logic from advanced pipeline)
        test_size = int(len(X_all) * 0.15)
        X_test_df = X_all.tail(test_size)
        y_test = y_all.tail(test_size)
        feature_names = X_all.columns.tolist()
    else:
        print("[WARN] heart_features.csv not found — falling back to raw splits")
        from utils import load_splits
        X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
        X_test_df = X_test
        feature_names = X_test.columns.tolist()

    # Load best RF model
    model_path = MODELS / "rf.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    rf = load(model_path)

    # ---- Feature Importances ----
    print("Calculating feature importances...")
    fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    fi.to_csv(OUTPUTS / "rf_feature_importances.csv")
    print("[SAVED] rf_feature_importances.csv")

    plt.figure(figsize=(8, 6))
    fi.head(15).plot(kind="barh")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "rf_top15_features.png", dpi=150)
    plt.close()
    print("[SAVED] rf_top15_features.png")

       # ---- SHAP Analysis ----
    print("Running SHAP analysis...")
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_test_df)

    # Handle multiclass/binary case
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # For binary classification, class 1

    # If shap_vals is 3D (e.g., shape (n_samples, n_features, 2)), reduce to 2D
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]  # take class 1

    # SHAP summary plot
    shap.summary_plot(shap_vals, X_test_df, feature_names=feature_names, max_display=20, show=False)
    plt.savefig(OUTPUTS / "shap_summary.png", dpi=150)
    plt.close()
    print("[SAVED] shap_summary.png")

    # Compute mean absolute SHAP values for ranking
    shap_abs_mean = np.mean(np.abs(shap_vals), axis=0)  # now it's 1D
    feat_imp = pd.Series(shap_abs_mean, index=feature_names).sort_values(ascending=False)
    feat_imp.to_csv(OUTPUTS / "shap_feature_importances.csv")
    print("[SAVED] shap_feature_importances.csv")

    # # Handle multiclass/binary case
    # if isinstance(shap_vals, list):
    #     shap_vals = shap_vals[1]

    # Bar chart of top 20 SHAP features
    plt.figure(figsize=(8, 6))
    feat_imp.head(20).plot(kind="barh")
    plt.title("Top 20 Features by SHAP Importance")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "shap_top20_features.png", dpi=150)
    plt.close()
    print("[SAVED] shap_top20_features.png")

    print("=== INTERPRETABILITY ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()
