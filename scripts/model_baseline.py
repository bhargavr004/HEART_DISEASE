from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from utils import load_splits, summarize_metrics, save_report, save_model, plot_roc_pr

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # Baseline model pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, solver="liblinear", penalty="l2"))
    ])

    pipe.fit(X_train, y_train)

    # Validation predictions
    y_val_pred = pipe.predict(X_val)
    y_val_proba = pipe.predict_proba(X_val)[:, 1]

    val_metrics = summarize_metrics(y_val, y_val_pred, y_val_proba)
    save_report({"model": "logreg_baseline", "val_metrics": val_metrics}, "baseline_logreg_val.json")

    # Test evaluation
    y_test_pred = pipe.predict(X_test)
    y_test_proba = pipe.predict_proba(X_test)[:, 1]
    test_metrics = summarize_metrics(y_test, y_test_pred, y_test_proba)
    save_report({"model": "logreg_baseline", "test_metrics": test_metrics}, "baseline_logreg_test.json")

    save_model(pipe, "baseline_logreg.joblib")

    # Generate ROC & PR plots
    plot_roc_pr(y_test, y_test_proba, "logreg_baseline")

    # Print confusion matrix and summary
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print("\n=== Baseline Logistic Regression ===")
    print(f"Validation Metrics: {val_metrics}")
    print(f"Test Metrics: {test_metrics}")
    print(f"Confusion Matrix (Test): TN={tn}, FP={fp}, FN={fn}, TP={tp}")

if __name__ == "__main__":
    main()
