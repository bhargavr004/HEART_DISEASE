import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from utils import save_report, save_model
from pathlib import Path
import joblib

# Paths
ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_PATH = ROOT / "models" / "column_transformer.joblib"
FEATURES = ROOT / "data" / "processed" / "heart_features.csv"

def tune_rf(X, y):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10]
    }
    gs = GridSearchCV(rf, grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X, y)
    return gs

def tune_svm(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, random_state=42))
    ])
    dist = {
        "svc__C": np.logspace(-2, 2, 20),
        "svc__gamma": np.logspace(-3, 1, 20),
        "svc__kernel": ["rbf", "linear"]
    }
    rs = RandomizedSearchCV(pipe, dist, n_iter=40, cv=5, scoring="f1", n_jobs=-1, random_state=42)
    rs.fit(X, y)
    return rs

def tune_mlp(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=1000, early_stopping=True, random_state=42))
    ])
    grid = {
        "mlp__hidden_layer_sizes": [(64, 32), (64, 32, 16), (32, 16)],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [1e-3, 5e-4, 1e-4]
    }
    gs = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X, y)
    return gs

def apply_preprocessing(X):
    """Apply saved ColumnTransformer to ensure consistent feature engineering."""
    if TRANSFORMER_PATH.exists():
        transformer = joblib.load(TRANSFORMER_PATH)
        return transformer.transform(X)
    return X

def main():
    # Always load the full feature-engineered dataset
    df = pd.read_csv(FEATURES)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Apply transformation if transformer exists
    # X = apply_preprocessing(X)

    # Run tuning
    print("Tuning Random Forest...")
    rf_gs = tune_rf(X, y)

    print("Tuning SVM...")
    svm_rs = tune_svm(X, y)

    print("Tuning MLP...")
    mlp_gs = tune_mlp(X, y)

    results = {
        "rf_best_params": rf_gs.best_params_,
        "rf_best_f1": float(rf_gs.best_score_),
        "svm_best_params": svm_rs.best_params_,
        "svm_best_f1": float(svm_rs.best_score_),
        "mlp_best_params": mlp_gs.best_params_,
        "mlp_best_f1": float(mlp_gs.best_score_),
    }
    
    # Save results and models
    save_report(results, "tuning_results.json")
    save_model(rf_gs.best_estimator_, "rf_tuned.joblib")
    save_model(svm_rs.best_estimator_, "svm_tuned.joblib")
    save_model(mlp_gs.best_estimator_, "mlp_tuned.joblib")
    print(results)

if __name__ == "__main__":
    main()
