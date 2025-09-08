import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from utils import load_splits, summarize_metrics, save_report, save_model, plot_roc_pr, OUTPUTS, MODELS
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
MODELS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

def train_xgb(X_train, y_train, X_val, y_val):
    print("Training XGBoost with early stopping using native API...")
    import xgboost as xgb_lib
    dtrain = xgb_lib.DMatrix(X_train, label=y_train)
    dval = xgb_lib.DMatrix(X_val, label=y_val)
    params = {
        'objective': 'binary:logistic',
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0
    }
    evals = [(dtrain, 'train'), (dval, 'eval')]
    booster = xgb_lib.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False
    )
    print("XGBoost trained. Best iteration:", booster.best_iteration)
    # Wrap into sklearn API for downstream use
    xgb_clf = XGBClassifier(
        n_estimators=booster.best_iteration+1,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print("XGBoost classifier ready for downstream use.")
    return xgb_clf

def smote_resample(X_train, y_train):
    print("Applying SMOTE to training set...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print("Resampled:", X_train.shape, "->", X_res.shape)
    return X_res, y_res

def find_best_threshold(model, X_val, y_val):
    """Find threshold maximizing F1 on validation set."""
    probs = model.predict_proba(X_val)[:,1]
    thresholds = np.linspace(0.1, 0.9, 81)
    best = {"th":0.5, "f1": -1, "acc":0}
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f = f1_score(y_val, preds)
        a = accuracy_score(y_val, preds)
        if f > best["f1"]:
            best.update({"th":t, "f1": f, "acc": a})
    print("Best threshold on val ->", best)
    return best

def evaluate_and_save(model, name, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:,1]
    preds = (proba >= threshold).astype(int)
    metrics = summarize_metrics(y_test, preds, proba)
    save_report({"model": name, "threshold": threshold, "metrics": metrics}, f"{name}_advanced_eval.json")
    plot_roc_pr(y_test, proba, f"{name}_advanced")
    save_model(model, f"{name}_advanced.joblib")
    print(f"[EVAL] {name}: threshold={threshold}, metrics={metrics}")
    return metrics, proba, preds

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # 1) SMOTE on train
    X_res, y_res = smote_resample(X_train, y_train)

    # 2) Train XGBoost on resampled data
    xgb_model = train_xgb(X_res, y_res, X_val, y_val)
    save_model(xgb_model, "xgb_smote.joblib")

    # 3) Evaluate XGB with best threshold
    th_res = find_best_threshold(xgb_model, X_val, y_val)
    evaluate_and_save(xgb_model, "xgb_smote", X_test, y_test, threshold=th_res["th"])

    print("Only XGBoost pipeline completed. Results saved to outputs/.")

if __name__ == "__main__":
    main()
