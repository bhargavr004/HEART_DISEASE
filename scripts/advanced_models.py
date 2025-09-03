"""
Advanced modeling:
- SMOTE on training data
- Train XGBoost with early stopping
- Build StackingClassifier (RF, SVM, XGB) with tuned meta-learner
- Tune VotingClassifier weights (quick grid search)
- Optimize decision threshold on validation set for F1
- Save best models and results

Run: python scripts/advanced_models.py
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from utils import load_splits, summarize_metrics, save_report, save_model, plot_roc_pr, OUTPUTS, MODELS
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

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
    # Create a new XGBClassifier with best n_estimators and fit to set attributes
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

def build_stack(X_train, y_train):
    print("Building stacking classifier (RF, SVM, XGB) with LR meta-learner...")
    # Base estimators
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ("svm", Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", probability=True, random_state=42))])),
        ("xgb", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric="logloss", verbosity=0))
    ]
    # meta-learner candidates: LR or shallow XGB; we'll GridSearch meta-learner C for LR
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500),
        passthrough=False,
        n_jobs=-1
    )
    # quick grid for final_estimator hyperparam (search C)
    param_grid = {
        "final_estimator__C": [0.1, 1.0, 10.0]
    }
    gs = GridSearchCV(stack, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    print("Stacking best params:", gs.best_params_, "best score:", gs.best_score_)
    return gs.best_estimator_, gs.best_score_

def tune_voting_weights(X_train, y_train, X_val, y_val):
    print("Tuning voting classifier weights (grid search)")
    # Create base learners trained on training set (no SMOTE here, but you can try)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    svm = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", probability=True, random_state=42))])
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric="logloss", verbosity=0)

    # Fit base models on training set
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    models = [("rf", rf), ("svm", svm), ("xgb", xgb)]

    # Grid of weight triples (coarse)
    weight_grid = []
    weights = [0.5, 1.0, 1.5]
    for a in weights:
        for b in weights:
            for c in weights:
                weight_grid.append((a,b,c))

    best = {"score": -1, "weights": None, "clf": None}
    for w in weight_grid:
        vc = VotingClassifier(estimators=models, voting="soft", weights=list(w), n_jobs=-1)
        vc.fit(X_train, y_train)  # they are already fitted, but sklearn requires fit to enable predict_proba pipeline
        preds = vc.predict(X_val)
        f1 = f1_score(y_val, preds)
        if f1 > best["score"]:
            best = {"score": f1, "weights": w, "clf": vc}
    print("Best voting weights:", best["weights"], "val F1:", best["score"])
    return best["clf"], best["weights"], best["score"]

def find_best_threshold(model, X_val, y_val):
    """Find threshold maximizing F1 on validation set."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:,1]
    else:
        # fallback to decision_function scaling
        dec = model.decision_function(X_val)
        probs = (dec - dec.min()) / (dec.max() - dec.min() + 1e-8)
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
    # Get probabilities and predictions
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:,1]
    else:
        dec = model.decision_function(X_test)
        proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-8)
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

    # Evaluate XGB with default 0.5 and find best threshold
    th_res = find_best_threshold(xgb_model, X_val, y_val)
    evaluate_and_save(xgb_model, "xgb_smote", X_test, y_test, threshold=th_res["th"])

    # 3) Build stacking (trained on resampled)
    stack_model, stack_score = build_stack(X_res, y_res)
    # find threshold for stacking
    th_stack = find_best_threshold(stack_model, X_val, y_val)
    evaluate_and_save(stack_model, "stacking_advanced", X_test, y_test, threshold=th_stack["th"])

    # 4) Tune voting weights (use resampled training)
    voting_clf, w, wscore = tune_voting_weights(X_res, y_res, X_val, y_val)
    th_vote = find_best_threshold(voting_clf, X_val, y_val)
    evaluate_and_save(voting_clf, "voting_advanced", X_test, y_test, threshold=th_vote["th"])

    # 5) Compare results and print best
    candidates = ["xgb_smote", "stacking_advanced", "voting_advanced"]
    summary = {}
    for c in candidates:
        path = OUTPUTS / f"{c}_advanced_eval.json"
        if path.exists():
            summary[c] = pd.read_json(path).to_dict()
    print("Advanced candidates evaluation saved to outputs/*.json")
    # Save combined summary
    save_report(summary, "advanced_models_summary.json")

if __name__ == "__main__":
    main()
