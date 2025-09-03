import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from utils import load_splits, save_report

def main(k=5):
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    X = np.concatenate([X_train.values, X_val.values], axis=0)
    y = np.concatenate([y_train.values, y_val.values], axis=0)

    models = {
        "logreg": Pipeline([("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=500, solver="liblinear"))]),
        "rf": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        "svm_rbf": Pipeline([("scaler", StandardScaler()),
                             ("svc", SVC(kernel="rbf", probability=True, random_state=42))]),
        "mlp": Pipeline([("scaler", StandardScaler()),
                         ("mlp", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42))])
    }

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
        results[name] = {"cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std())}
        print(name, results[name])

    save_report({"k":k, "cv_results":results}, "cv_results.json")

if __name__ == "__main__":
    main(k=5)
