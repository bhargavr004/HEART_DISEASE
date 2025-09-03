from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import load_splits, summarize_metrics, save_report, save_model

def train_eval_svm(kernel="rbf"):
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    svc = SVC(kernel=kernel, probability=True, C=1.0, gamma="scale", random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", svc)])
    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    y_val_proba = pipe.predict_proba(X_val)[:,1]
    metrics = summarize_metrics(y_val, y_val_pred, y_val_proba)
    tag = f"svm_{kernel}"
    save_report({"model":tag,"val_metrics":metrics}, f"{tag}_val.json")
    save_model(pipe, f"{tag}.joblib")
    print(f"[{tag}] VAL:", metrics)

def main():
    train_eval_svm("linear")
    train_eval_svm("rbf")

if __name__ == "__main__":
    main()
