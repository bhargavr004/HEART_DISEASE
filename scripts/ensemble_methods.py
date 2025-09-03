from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import load_splits, summarize_metrics, save_report, save_model

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    logreg = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=200, solver="liblinear"))])

    svm_rbf = Pipeline([("scaler", StandardScaler()),
                        ("svc", SVC(kernel="rbf", probability=True, random_state=42))])

    rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)

    # Voting (soft)
    voting = VotingClassifier(estimators=[("lr", logreg), ("svm", svm_rbf), ("rf", rf)],
                              voting="soft", n_jobs=-1)
    voting.fit(X_train, y_train)
    yv = voting.predict(X_val); pv = voting.predict_proba(X_val)[:,1]
    m_v = summarize_metrics(y_val, yv, pv)

    # Bagging (LR base)
    bag = BaggingClassifier(estimator=LogisticRegression(max_iter=200, solver="liblinear"),
                            n_estimators=25, random_state=42, n_jobs=-1)
    bag.fit(X_train, y_train)
    yb = bag.predict(X_val); pb = bag.predict_proba(X_val)[:,1]
    m_b = summarize_metrics(y_val, yb, pb)

    # Stacking (meta LR)
    stack = StackingClassifier(estimators=[("svm", svm_rbf), ("rf", rf)],
                               final_estimator=LogisticRegression(max_iter=200, solver="liblinear"))
    stack.fit(X_train, y_train)
    ys = stack.predict(X_val); ps = stack.predict_proba(X_val)[:,1]
    m_s = summarize_metrics(y_val, ys, ps)

    out = {"voting_soft": m_v, "bagging_lr": m_b, "stacking": m_s}
    save_report(out, "ensembles_val.json")
    save_model(voting, "voting_soft.joblib")
    save_model(stack, "stacking.joblib")
    print(out)

if __name__ == "__main__":
    main()
