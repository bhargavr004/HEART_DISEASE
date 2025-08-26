import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / "data" / "processed" / "heart_cleaned.csv"
OUT = ROOT / "data" / "processed" / "heart_features.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def load():
    return pd.read_csv(CLEANED)

def derive_features(df):
    # BMI example: if dataset doesn't have weight/height this is illustrative.
    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[0,35,50,65,120], labels=['young','mid','senior','old'])
    # chest pain type as categorical - if numeric codes, keep as category
    df['cp'] = df['cp'].astype('category')
    # risk score (simple linear combination as example)
    df['risk_score_simple'] = (df['age'] / df['age'].max()) + df['trestbps']/df['trestbps'].max() + df['chol']/df['chol'].max()
    return df

def transform(df):
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    numeric_features = [c for c in numeric_features if c != 'target']
    categorical_features = df.select_dtypes(include=['category','object']).columns.tolist()

    ct = ColumnTransformer([
        ('scale', StandardScaler(), numeric_features),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ], remainder='drop')

    X = ct.fit_transform(df[numeric_features + categorical_features])
    # create feature names
    ohe_cols = []
    if categorical_features:
        ohe = ct.named_transformers_['onehot']
        ohe_cols = ct.named_transformers_['onehot'].get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_features + ohe_cols
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['target'] = df['target'].values
    return X_df, ct

def feature_importance(X_df):
    X = X_df.drop('target', axis=1)
    y = X_df['target']
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.to_csv(ROOT / "outputs" / "feature_importances.csv")
    return imp

def main():
    df = load()
    df = derive_features(df)
    X_df, ct = transform(df)
    X_df.to_csv(OUT, index=False)
    imp = feature_importance(X_df)
    print("Features saved to", OUT)
    print("Top features:\n", imp.head(15))

if __name__ == "__main__":
    main()
