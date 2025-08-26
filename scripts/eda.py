import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "heart_raw.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def load():
    df = pd.read_csv(RAW)
    return df

def basic_statistics(df):
    stats = df.describe(include='all')
    stats.to_csv(OUT / "dataset_summary_statistics.csv")
    # target distribution
    df['target'].value_counts().to_csv(OUT / "target_distribution.csv")

def plots(df):
    plt.figure(figsize=(8,5))
    sns.countplot(x="target", data=df)
    plt.title("Target distribution (0=no disease,1=disease)")
    plt.savefig(FIG / "target_distribution.png", bbox_inches='tight')
    plt.close()

    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram: {col}")
        plt.savefig(FIG / f"hist_{col}.png", bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.savefig(FIG / "correlation_matrix.png", bbox_inches='tight')
    plt.close()

def profile_report(df):
    profile = ProfileReport(df, title="Heart Disease EDA", explorative=True)
    profile.to_file(OUT / "eda_profile_report.html")

def main():
    df = load()
    basic_statistics(df)
    plots(df)
    profile_report(df)
    print("EDA complete. Outputs in", OUT)

if __name__ == "__main__":
    main()
