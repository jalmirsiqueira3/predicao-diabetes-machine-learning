import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from artifact_utils import DATA_PATH, PREPROCESSED_DIR, PREPROCESSED_PATH


def save_correlation_plot(df, output_path):
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", aspect="auto")
    fig.colorbar(im, ax=ax, label="Correlação")

    labels = numeric_df.columns.tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    ax.set_title("Matriz de Correlação")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)

    print("Dataset original:")
    print(df.head())
    print(f"\nFormato: {df.shape}")
    print(f"Linhas duplicadas: {df.duplicated().sum()}")
    print("\nValores ausentes antes do tratamento:")
    print(df.isnull().sum())

    df = df.drop_duplicates().copy()

    columns_with_zero_as_missing = [
        "Glicose",
        "PressaoArterial",
        "EspessuraPele",
        "Insulina",
        "IMC",
    ]
    for col in columns_with_zero_as_missing:
        df[col] = df[col].replace(0, np.nan)

    print("\nValores ausentes após substituir zeros inválidos por NaN:")
    print(df.isnull().sum())

    medians = df.median(numeric_only=True)
    df = df.fillna(medians)

    print("\nValores ausentes após imputação por mediana:")
    print(df.isnull().sum())

    X = df.drop(columns=["Resultado"])
    y = df["Resultado"]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    joblib.dump({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X.columns.tolist(),
        "scaler": scaler,
        "cleaned_df": df,
    }, PREPROCESSED_PATH)

    report = {
        "rows_original": int(df.shape[0] + 0),
        "rows_after_drop_duplicates": int(df.shape[0]),
        "target_distribution": y.value_counts().to_dict(),
        "features": X.columns.tolist(),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "scaler": "StandardScaler",
        "imputation": "median"
    }

    with open(PREPROCESSED_DIR / "preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    descriptive_summary = {
        "summary_statistics": df.describe().round(2).to_dict(),
        "mean_by_outcome": df.groupby("Resultado").mean().round(2).to_dict(),
    }

    with open(PREPROCESSED_DIR / "descriptive_summary.json", "w", encoding="utf-8") as f:
        json.dump(descriptive_summary, f, indent=2, ensure_ascii=False)

    save_correlation_plot(df, PREPROCESSED_DIR / "correlation_matrix.png")

    report["descriptive_summary"] = descriptive_summary
    with open(PREPROCESSED_DIR / "preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nResumo descritivo:")
    print(json.dumps(descriptive_summary, indent=2, ensure_ascii=False))
    print("\nResumo da etapa de pré-processamento:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nArquivos salvos em: {PREPROCESSED_DIR}")


if __name__ == "__main__":
    main()
