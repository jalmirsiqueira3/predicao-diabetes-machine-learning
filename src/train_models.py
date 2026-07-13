import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from artifact_utils import METRICS_PATH, PREPROCESSED_PATH, REPORT_PATH, TRAINED_DIR


def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Matriz de confusão - {model_name}")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Não diabetes", "Diabetes"])
    ax.set_yticklabels(["Não diabetes", "Diabetes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    fig.savefig(TRAINED_DIR / f"{model_name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }

    joblib.dump(model, TRAINED_DIR / f"{model_name}.joblib")
    save_confusion_matrix(y_test, predictions, model_name)
    return metrics


def main():
    data = joblib.load(PREPROCESSED_PATH)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    results = []

    for depth in [3, 5, 7]:
        model_name = f"decision_tree_depth_{depth}"
        metrics = evaluate_model(
            DecisionTreeClassifier(max_depth=depth, random_state=42),
            model_name,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        results.append(metrics)

    nb_metrics = evaluate_model(
        GaussianNB(),
        "naive_bayes_gaussian",
        X_train,
        X_test,
        y_train,
        y_test,
    )
    results.append(nb_metrics)

    comparison_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
    comparison_df.to_csv(TRAINED_DIR / "model_comparison.csv", index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    best_model = comparison_df.iloc[0]
    report_lines = [
        "# Relatório de Modelos",
        "",
        "| Modelo | Acurácia | Precisão | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison_df.iterrows():
        report_lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )

    report_lines.extend([
        "",
        f"Modelo destaque: {best_model['model']} com F1-score de {best_model['f1']:.4f}.",
        "",
        "O modelo com melhor desempenho foi escolhido com base no F1-score, pois equilibra precisão e recall para o problema de classificação.",
    ])
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("Resultados dos modelos:")
    print(comparison_df.to_string(index=False))
    print(f"\nRelatório salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    main()
