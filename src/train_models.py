from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_PATH = ROOT / "models" / "preprocessed" / "diabetes_preprocessed.joblib"
OUTPUT_DIR = ROOT / "models" / "trained"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

    joblib.dump(model, OUTPUT_DIR / f"{model_name}.joblib")
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
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        metrics = evaluate_model(model, model_name, X_train, X_test, y_train, y_test)
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

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Resultados dos modelos:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
