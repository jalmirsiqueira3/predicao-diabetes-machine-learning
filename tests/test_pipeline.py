import json
from pathlib import Path

from src.artifact_utils import PREPROCESSED_PATH, ROOT, TRAINED_DIR


def test_expected_artifacts_exist():
    assert ROOT.exists()
    assert PREPROCESSED_PATH.exists(), "O arquivo pré-processado deveria existir após executar o pipeline"
    assert TRAINED_DIR.exists()


def test_model_metrics_are_available():
    metrics_path = TRAINED_DIR / "model_metrics.json"
    assert metrics_path.exists(), "As métricas dos modelos deveriam ter sido salvas"

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    assert metrics, "A lista de métricas não deveria estar vazia"
    assert all({"model", "accuracy", "precision", "recall", "f1"}.issubset(metric.keys()) for metric in metrics)
