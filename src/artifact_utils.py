from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "diabetes.csv"
MODELS_DIR = ROOT / "models"
PREPROCESSED_DIR = MODELS_DIR / "preprocessed"
TRAINED_DIR = MODELS_DIR / "trained"
PREPROCESSED_PATH = PREPROCESSED_DIR / "diabetes_preprocessed.joblib"
METRICS_PATH = TRAINED_DIR / "model_metrics.json"
REPORT_PATH = TRAINED_DIR / "model_report.md"

PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_DIR.mkdir(parents=True, exist_ok=True)


def load_preprocessed_artifacts():
    return joblib.load(PREPROCESSED_PATH)
