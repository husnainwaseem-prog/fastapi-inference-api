"""Project-wide settings for the inference API."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

APP_TITLE = "Text Classifier Inference & Embeddings API"
APP_VERSION = "1.0.0"
MODEL_VERSION = "smart-text-classifier v1 (TF-IDF + LogisticRegression)"

# Input limits
MAX_TEXT_LENGTH = 2000
MAX_BATCH_SIZE = 50

# Latency budget from the task brief: predictions must stay under 200 ms.
LATENCY_BUDGET_MS = 200.0
