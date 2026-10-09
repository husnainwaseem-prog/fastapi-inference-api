"""Model loading, prediction and embedding logic.

The trained artefact is a scikit-learn Pipeline (TF-IDF -> LogisticRegression)
saved as ``models/model.joblib``. It is loaded exactly once (cached) and then
reused by every request, which is what keeps per-request latency low.
"""

import time
from functools import lru_cache

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from .config import MODEL_PATH


class ModelNotLoadedError(RuntimeError):
    """Raised when the model file is missing or unreadable."""


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    """Load the trained pipeline once and reuse it for all requests."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError as exc:
        raise ModelNotLoadedError(
            f"Model file not found at {MODEL_PATH}. "
            "Make sure models/model.joblib is present."
        ) from exc
    except Exception as exc:  # corrupted file, version mismatch, ...
        raise ModelNotLoadedError(f"Could not load the model: {exc}") from exc


def _vectorizer(model: Pipeline):
    return model.named_steps["tfidf"]


def predict_one(text: str, model: Pipeline | None = None) -> dict:
    """Classify a single text. Returns label, confidence, probabilities and latency."""
    model = model or load_model()
    start = time.perf_counter()
    probabilities = model.predict_proba([text])[0]
    best = int(probabilities.argmax())
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "label": str(model.classes_[best]),
        "confidence": round(float(probabilities[best]), 4),
        "probabilities": {
            str(cls): round(float(p), 4) for cls, p in zip(model.classes_, probabilities)
        },
        "latency_ms": round(latency_ms, 2),
    }


def predict_many(texts: list[str], model: Pipeline | None = None) -> tuple[list[dict], float]:
    """Classify a batch of texts in one vectorised call. Returns (results, latency_ms)."""
    model = model or load_model()
    start = time.perf_counter()
    all_probabilities = model.predict_proba(texts)
    latency_ms = (time.perf_counter() - start) * 1000
    results = []
    for probabilities in all_probabilities:
        best = int(probabilities.argmax())
        results.append(
            {
                "label": str(model.classes_[best]),
                "confidence": round(float(probabilities[best]), 4),
                "probabilities": {
                    str(cls): round(float(p), 4)
                    for cls, p in zip(model.classes_, probabilities)
                },
                "latency_ms": round(latency_ms / max(len(texts), 1), 2),
            }
        )
    return results, round(latency_ms, 2)


def embed_text(text: str, model: Pipeline | None = None) -> np.ndarray:
    """Return the text as an L2-normalised TF-IDF embedding vector.

    The embedding is the TF-IDF vector produced by the same vectorizer the
    classifier was trained with, normalised to unit length so that a dot
    product between two embeddings equals their cosine similarity.
    """
    model = model or load_model()
    vector = _vectorizer(model).transform([text]).toarray()[0]
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def cosine_similarity(text_a: str, text_b: str, model: Pipeline | None = None) -> float:
    """Cosine similarity between two texts, in [0, 1] for TF-IDF vectors."""
    model = model or load_model()
    return float(np.dot(embed_text(text_a, model), embed_text(text_b, model)))
