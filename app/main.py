"""FastAPI application: async text-classifier inference & embeddings API.

Design notes:
- The model is loaded once at startup (lifespan) and shared by all requests.
- Endpoints are ``async``; the (CPU-bound, few-millisecond) scikit-learn calls
  run in FastAPI's threadpool via ``run_in_threadpool`` so the event loop is
  never blocked and concurrent requests stay fast.
- Every prediction response carries its own measured ``latency_ms``.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from . import config
from . import model as model_service
from .model import ModelNotLoadedError
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    InfoResponse,
    PredictRequest,
    PredictionResult,
    SimilarityRequest,
    SimilarityResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts."""
    try:
        app.state.model = model_service.load_model()
    except ModelNotLoadedError as exc:
        # Start anyway so /health can report the problem instead of crashing.
        app.state.model = None
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description=(
        "Real-time predictions from a TF-IDF + LogisticRegression text classifier, "
        "plus TF-IDF embedding vectors and cosine-similarity scoring."
    ),
    lifespan=lifespan,
)


def get_model():
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return model


@app.get("/", tags=["meta"])
async def root():
    return {
        "message": "Text Classifier Inference & Embeddings API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "model_version": config.MODEL_VERSION,
    }


@app.get("/info", response_model=InfoResponse, tags=["meta"])
async def info():
    model = get_model()
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    return {
        "model_version": config.MODEL_VERSION,
        "classes": [str(c) for c in model.classes_],
        "embedding_dim": len(vectorizer.vocabulary_),
        "vectorizer": f"TF-IDF ({vectorizer.ngram_range[0]}-{vectorizer.ngram_range[1]}-grams)",
        "classifier": type(classifier).__name__,
    }


@app.post("/predict", response_model=PredictionResult, tags=["inference"])
async def predict(request: PredictRequest):
    """Classify a single text. Typical latency: a few milliseconds."""
    return await run_in_threadpool(model_service.predict_one, request.text, get_model())


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
async def predict_batch(request: BatchPredictRequest):
    """Classify up to 50 texts in one vectorised call."""
    results, latency_ms = await run_in_threadpool(
        model_service.predict_many, request.texts, get_model()
    )
    return {"predictions": results, "count": len(results), "latency_ms": latency_ms}


@app.post("/embed", response_model=EmbedResponse, tags=["embeddings"])
async def embed(request: EmbedRequest):
    """Return the text as an L2-normalised TF-IDF embedding vector."""
    vector = await run_in_threadpool(model_service.embed_text, request.text, get_model())
    return {"embedding": [round(float(x), 6) for x in vector], "dim": len(vector)}


@app.post("/similarity", response_model=SimilarityResponse, tags=["embeddings"])
async def similarity(request: SimilarityRequest):
    """Cosine similarity between two texts, from 0 (unrelated) to 1 (identical)."""
    score = await run_in_threadpool(
        model_service.cosine_similarity, request.text_a, request.text_b, get_model()
    )
    return {"similarity": round(score, 4)}
