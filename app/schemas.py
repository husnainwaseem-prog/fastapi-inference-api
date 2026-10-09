"""Pydantic request/response schemas for the API."""

from typing import Annotated

from pydantic import BaseModel, Field

from .config import MAX_BATCH_SIZE, MAX_TEXT_LENGTH


class PredictRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        examples=["My order arrived broken and nobody replied."],
    )


class PredictionResult(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


class BatchPredictRequest(BaseModel):
    texts: list[Annotated[str, Field(min_length=1, max_length=MAX_TEXT_LENGTH)]] = Field(
        min_length=1, max_length=MAX_BATCH_SIZE
    )


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResult]
    count: int
    latency_ms: float


class EmbedRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class EmbedResponse(BaseModel):
    embedding: list[float]
    dim: int


class SimilarityRequest(BaseModel):
    text_a: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    text_b: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class SimilarityResponse(BaseModel):
    similarity: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class InfoResponse(BaseModel):
    model_version: str
    classes: list[str]
    embedding_dim: int
    vectorizer: str
    classifier: str
