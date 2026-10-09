# Project Report — FastAPI Model Inference & Embeddings API

## Approach

The task was to deploy an asynchronous FastAPI service exposing real-time model
predictions with sub-200 ms latency, plus an embeddings endpoint. Instead of
training a new model, I reused a proven artefact, which I acknowledge here: the TF-IDF +
LogisticRegression text-classification pipeline (4 classes: Complaint, Inquiry,
Feedback, Other) from my earlier Smart Text Classifier project, saved as
`models/model.joblib`. Its training script, dataset and metrics are included
(`src/train.py`, `data/dataset.csv`, `models/metrics.json`) so the model can be
reproduced with `python -m src.train`. This kept the focus of this task where it belongs — on
API design, async serving, validation, and latency — rather than on
re-training.

The service exposes six endpoints. `POST /predict` and `POST /predict/batch`
serve classifications; `POST /embed` returns the text as an
L2-normalised TF-IDF vector; `POST /similarity` scores the cosine similarity of two
texts; `GET /health` and `GET /info` provide liveness checks and model
metadata. Every prediction response carries its own measured `latency_ms`, so
the latency budget is observable, not just claimed.

## Key decisions

1. **Load the model once at startup.** The pipeline is loaded in the app's
   lifespan handler and shared by all requests. Reloading per request would
   have added tens of milliseconds of pure overhead.
2. **Async endpoints with a threadpool for inference.** The routes are
   `async def`, and the short CPU-bound scikit-learn calls run via
   `run_in_threadpool`, so the event loop never blocks and concurrent
   requests don't queue behind each other.
3. **TF-IDF vectors as embeddings.** The embedding is the classifier's own
   TF-IDF vector, normalised to unit length, which makes a dot product equal
   cosine similarity. This keeps the project dependency-light (no 500 MB+
   transformer download) and guarantees the embeddings stay consistent with
   what the classifier actually sees.
4. **Strict input validation.** Pydantic schemas reject empty text, text over
   2000 characters, and batches over 50 items with HTTP 422, so bad input
   fails fast with a clear message instead of producing garbage predictions.
5. **Tests prove the latency budget.** Beyond correctness tests, the suite
   asserts every prediction over 15 sequential requests stays under 200 ms,
   and `scripts/benchmark.py` reports mean/p95/max latency over 50 requests.

## Challenges

- **Unpickling the pipeline** required the original `clean_text`
  preprocessing function to be importable at the same module path
  (`src.preprocessing`), since the saved pipeline references it. Copying that
  one module into the new project solved it.
- **Keeping responses fast under concurrency** meant being deliberate about
  not blocking the event loop; moving even millisecond-scale sklearn calls
  into the threadpool was the fix.
- **Batch latency reporting:** per-item latency in a batch is really the
  shared vectorised call, so the response reports both the total batch
  latency and a per-item average to stay honest.

## Improvements (future work)

- Measure and report end-to-end latency (including HTTP) separately from the
  model-only `latency_ms` in responses; `scripts/benchmark.py` already does
  the end-to-end measurement.
- Replace TF-IDF embeddings with dense neural embeddings (e.g. a
  sentence-transformer model) for better paraphrase/semantic similarity.
- Add request logging and a `/metrics` endpoint (Prometheus) for production
  monitoring.
- Run uvicorn with multiple workers and add rate limiting for public
  deployment.
- Expand the training data beyond 240 examples to lift accuracy on
  out-of-domain text.

## Limitations

The classifier inherits the limits of its 240-example customer-service
dataset, and TF-IDF embeddings capture keyword overlap rather than deep
meaning. The service is a single-process deployment, suitable for demo and
moderate traffic rather than high-scale production.
