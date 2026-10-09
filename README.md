# FastAPI Model Inference & Embeddings API

An asynchronous FastAPI service that serves real-time predictions from a trained
text classifier (TF-IDF + LogisticRegression, 4 classes: Complaint, Inquiry,
Feedback, Other), plus text embedding vectors and cosine-similarity scoring —
all with per-request latency well under the 200 ms budget.

## Features

- `POST /predict` — classify a single text (label, confidence, per-class
  probabilities, measured `latency_ms`)
- `POST /predict/batch` — classify up to 50 texts in one vectorised call
- `POST /embed` — text as an L2-normalised TF-IDF embedding vector (1804 values, mostly zeros)
- `POST /similarity` — cosine similarity between two texts (0–1)
- `GET /health`, `GET /info` — liveness and model metadata
- Async endpoints: scikit-learn calls run in a threadpool so the event loop is
  never blocked; the model is loaded once at startup and shared
- Input validation with Pydantic (empty/oversized input → HTTP 422)
- 13 pytest tests covering correctness, validation, embeddings and latency

## Project structure

```
fastapi-inference-api/
├── app/
│   ├── main.py        # FastAPI app, lifespan, routes
│   ├── model.py       # model loading, predict, embed, similarity
│   ├── schemas.py     # Pydantic request/response models
│   └── config.py      # paths, limits, latency budget
├── src/
│   ├── preprocessing.py  # text cleaner (also used inside the saved pipeline)
│   ├── utils.py          # paths, constants, dataset loading/validation
│   └── train.py          # trains, evaluates and saves the model
├── data/
│   └── dataset.csv    # 240 labelled messages (written by me)
├── models/
│   ├── model.joblib   # trained TF-IDF + LogisticRegression pipeline
│   └── metrics.json   # evaluation numbers from training
├── tests/
│   └── test_api.py    # 13 tests (pytest)
├── scripts/
│   └── benchmark.py   # latency benchmark against the 200 ms budget
├── requirements.txt
├── Dockerfile
├── README.md
└── REPORT.md          # 1–2 page project report
```

## Setup

**Windows (PowerShell / cmd):**

```bat
cd fastapi-inference-api
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
cd fastapi-inference-api
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run the server

```bash
# Windows:
.venv\Scripts\python -m uvicorn app.main:app --reload

# macOS / Linux:
.venv/bin/python -m uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for the interactive API docs.

With Docker:

```bash
docker build -t text-api .
docker run -p 8000:8000 text-api
```

## Example requests

```bash
# Single prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My order arrived broken and nobody replied."}'

# Batch prediction
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Do you deliver on Sundays?", "The service was very helpful."]}'

# Embedding vector
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Where is my parcel?"}'

# Similarity between two texts
curl -X POST http://127.0.0.1:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{"text_a": "my refund never arrived", "text_b": "where is my refund"}'

# Health and model info
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/info
```

## Tests

```bash
# Windows:  .venv\Scripts\python -m pytest -v
# macOS/Linux:  .venv/bin/python -m pytest -v
python -m pytest -v
```

The suite includes a latency test: 15 sequential predictions must each stay
under 200 ms. `scripts/benchmark.py` reports mean/p95/max latency over
50 requests.

## Retraining the model (optional)

The trained model is already included, so this is not needed to run the API.
To reproduce it from the dataset:

```bash
python -m src.train
```

This reads `data/dataset.csv`, trains on an 80/20 stratified split
(`random_state=42`), prints test metrics (about 73% accuracy) and overwrites
`models/model.joblib` and `models/metrics.json`. Retraining with the pinned
scikit-learn version gives the same model that is shipped here.

## Acknowledgement

The classifier, its dataset and the training code (`src/train.py`,
`src/utils.py`, `src/preprocessing.py`, `data/dataset.csv`) come from my earlier
**Smart Text Classifier** internship project and are reused here on purpose.
The work done for this task is the FastAPI service in `app/`, the embeddings and
similarity endpoints, the tests, the benchmark script, the Dockerfile and the
documentation. Libraries used: FastAPI, Pydantic, scikit-learn, joblib, NumPy,
pandas, pytest, httpx.

## Limitations

- The classifier was trained on 240 short customer-service messages; accuracy
  on out-of-domain text (news, slang, other languages) will be lower.
- Embeddings are TF-IDF based (sparse vocabulary vectors, normalised, stored as
  1804 floats per text), not dense neural embeddings — good for keyword overlap, weak on paraphrase
  meaning. Swapping in a sentence-transformer model is the natural upgrade.
- A text containing only words the model has never seen gets an all-zero
  embedding, so its similarity to anything (even an identical text) is 0.
- scikit-learn is pinned to 1.9.1 because the saved model is only guaranteed to
  load with the version that created it.
- Single-process deployment; for heavy traffic run uvicorn with multiple
  workers (`--workers 4`) or behind a process manager.
