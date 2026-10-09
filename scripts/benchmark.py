"""Latency benchmark: measures real request latency against the 200 ms budget.

Run with:  python scripts/benchmark.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app import config
from app.main import app

TEXTS = [
    "My order arrived broken and nobody replied.",
    "Do you deliver on Sundays?",
    "The service was very helpful.",
    "The weather forecast says rain tomorrow.",
    "Where is my parcel? It has been two weeks.",
]

N_REQUESTS = 50


def main() -> None:
    with TestClient(app) as client:
        predict_latencies: list[float] = []
        for i in range(N_REQUESTS):
            start = time.perf_counter()
            response = client.post("/predict", json={"text": TEXTS[i % len(TEXTS)]})
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert response.status_code == 200
            predict_latencies.append(elapsed_ms)

        start = time.perf_counter()
        response = client.post("/embed", json={"text": TEXTS[0]})
        embed_ms = (time.perf_counter() - start) * 1000
        assert response.status_code == 200

    print(f"predict: {N_REQUESTS} requests")
    print(f"  mean   : {statistics.mean(predict_latencies):7.2f} ms")
    print(f"  p95    : {statistics.quantiles(predict_latencies, n=20)[18]:7.2f} ms")
    print(f"  max    : {max(predict_latencies):7.2f} ms")
    print(f"  budget : {config.LATENCY_BUDGET_MS:.0f} ms")
    print(f"embed  : single request took {embed_ms:.2f} ms")
    slowest = max(predict_latencies)
    if slowest < config.LATENCY_BUDGET_MS:
        print("OK: all requests stayed under the latency budget.")
    else:
        print("WARNING: some requests exceeded the latency budget.")


if __name__ == "__main__":
    main()
