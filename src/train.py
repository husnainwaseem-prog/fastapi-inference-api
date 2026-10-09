"""Train the TF-IDF + Logistic Regression classifier and save it.

Run from the project root:  python -m src.train
"""
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text
from src.utils import (CATEGORIES, METRICS_PATH, MODEL_PATH, DatasetError,
                       load_dataset)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_pipeline() -> Pipeline:
    """TF-IDF -> Logistic Regression, in one object.

    Keeping both steps in one Pipeline means the vectorizer is fitted on the
    training data only (no data leakage) and the saved model accepts raw text.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,  # same cleaning in training and prediction
        ngram_range=(1, 2),       # single words + word pairs ("not working")
        sublinear_tf=True,        # dampen very frequent words
    )
    classifier = LogisticRegression(
        C=10,                     # weaker regularisation: tiny dataset, short texts
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def evaluate(model: Pipeline, x_test, y_test) -> dict:
    """Compute metrics on the held-out test set."""
    predictions = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        # weighted = average of per-class scores, weighted by class size
        "f1_weighted": f1_score(y_test, predictions, average="weighted"),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "precision_weighted": precision_score(y_test, predictions, average="weighted"),
        "recall_weighted": recall_score(y_test, predictions, average="weighted"),
        "classification_report": classification_report(
            y_test, predictions, labels=CATEGORIES, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CATEGORIES).tolist(),
    }


def train_and_save() -> dict:
    """Full training workflow. Returns the metrics dictionary."""
    df, report = load_dataset()
    print("Dataset check:", report)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["category"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["category"],
    )
    print(f"Train size: {len(x_train)} | Test size: {len(x_test)}")

    model = build_pipeline()
    model.fit(x_train, y_train)

    metrics = evaluate(model, x_test, y_test)
    # 5-fold cross-validation on the training part only: a second, more stable estimate
    cv_scores = cross_val_score(build_pipeline(), x_train, y_train, cv=5, scoring="accuracy")
    metrics["cv_accuracy_mean"] = float(cv_scores.mean())
    metrics["cv_accuracy_std"] = float(cv_scores.std())
    metrics["train_size"], metrics["test_size"] = len(x_train), len(x_test)
    metrics["train_accuracy"] = accuracy_score(y_train, model.predict(x_train))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def print_results(metrics: dict) -> None:
    print(f"\nTraining accuracy (for comparison only): {metrics['train_accuracy']:.4f}")
    print(f"Test accuracy:            {metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted):       {metrics['f1_weighted']:.4f}")
    print(f"Test F1 (macro):          {metrics['f1_macro']:.4f}")
    print(f"Test precision (weighted):{metrics['precision_weighted']:.4f}")
    print(f"Test recall (weighted):   {metrics['recall_weighted']:.4f}")
    print(f"5-fold CV accuracy (train only): "
          f"{metrics['cv_accuracy_mean']:.4f} +/- {metrics['cv_accuracy_std']:.4f}")
    print("\nClassification report:\n", metrics["classification_report"])
    print("Confusion matrix (rows = true, columns = predicted)")
    print("Order:", CATEGORIES)
    for label, row in zip(CATEGORIES, metrics["confusion_matrix"]):
        print(f"{label:>10}: {row}")
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    try:
        print_results(train_and_save())
    except DatasetError as error:
        print(f"Training stopped: {error}")
        raise SystemExit(1)
