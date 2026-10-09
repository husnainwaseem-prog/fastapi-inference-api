"""Shared paths, constants, custom errors and dataset loading/validation."""
from pathlib import Path

import pandas as pd

from src.preprocessing import clean_text

# Paths are built relative to this file, so the project works on any machine.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

CATEGORIES = ["Complaint", "Inquiry", "Feedback", "Other"]
REQUIRED_COLUMNS = ["text", "category"]


class DatasetError(Exception):
    """The dataset file is missing, unreadable, or has invalid content."""


class ModelNotFoundError(Exception):
    """The trained model file does not exist or cannot be loaded."""


class InvalidInputError(ValueError):
    """The text given for prediction cannot be classified."""


def load_dataset(path=DATASET_PATH):
    """Load and validate the CSV. Returns (clean_dataframe, report_dict).

    Rows with missing text/label and exact duplicates are removed and counted
    in the report. Structural problems raise DatasetError with a clear message.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")

    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DatasetError(f"Dataset file is not a valid CSV: {error}") from error

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise DatasetError(f"Dataset is missing required column(s): {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    rows_loaded = len(df)

    # Missing or blank text / label
    df["text"] = df["text"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip()
    df = df.replace("", pd.NA)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    rows_missing = rows_loaded - len(df)

    # Rows whose text becomes empty after cleaning (e.g. "!!!") are useless
    df = df[df["text"].map(clean_text) != ""]

    # Exact duplicates (ignoring upper/lower case)
    before_dedup = len(df)
    df = df.loc[~df["text"].str.lower().duplicated()]
    rows_duplicate = before_dedup - len(df)

    unknown = sorted(set(df["category"]) - set(CATEGORIES))
    if unknown:
        raise DatasetError(f"Unknown category label(s) found: {unknown}. Expected: {CATEGORIES}")

    counts = df["category"].value_counts()
    absent = [c for c in CATEGORIES if counts.get(c, 0) < 2]
    if absent:
        raise DatasetError(f"Each category needs at least 2 examples. Too few for: {absent}")

    df = df.reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["category"] = df["category"].astype(str)
    report = {
        "rows_loaded": rows_loaded,
        "rows_removed_missing_or_empty": rows_loaded - before_dedup,
        "rows_removed_duplicates": rows_duplicate,
        "rows_final": len(df),
        "class_distribution": {c: int(counts[c]) for c in CATEGORIES},
    }
    return df, report


if __name__ == "__main__":
    # `python -m src.utils` checks the dataset and prints a summary.
    try:
        _, summary = load_dataset()
    except DatasetError as error:
        print(f"ERROR: {error}")
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
