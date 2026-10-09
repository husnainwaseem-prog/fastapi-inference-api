"""Text cleaning used for BOTH training and prediction.

The same function is plugged into the TF-IDF vectorizer inside the saved
pipeline (see train.py), so raw text is always cleaned in exactly the same way.
"""
import re

_APOSTROPHES = ("'", "\u2019")  # normal and curly apostrophe


def clean_text(text) -> str:
    """Return a cleaned version of `text`.

    Steps:
      1. Anything that is not a string (None, NaN, numbers) becomes "".
      2. Lowercase, so "Order" and "order" are the same word.
      3. Remove apostrophes, so "won't" becomes "wont" (one token, not two).
      4. Replace other punctuation/symbols with a space ("late,again" -> "late again").
      5. Collapse repeated whitespace into a single space and trim the ends.

    Stop words (e.g. "not", "how") are deliberately KEPT: "not working" and
    "how do I" are exactly the clues that separate complaints and inquiries.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for apostrophe in _APOSTROPHES:
        text = text.replace(apostrophe, "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
