"""
Utilities for loading the Azure AutoML model, parsing CSV uploads, and running inference.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from .constants import BOX_ID_COLUMN, BOX_ID_PRED_COLUMN, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"


class MissingColumnsError(Exception):
    """Raised when the uploaded CSV lacks required feature columns."""


class CsvDecodingError(Exception):
    """Raised when the CSV cannot be decoded using the supported encodings."""


@lru_cache
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    logger.info("Loading model from %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


def _read_csv_with_fallbacks(file_bytes: bytes) -> pd.DataFrame:
    """Attempt to read the CSV using utf-8 first, then cp932."""
    last_error: Exception | None = None
    for encoding in ("utf-8", "cp932"):
        try:
            text = file_bytes.decode(encoding)
            return pd.read_csv(io.StringIO(text))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise CsvDecodingError("CSV decoding failed for encodings utf-8 and cp932") from last_error


def _validate_columns(columns: Iterable[str]) -> None:
    missing = [col for col in FEATURE_COLUMNS if col not in columns]
    if missing:
        raise MissingColumnsError(", ".join(missing))


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[FEATURE_COLUMNS].copy()
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)
    return features


def predict_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Decode the CSV, run inference, and return the enriched DataFrame."""
    df = _read_csv_with_fallbacks(file_bytes)
    _validate_columns(df.columns)
    features = _prepare_features(df)
    model = load_model()
    predictions = model.predict(features)

    result_df = df.copy()
    result_df[BOX_ID_PRED_COLUMN] = predictions

    if BOX_ID_COLUMN in result_df.columns:
        box_id_series = result_df[BOX_ID_COLUMN]
        empty_mask = box_id_series.isna() | (box_id_series.astype(str).str.strip() == "")
        result_df.loc[empty_mask, BOX_ID_COLUMN] = result_df.loc[empty_mask, BOX_ID_PRED_COLUMN]

    return result_df
