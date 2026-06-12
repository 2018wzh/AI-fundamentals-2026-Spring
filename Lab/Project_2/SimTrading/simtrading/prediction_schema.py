from __future__ import annotations

from pathlib import Path

import pandas as pd


class PredictionSchemaError(ValueError):
    """Raised when a prediction file violates the required schema."""


REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "setting",
    "symbol",
    "end_date",
    "window",
    "horizon",
    "y_true",
    "y_pred",
    "split",
}
QUANTILE_COLUMNS = {"q10", "q50", "q90"}
ALLOWED_SPLITS = {"train", "val", "test"}
UNIQUE_KEY_COLUMNS = ["dataset", "model", "setting", "symbol", "end_date", "window", "horizon"]


def _read_predictions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise PredictionSchemaError(f"Unsupported prediction file format: {path}. Only .csv and .parquet are supported.")


def validate_prediction_schema(df: pd.DataFrame, require_quantiles: bool = False) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise PredictionSchemaError(f"Prediction file is missing required columns: {sorted(missing)}.")

    if require_quantiles:
        quantile_missing = QUANTILE_COLUMNS.difference(df.columns)
        if quantile_missing:
            raise PredictionSchemaError(
                f"Prediction file is missing required quantile columns: {sorted(quantile_missing)}."
            )

    parsed_dates = pd.to_datetime(df["end_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise PredictionSchemaError("Column `end_date` contains unparseable values.")

    invalid_splits = sorted(set(df["split"].astype(str).str.lower()) - ALLOWED_SPLITS)
    if invalid_splits:
        raise PredictionSchemaError(f"Column `split` contains unsupported values: {invalid_splits}.")

    for column in ("dataset", "model", "setting", "symbol"):
        if df[column].astype(str).str.strip().eq("").any():
            raise PredictionSchemaError(f"Column `{column}` contains empty values.")

    numeric_columns = ["window", "horizon", "y_true", "y_pred"]
    if require_quantiles:
        numeric_columns.extend(sorted(QUANTILE_COLUMNS))
    for column in numeric_columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().any():
            raise PredictionSchemaError(f"Column `{column}` contains non-numeric values.")

    if df[UNIQUE_KEY_COLUMNS].duplicated().any():
        raise PredictionSchemaError(
            "Prediction file contains duplicate rows for the same dataset/model/setting/symbol/date/window/horizon."
        )


def load_predictions(path: Path, require_quantiles: bool = False) -> pd.DataFrame:
    df = _read_predictions(path)
    validate_prediction_schema(df, require_quantiles=require_quantiles)
    out = df.copy()
    out["end_date"] = pd.to_datetime(out["end_date"])
    out["window"] = pd.to_numeric(out["window"]).astype(int)
    out["horizon"] = pd.to_numeric(out["horizon"]).astype(int)
    out["y_true"] = pd.to_numeric(out["y_true"])
    out["y_pred"] = pd.to_numeric(out["y_pred"])
    if QUANTILE_COLUMNS.issubset(out.columns):
        for column in QUANTILE_COLUMNS:
            out[column] = pd.to_numeric(out[column])
    out["dataset"] = out["dataset"].astype(str)
    out["model"] = out["model"].astype(str)
    out["setting"] = out["setting"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["split"] = out["split"].astype(str).str.lower()
    return out
