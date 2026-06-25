from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from simtrading.config import AppConfig, DatasetConfig, PROJECT_ROOT
from simtrading.prediction_schema import load_predictions


class DataLoaderError(ValueError):
    """Raised when dataset loading or alignment fails."""


@dataclass(frozen=True)
class DatasetBundle:
    dataset_name: str
    display_name: str
    setting_key: str
    model_name: str
    dataset_config: DatasetConfig
    market_data: pd.DataFrame
    sample_data: pd.DataFrame
    predictions: pd.DataFrame


def _read_frame(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise DataLoaderError(f"Unsupported data file format: {path}. Only .csv and .parquet are supported.")


def _ensure_market_columns(frame: pd.DataFrame, dataset_cfg: DatasetConfig) -> None:
    required = {
        dataset_cfg.date_column,
        dataset_cfg.price_columns.execution_price,
        dataset_cfg.price_columns.mark_price,
        dataset_cfg.price_columns.benchmark_price,
    }
    if dataset_cfg.symbol_column in frame.columns:
        required.add(dataset_cfg.symbol_column)
    missing = required.difference(frame.columns)
    if missing:
        raise DataLoaderError(
            f"Market panel for `{dataset_cfg.name}` is missing required columns: {sorted(missing)}."
        )


def _normalize_market_data(frame: pd.DataFrame, dataset_cfg: DatasetConfig) -> pd.DataFrame:
    _ensure_market_columns(frame, dataset_cfg)
    out = frame.copy()
    out[dataset_cfg.date_column] = pd.to_datetime(out[dataset_cfg.date_column], errors="coerce")
    if out[dataset_cfg.date_column].isna().any():
        raise DataLoaderError(f"Market panel `{dataset_cfg.panel_path}` contains unparseable dates.")
    for column in (
        dataset_cfg.price_columns.execution_price,
        dataset_cfg.price_columns.mark_price,
        dataset_cfg.price_columns.benchmark_price,
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any():
            raise DataLoaderError(f"Market panel column `{column}` contains non-numeric values.")

    if dataset_cfg.symbol_column not in out.columns:
        out[dataset_cfg.symbol_column] = dataset_cfg.benchmark_symbols[0]
    return out.sort_values([dataset_cfg.symbol_column, dataset_cfg.date_column]).reset_index(drop=True)


def _normalize_sample_data(frame: pd.DataFrame) -> pd.DataFrame:
    if {"series_id", "csv_path"}.issubset(frame.columns):
        pieces = []
        for _, row in frame.iterrows():
            path = _resolve_local_path(row["csv_path"])
            if not path.exists():
                raise DataLoaderError(f"Sample manifest points to a missing path: {path}.")
            piece = pd.read_csv(path, usecols=["end_date"]) if path.suffix.lower() == ".csv" else _read_frame(path)
            if "end_date" not in piece.columns:
                raise DataLoaderError(f"Sample file `{path}` is missing `end_date`.")
            piece = piece[["end_date"]].copy()
            piece["symbol"] = str(row["series_id"])
            pieces.append(piece)
        if not pieces:
            raise DataLoaderError("Sample manifest did not contain any series.")
        out = pd.concat(pieces, ignore_index=True)
        out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
        if out["end_date"].isna().any():
            raise DataLoaderError("Sample metadata contains unparseable `end_date` values.")
        return out

    if "end_date" in frame.columns:
        out = frame.copy()
        out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
        if out["end_date"].isna().any():
            raise DataLoaderError("Sample metadata contains unparseable `end_date` values.")
        return out

    required = {"sample_id", "symbol", "end_date", "H", "F", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataLoaderError(f"Sample metadata is missing required columns: {sorted(missing)}.")
    out = frame.copy()
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    if out["end_date"].isna().any():
        raise DataLoaderError("Sample metadata contains unparseable `end_date` values.")
    return out


def _resolve_local_path(value) -> Path:
    raw = str(value).replace("\\", "/")
    if raw.startswith("/home/zhw/Project_2/"):
        return PROJECT_ROOT.parent / raw.removeprefix("/home/zhw/Project_2/")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _filter_predictions(predictions: pd.DataFrame, dataset_name: str, setting_key: str, model_name: str) -> pd.DataFrame:
    window_value, horizon_value = setting_key.split("_")
    window = int(window_value.removeprefix("H"))
    horizon = int(horizon_value.removeprefix("F"))
    mask = (predictions["dataset"].astype(str).str.lower() == dataset_name.lower()) & (
        predictions["model"].astype(str) == model_name
    )
    if "setting" in predictions.columns:
        mask = mask & (predictions["setting"].astype(str) == setting_key)
    else:
        legacy_mask = mask & (predictions["window"].astype(int) == window) & (predictions["horizon"].astype(int) == horizon)
        if legacy_mask.any():
            mask = legacy_mask
    out = predictions.loc[mask].copy()
    if out.empty:
        raise DataLoaderError(
            f"No prediction rows matched dataset={dataset_name}, model={model_name}, setting={setting_key}."
        )
    return out.sort_values(["symbol", "end_date"]).reset_index(drop=True)


def _align_single_symbol_market(market_data: pd.DataFrame, predictions: pd.DataFrame, symbol_column: str) -> pd.DataFrame:
    market_symbols = market_data[symbol_column].dropna().astype(str).unique()
    prediction_symbols = predictions["symbol"].dropna().astype(str).unique()
    if len(market_symbols) == 1 and len(prediction_symbols) == 1 and market_symbols[0] != prediction_symbols[0]:
        out = market_data.copy()
        out[symbol_column] = prediction_symbols[0]
        return out
    return market_data


def load_dataset(app_config: AppConfig, dataset_name: str, setting_key: str, model_name: str) -> DatasetBundle:
    if dataset_name not in app_config.datasets:
        raise DataLoaderError(f"Unknown dataset `{dataset_name}`.")
    dataset_cfg = app_config.datasets[dataset_name]
    if setting_key not in dataset_cfg.sample_paths:
        raise DataLoaderError(f"Unknown setting `{setting_key}` for dataset `{dataset_name}`.")
    if model_name not in dataset_cfg.prediction_result_paths:
        raise DataLoaderError(f"Unknown model `{model_name}` for dataset `{dataset_name}`.")

    market_data = _normalize_market_data(_read_frame(dataset_cfg.panel_path), dataset_cfg)
    sample_data = _normalize_sample_data(_read_frame(dataset_cfg.sample_paths[setting_key]))
    predictions = load_predictions(dataset_cfg.prediction_result_paths[model_name], require_quantiles=False)
    filtered_predictions = _filter_predictions(predictions, dataset_name, setting_key, model_name)
    market_data = _align_single_symbol_market(market_data, filtered_predictions, dataset_cfg.symbol_column)

    sample_dates = set(sample_data["end_date"])
    prediction_dates = set(filtered_predictions["end_date"])
    if not prediction_dates.issubset(sample_dates):
        raise DataLoaderError("Prediction results contain `end_date` values that are not present in the sample metadata.")

    return DatasetBundle(
        dataset_name=dataset_name,
        display_name=dataset_cfg.display_name,
        setting_key=setting_key,
        model_name=model_name,
        dataset_config=dataset_cfg,
        market_data=market_data,
        sample_data=sample_data,
        predictions=filtered_predictions,
    )
