from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.csv as pacsv

from simtrading.config import AppConfig, DatasetConfig
from simtrading.prediction_schema import load_predictions


class DataLoaderError(ValueError):
    """Raised when dataset loading or alignment fails."""


_MARKET_CACHE: dict[tuple[object, ...], pd.DataFrame] = {}
_SAMPLE_CACHE: dict[tuple[object, ...], pd.DataFrame] = {}


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


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise DataLoaderError(f"Unsupported data file format: {path}. Only .csv and .parquet are supported.")


def _parse_setting(setting_key: str) -> tuple[int, int]:
    try:
        history_value, forecast_value = setting_key.split("_")
        return int(history_value.removeprefix("H")), int(forecast_value.removeprefix("F"))
    except Exception as exc:  # noqa: BLE001
        raise DataLoaderError(f"Invalid setting key `{setting_key}`. Expected format like `H60_F1`.") from exc


def _resolve_project2_path(raw_path, base_root: Path) -> Path:
    raw_text = str(raw_path).replace("\\", "/")
    resolved = Path(raw_text).expanduser()
    if resolved.exists():
        return resolved
    if raw_text.startswith("/home/zhw/Project_2/"):
        return base_root / raw_text.removeprefix("/home/zhw/Project_2/")
    if raw_text.startswith("/home/zhw/OilETF-TimeMMD/"):
        return Path("D:/Workspace/OilETF-TimeMMD") / raw_text.removeprefix("/home/zhw/OilETF-TimeMMD/")
    return resolved


def _normalize_market_data(frame: pd.DataFrame, dataset_cfg: DatasetConfig) -> pd.DataFrame:
    panel = dataset_cfg.panel
    required = {
        panel.date_column,
        panel.price_columns.execution_price,
        panel.price_columns.mark_price,
        panel.price_columns.benchmark_price,
    }
    if panel.symbol_mode == "column":
        required.add(panel.symbol_column)
    missing = required.difference(frame.columns)
    if missing:
        raise DataLoaderError(
            f"Market panel for `{dataset_cfg.name}` is missing explicitly configured columns: {sorted(missing)}."
        )

    out = frame.copy()
    out["end_date"] = pd.to_datetime(out[panel.date_column], errors="coerce")
    if out["end_date"].isna().any():
        raise DataLoaderError(f"Market panel `{panel.path}` contains unparseable dates in `{panel.date_column}`.")

    if panel.symbol_mode == "column":
        out["symbol"] = out[panel.symbol_column].astype(str)
    else:
        out["symbol"] = str(panel.constant_symbol)
    if out["symbol"].str.strip().eq("").any():
        raise DataLoaderError(f"Market panel `{panel.path}` contains empty symbols after normalization.")

    for column in (
        panel.price_columns.execution_price,
        panel.price_columns.mark_price,
        panel.price_columns.benchmark_price,
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any():
            raise DataLoaderError(f"Market panel column `{column}` contains non-numeric values.")

    return out.sort_values(["symbol", "end_date"]).reset_index(drop=True)


def _normalize_metadata_sample(frame: pd.DataFrame, dataset_cfg: DatasetConfig, setting_key: str) -> pd.DataFrame:
    required = {"sample_id", "symbol", "end_date", "H", "F", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataLoaderError(f"Sample metadata is missing explicitly required columns: {sorted(missing)}.")

    history, forecast = _parse_setting(setting_key)
    out = frame.copy()
    out["H"] = pd.to_numeric(out["H"], errors="coerce").astype("Int64")
    out["F"] = pd.to_numeric(out["F"], errors="coerce").astype("Int64")
    out = out[(out["H"] == history) & (out["F"] == forecast)].copy()
    if out.empty:
        raise DataLoaderError(f"Sample metadata has no rows for setting `{setting_key}`.")

    if dataset_cfg.samples.symbol_mode == "constant":
        out["symbol"] = str(dataset_cfg.samples.constant_symbol)
    else:
        out["symbol"] = out[dataset_cfg.samples.symbol_column].astype(str)
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    if out["end_date"].isna().any():
        raise DataLoaderError("Sample metadata contains unparseable `end_date` values.")
    out["split"] = out["split"].astype(str).str.lower()
    return out.sort_values(["symbol", "end_date"]).reset_index(drop=True)


def _normalize_series_manifest_sample(frame: pd.DataFrame, setting_key: str, base_root: Path) -> pd.DataFrame:
    required = {"series_id", "csv_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataLoaderError(f"Series manifest is missing explicitly required columns: {sorted(missing)}.")

    history, forecast = _parse_setting(setting_key)
    records = []
    for series_id, csv_path in frame[["series_id", "csv_path"]].itertuples(index=False):
        symbol = str(series_id)
        resolved = _resolve_project2_path(csv_path, base_root)
        if not resolved.exists():
            raise DataLoaderError(f"Series manifest for `{symbol}` points to a missing file: {resolved}.")
        series_table = pacsv.read_csv(resolved, convert_options=pacsv.ConvertOptions(include_columns=["end_date"]))
        end_dates = pd.to_datetime(series_table["end_date"].to_pylist(), errors="coerce")
        if end_dates.isna().any():
            raise DataLoaderError(f"Series file `{resolved}` contains unparseable `end_date` values.")

        n = len(end_dates)
        train_rows = int(n * 0.7)
        val_rows = int(n * 0.2)
        test_start = max(0, train_rows + val_rows - history)
        max_windows = n - test_start - (history + forecast) + 1
        if max_windows <= 0:
            continue

        for window_id, end_date in enumerate(end_dates[test_start : test_start + max_windows]):
            records.append(
                {
                    "sample_id": f"{symbol}_{end_date.strftime('%Y-%m-%d')}_{setting_key}_w{window_id}",
                    "symbol": symbol,
                    "end_date": end_date,
                    "H": history,
                    "F": forecast,
                    "split": "test",
                    "window": window_id,
                }
            )

    if not records:
        raise DataLoaderError(f"Series manifest produced no sample rows for setting `{setting_key}`.")
    return pd.DataFrame.from_records(records).sort_values(["symbol", "end_date"]).reset_index(drop=True)


def _normalize_sample_data(frame: pd.DataFrame, dataset_cfg: DatasetConfig, setting_key: str, base_root: Path) -> pd.DataFrame:
    if dataset_cfg.samples.format == "metadata_table":
        return _normalize_metadata_sample(frame, dataset_cfg, setting_key)
    if dataset_cfg.samples.format == "series_manifest":
        return _normalize_series_manifest_sample(frame, setting_key, base_root)
    raise DataLoaderError(f"Unsupported sample format `{dataset_cfg.samples.format}`.")


def _load_market_data(dataset_cfg: DatasetConfig) -> pd.DataFrame:
    panel = dataset_cfg.panel
    key = (
        str(panel.path),
        panel.date_column,
        panel.symbol_mode,
        panel.symbol_column,
        panel.constant_symbol,
        panel.price_columns.execution_price,
        panel.price_columns.mark_price,
        panel.price_columns.benchmark_price,
    )
    if key not in _MARKET_CACHE:
        _MARKET_CACHE[key] = _normalize_market_data(_read_frame(panel.path), dataset_cfg)
    return _MARKET_CACHE[key].copy()


def _load_sample_data(dataset_cfg: DatasetConfig, setting_key: str, base_root: Path) -> pd.DataFrame:
    key = (
        dataset_cfg.samples.format,
        str(dataset_cfg.sample_paths[setting_key]),
        setting_key,
        dataset_cfg.samples.symbol_mode,
        dataset_cfg.samples.symbol_column,
        dataset_cfg.samples.constant_symbol,
        str(base_root),
    )
    if key not in _SAMPLE_CACHE:
        _SAMPLE_CACHE[key] = _normalize_sample_data(
            _read_frame(dataset_cfg.sample_paths[setting_key]),
            dataset_cfg,
            setting_key,
            base_root,
        )
    return _SAMPLE_CACHE[key].copy()


def _filter_predictions(
    predictions: pd.DataFrame,
    dataset_cfg: DatasetConfig,
    setting_key: str,
    model_name: str,
) -> pd.DataFrame:
    if setting_key not in dataset_cfg.predictions.signal_horizons:
        raise DataLoaderError(f"Missing signal horizon configuration for setting `{setting_key}`.")
    signal_horizon = dataset_cfg.predictions.signal_horizons[setting_key]
    mask = (
        (predictions["dataset"] == dataset_cfg.dataset_label)
        & (predictions["model"] == model_name)
        & (predictions["setting"] == setting_key)
    )
    out = predictions.loc[mask].copy()
    if out.empty:
        raise DataLoaderError(
            f"No prediction rows matched dataset={dataset_cfg.dataset_label}, model={model_name}, setting={setting_key}."
        )

    if dataset_cfg.panel.symbol_mode == "constant":
        out["symbol"] = str(dataset_cfg.panel.constant_symbol)

    out = out.loc[out["horizon"] == signal_horizon].copy()
    if out.empty:
        raise DataLoaderError(
            f"No prediction rows matched configured signal horizon {signal_horizon} for setting `{setting_key}`."
        )
    if out[["symbol", "end_date"]].duplicated().any():
        raise DataLoaderError(
            f"Prediction rows for `{model_name}` and `{setting_key}` are not unique by symbol/end_date after horizon filtering."
        )
    return out.sort_values(["symbol", "end_date"]).reset_index(drop=True)


def _ensure_prediction_alignment(
    sample_data: pd.DataFrame,
    market_data: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    sample_keys = set(zip(sample_data["symbol"], sample_data["end_date"]))
    prediction_keys = set(zip(predictions["symbol"], predictions["end_date"]))
    missing_samples = prediction_keys.difference(sample_keys)
    if missing_samples:
        example = sorted(missing_samples, key=lambda item: (item[0], item[1]))[:5]
        raise DataLoaderError(f"Prediction rows are missing from sample metadata. Examples: {example}.")

    market_keys = set(zip(market_data["symbol"], market_data["end_date"]))
    missing_market = prediction_keys.difference(market_keys)
    if missing_market:
        example = sorted(missing_market, key=lambda item: (item[0], item[1]))[:5]
        raise DataLoaderError(f"Prediction rows are missing from market calendar. Examples: {example}.")

    market_last_dates = market_data.groupby("symbol")["end_date"].max()
    executable = predictions.apply(lambda row: row["end_date"] < market_last_dates.get(row["symbol"], pd.Timestamp.min), axis=1)
    if not executable.any():
        raise DataLoaderError("Prediction data has no rows with an executable market date after `end_date`.")


def prediction_file_supports_setting(dataset_cfg: DatasetConfig, model_name: str, setting_key: str) -> bool:
    path = dataset_cfg.prediction_result_paths[model_name]
    try:
        predictions = load_predictions(path, require_quantiles=False)
    except Exception:  # noqa: BLE001
        return False
    signal_horizon = dataset_cfg.predictions.signal_horizons.get(setting_key)
    if signal_horizon is None:
        return False
    mask = (
        (predictions["dataset"] == dataset_cfg.dataset_label)
        & (predictions["model"] == model_name)
        & (predictions["setting"] == setting_key)
        & (predictions["horizon"] == signal_horizon)
    )
    return bool(mask.any())


def load_dataset(app_config: AppConfig, dataset_name: str, setting_key: str, model_name: str) -> DatasetBundle:
    if dataset_name not in app_config.datasets:
        raise DataLoaderError(f"Unknown dataset `{dataset_name}`.")
    dataset_cfg = app_config.datasets[dataset_name]
    if setting_key not in dataset_cfg.sample_paths:
        raise DataLoaderError(f"Unknown setting `{setting_key}` for dataset `{dataset_name}`.")
    if model_name not in dataset_cfg.prediction_result_paths:
        raise DataLoaderError(f"Unknown model `{model_name}` for dataset `{dataset_name}`.")

    base_root = Path(__file__).resolve().parents[2]
    market_data = _load_market_data(dataset_cfg)
    sample_data = _load_sample_data(dataset_cfg, setting_key, base_root)
    predictions = load_predictions(dataset_cfg.prediction_result_paths[model_name], require_quantiles=False)
    filtered_predictions = _filter_predictions(predictions, dataset_cfg, setting_key, model_name)
    _ensure_prediction_alignment(sample_data, market_data, filtered_predictions)

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
