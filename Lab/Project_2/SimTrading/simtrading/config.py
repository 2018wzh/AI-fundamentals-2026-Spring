from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"


class ConfigError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True)
class PriceColumns:
    execution_price: str
    mark_price: str
    benchmark_price: str


@dataclass(frozen=True)
class PanelConfig:
    path: Path
    date_column: str
    symbol_mode: str
    symbol_column: str | None
    constant_symbol: str | None
    price_columns: PriceColumns


@dataclass(frozen=True)
class SamplesConfig:
    format: str
    symbol_mode: str
    symbol_column: str | None
    constant_symbol: str | None
    paths: Dict[str, Path]


@dataclass(frozen=True)
class PredictionsConfig:
    require_setting: bool
    signal_horizons: Dict[str, int]
    result_paths: Dict[str, Path]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display_name: str
    dataset_label: str
    panel: PanelConfig
    samples: SamplesConfig
    predictions: PredictionsConfig
    benchmark_symbols: list[str]

    @property
    def panel_path(self) -> Path:
        return self.panel.path

    @property
    def sample_paths(self) -> Dict[str, Path]:
        return self.samples.paths

    @property
    def date_column(self) -> str:
        return "end_date"

    @property
    def symbol_column(self) -> str:
        return "symbol"

    @property
    def price_columns(self) -> PriceColumns:
        return self.panel.price_columns

    @property
    def prediction_result_paths(self) -> Dict[str, Path]:
        return self.predictions.result_paths


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    datasets: Dict[str, DatasetConfig]


def _require_mapping(value, message: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(message)
    return value


def _require_non_empty_str(value, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"`{field_name}` must be a non-empty string.")
    return value.strip()


def _require_existing_path(value, field_name: str) -> Path:
    path = Path(_require_non_empty_str(value, field_name)).expanduser()
    if not path.exists():
        raise ConfigError(
            f"`{field_name}` points to a missing path: {path}. Update `config/datasets.yaml` with a valid local path."
        )
    return path


def _parse_price_columns(raw_value, dataset_name: str) -> PriceColumns:
    price_columns_raw = _require_mapping(
        raw_value, f"`datasets.{dataset_name}.panel.price_columns` must be a mapping."
    )
    return PriceColumns(
        execution_price=_require_non_empty_str(
            price_columns_raw.get("execution_price"), f"datasets.{dataset_name}.panel.price_columns.execution_price"
        ),
        mark_price=_require_non_empty_str(
            price_columns_raw.get("mark_price"), f"datasets.{dataset_name}.panel.price_columns.mark_price"
        ),
        benchmark_price=_require_non_empty_str(
            price_columns_raw.get("benchmark_price"), f"datasets.{dataset_name}.panel.price_columns.benchmark_price"
        ),
    )


def _parse_panel(raw_value, dataset_name: str) -> PanelConfig:
    panel = _require_mapping(raw_value, f"`datasets.{dataset_name}.panel` must be a mapping.")
    symbol_mode = _require_non_empty_str(panel.get("symbol_mode"), f"datasets.{dataset_name}.panel.symbol_mode")
    if symbol_mode not in {"column", "constant"}:
        raise ConfigError(f"`datasets.{dataset_name}.panel.symbol_mode` must be either `column` or `constant`.")

    symbol_column = panel.get("symbol_column")
    constant_symbol = panel.get("constant_symbol")
    if symbol_mode == "column":
        symbol_column = _require_non_empty_str(symbol_column, f"datasets.{dataset_name}.panel.symbol_column")
        constant_symbol = None
    else:
        constant_symbol = _require_non_empty_str(constant_symbol, f"datasets.{dataset_name}.panel.constant_symbol")
        symbol_column = None

    return PanelConfig(
        path=_require_existing_path(panel.get("path"), f"datasets.{dataset_name}.panel.path"),
        date_column=_require_non_empty_str(panel.get("date_column"), f"datasets.{dataset_name}.panel.date_column"),
        symbol_mode=symbol_mode,
        symbol_column=symbol_column,
        constant_symbol=constant_symbol,
        price_columns=_parse_price_columns(panel.get("price_columns"), dataset_name),
    )


def _parse_sample_paths(raw_value, dataset_name: str) -> Dict[str, Path]:
    sample_paths = _require_mapping(raw_value, f"`datasets.{dataset_name}.samples.paths` must be a mapping.")
    required = {"H60_F1", "H120_F5"}
    missing = required.difference(sample_paths.keys())
    if missing:
        raise ConfigError(f"`datasets.{dataset_name}.samples.paths` is missing required settings: {sorted(missing)}.")
    return {
        key: _require_existing_path(value, f"datasets.{dataset_name}.samples.paths.{key}")
        for key, value in sample_paths.items()
    }


def _parse_samples(raw_value, dataset_name: str) -> SamplesConfig:
    samples = _require_mapping(raw_value, f"`datasets.{dataset_name}.samples` must be a mapping.")
    sample_format = _require_non_empty_str(samples.get("format"), f"datasets.{dataset_name}.samples.format")
    if sample_format not in {"metadata_table", "series_manifest"}:
        raise ConfigError(
            f"`datasets.{dataset_name}.samples.format` must be either `metadata_table` or `series_manifest`."
        )
    symbol_mode = _require_non_empty_str(samples.get("symbol_mode"), f"datasets.{dataset_name}.samples.symbol_mode")
    if symbol_mode not in {"column", "constant"}:
        raise ConfigError(f"`datasets.{dataset_name}.samples.symbol_mode` must be either `column` or `constant`.")
    symbol_column = samples.get("symbol_column")
    constant_symbol = samples.get("constant_symbol")
    if symbol_mode == "column":
        symbol_column = _require_non_empty_str(symbol_column, f"datasets.{dataset_name}.samples.symbol_column")
        constant_symbol = None
    else:
        constant_symbol = _require_non_empty_str(constant_symbol, f"datasets.{dataset_name}.samples.constant_symbol")
        symbol_column = None
    return SamplesConfig(
        format=sample_format,
        symbol_mode=symbol_mode,
        symbol_column=symbol_column,
        constant_symbol=constant_symbol,
        paths=_parse_sample_paths(samples.get("paths"), dataset_name),
    )


def _parse_prediction_paths(raw_value, dataset_name: str) -> Dict[str, Path]:
    prediction_paths = _require_mapping(
        raw_value, f"`datasets.{dataset_name}.predictions.result_paths` must be a mapping."
    )
    if not prediction_paths:
        raise ConfigError(f"`datasets.{dataset_name}.predictions.result_paths` must declare at least one model.")
    return {
        model: _require_existing_path(path, f"datasets.{dataset_name}.predictions.result_paths.{model}")
        for model, path in prediction_paths.items()
    }


def _parse_signal_horizons(raw_value, dataset_name: str, sample_paths: Dict[str, Path]) -> Dict[str, int]:
    raw = _require_mapping(raw_value, f"`datasets.{dataset_name}.predictions.signal_horizons` must be a mapping.")
    missing = set(sample_paths).difference(raw.keys())
    if missing:
        raise ConfigError(
            f"`datasets.{dataset_name}.predictions.signal_horizons` is missing settings: {sorted(missing)}."
        )
    horizons: Dict[str, int] = {}
    for setting, value in raw.items():
        if not isinstance(value, int) or value < 0:
            raise ConfigError(
                f"`datasets.{dataset_name}.predictions.signal_horizons.{setting}` must be a non-negative integer."
            )
        horizons[setting] = value
    return horizons


def _parse_predictions(raw_value, dataset_name: str, sample_paths: Dict[str, Path]) -> PredictionsConfig:
    predictions = _require_mapping(raw_value, f"`datasets.{dataset_name}.predictions` must be a mapping.")
    require_setting = predictions.get("require_setting")
    if require_setting is not True:
        raise ConfigError(f"`datasets.{dataset_name}.predictions.require_setting` must be true.")
    return PredictionsConfig(
        require_setting=True,
        signal_horizons=_parse_signal_horizons(predictions.get("signal_horizons"), dataset_name, sample_paths),
        result_paths=_parse_prediction_paths(predictions.get("result_paths"), dataset_name),
    )


def _parse_benchmarks(raw_value, dataset_name: str) -> list[str]:
    if not isinstance(raw_value, list) or not raw_value:
        raise ConfigError(f"`datasets.{dataset_name}.benchmark_symbols` must be a non-empty list.")
    return [_require_non_empty_str(item, f"datasets.{dataset_name}.benchmark_symbols[]") for item in raw_value]


def _parse_dataset(name: str, raw_value) -> DatasetConfig:
    dataset = _require_mapping(raw_value, f"`datasets.{name}` must be a mapping.")
    samples = _parse_samples(dataset.get("samples"), name)
    return DatasetConfig(
        name=name,
        display_name=_require_non_empty_str(dataset.get("display_name"), f"datasets.{name}.display_name"),
        dataset_label=_require_non_empty_str(dataset.get("dataset_label"), f"datasets.{name}.dataset_label"),
        panel=_parse_panel(dataset.get("panel"), name),
        samples=samples,
        predictions=_parse_predictions(dataset.get("predictions"), name, samples.paths),
        benchmark_symbols=_parse_benchmarks(dataset.get("benchmark_symbols"), name),
    )


def load_app_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not config_path.exists():
        raise ConfigError(
            f"Missing config file: {config_path}. Copy `config/datasets.example.yaml` to `config/datasets.yaml` first."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("Top-level config must be a mapping.")
    datasets_raw = _require_mapping(raw.get("datasets"), "Top-level `datasets` must be a mapping.")
    if not datasets_raw:
        raise ConfigError("Config must declare at least one dataset.")
    datasets = {name: _parse_dataset(name, value) for name, value in datasets_raw.items()}
    return AppConfig(config_path=config_path, datasets=datasets)
