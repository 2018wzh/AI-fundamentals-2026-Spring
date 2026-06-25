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
class DatasetConfig:
    name: str
    display_name: str
    panel_path: Path
    sample_paths: Dict[str, Path]
    date_column: str
    symbol_column: str
    price_columns: PriceColumns
    prediction_result_paths: Dict[str, Path]
    benchmark_symbols: list[str]


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


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _require_existing_path(value, field_name: str) -> Path:
    path = _resolve_path(_require_non_empty_str(value, field_name))
    if not path.exists():
        raise ConfigError(
            f"`{field_name}` points to a missing path: {path}. Update `config/datasets.yaml` with a valid local path."
        )
    return path


def _parse_sample_paths(raw_value) -> Dict[str, Path]:
    sample_paths = _require_mapping(raw_value, "`sample_paths` must be a mapping.")
    if not sample_paths:
        raise ConfigError("`sample_paths` must declare at least one setting.")
    return {key: _require_existing_path(value, f"sample_paths.{key}") for key, value in sample_paths.items()}


def _parse_prediction_paths(raw_value) -> Dict[str, Path]:
    prediction_paths = _require_mapping(raw_value, "`prediction_result_paths` must be a mapping.")
    if not prediction_paths:
        raise ConfigError("`prediction_result_paths` must declare at least one model result file.")
    return {model: _require_existing_path(path, f"prediction_result_paths.{model}") for model, path in prediction_paths.items()}


def _parse_benchmarks(raw_value) -> list[str]:
    if not isinstance(raw_value, list) or not raw_value:
        raise ConfigError("`benchmark_symbols` must be a non-empty list.")
    return [_require_non_empty_str(item, "benchmark_symbols[]") for item in raw_value]


def _parse_dataset(name: str, raw_value) -> DatasetConfig:
    dataset = _require_mapping(raw_value, f"`datasets.{name}` must be a mapping.")
    price_columns_raw = _require_mapping(dataset.get("price_columns"), f"`datasets.{name}.price_columns` must be a mapping.")
    price_columns = PriceColumns(
        execution_price=_require_non_empty_str(
            price_columns_raw.get("execution_price"), f"datasets.{name}.price_columns.execution_price"
        ),
        mark_price=_require_non_empty_str(
            price_columns_raw.get("mark_price"), f"datasets.{name}.price_columns.mark_price"
        ),
        benchmark_price=_require_non_empty_str(
            price_columns_raw.get("benchmark_price"), f"datasets.{name}.price_columns.benchmark_price"
        ),
    )
    return DatasetConfig(
        name=name,
        display_name=_require_non_empty_str(dataset.get("display_name"), f"datasets.{name}.display_name"),
        panel_path=_require_existing_path(dataset.get("panel_path"), f"datasets.{name}.panel_path"),
        sample_paths=_parse_sample_paths(dataset.get("sample_paths")),
        date_column=_require_non_empty_str(dataset.get("date_column"), f"datasets.{name}.date_column"),
        symbol_column=_require_non_empty_str(dataset.get("symbol_column"), f"datasets.{name}.symbol_column"),
        price_columns=price_columns,
        prediction_result_paths=_parse_prediction_paths(dataset.get("prediction_result_paths")),
        benchmark_symbols=_parse_benchmarks(dataset.get("benchmark_symbols")),
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
