from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from simtrading.prediction_schema import QUANTILE_COLUMNS


class StrategyError(ValueError):
    """Raised when a strategy cannot be executed with the provided data."""


class StrategyName(str, Enum):
    LONG_CASH = "Long/Cash Threshold"
    CONFIDENCE_LONG_CASH = "Confidence-Adjusted Long/Cash"
    TOP_K_ROTATION = "Top-K Rotation"


@dataclass(frozen=True)
class StrategyConfig:
    name: StrategyName
    threshold: float
    fee_bps: float
    slippage_bps: float
    top_k: int
    max_position: float
    score_column: str


def get_strategy_availability(predictions: pd.DataFrame) -> dict[StrategyName, tuple[bool, str]]:
    availability: dict[StrategyName, tuple[bool, str]] = {
        StrategyName.LONG_CASH: (True, "Requires `y_pred`."),
        StrategyName.CONFIDENCE_LONG_CASH: (True, "Requires `q10`, `q50`, and `q90`."),
        StrategyName.TOP_K_ROTATION: (True, "Requires multiple symbols per date."),
    }
    if not QUANTILE_COLUMNS.issubset(predictions.columns):
        availability[StrategyName.CONFIDENCE_LONG_CASH] = (
            False,
            "Unavailable because the prediction file does not include `q10`, `q50`, and `q90`.",
        )
    symbol_counts = predictions.groupby("end_date")["symbol"].nunique()
    if symbol_counts.empty or int(symbol_counts.min()) < 2:
        availability[StrategyName.TOP_K_ROTATION] = (
            False,
            "Unavailable because at least one prediction date has fewer than 2 unique symbols.",
        )
    return availability


def _require_quantiles(predictions: pd.DataFrame) -> None:
    missing = QUANTILE_COLUMNS.difference(predictions.columns)
    if missing:
        raise StrategyError(
            f"Selected strategy requires quantile columns, but the prediction file is missing: {sorted(missing)}."
        )


def validate_strategy_inputs(predictions: pd.DataFrame, strategy_config: StrategyConfig) -> None:
    if strategy_config.score_column not in predictions.columns:
        raise StrategyError(f"Prediction data is missing score column `{strategy_config.score_column}`.")
    if strategy_config.max_position <= 0 or strategy_config.max_position > 1:
        raise StrategyError("`max_position` must be within (0, 1].")
    if strategy_config.top_k < 1:
        raise StrategyError("`top_k` must be at least 1.")
    if strategy_config.name == StrategyName.CONFIDENCE_LONG_CASH:
        _require_quantiles(predictions)
    if strategy_config.name == StrategyName.TOP_K_ROTATION:
        counts = predictions.groupby("end_date")["symbol"].nunique()
        if (counts < strategy_config.top_k).any():
            raise StrategyError("`Top-K Rotation` requires at least `top_k` unique symbols on every trading date.")


def build_signal_frame(predictions: pd.DataFrame, strategy_config: StrategyConfig) -> pd.DataFrame:
    validate_strategy_inputs(predictions, strategy_config)
    signals = predictions.copy()

    if strategy_config.name == StrategyName.LONG_CASH:
        signals["score"] = signals[strategy_config.score_column]
        signals["raw_weight"] = (signals["score"] > strategy_config.threshold).astype(float) * strategy_config.max_position
    elif strategy_config.name == StrategyName.CONFIDENCE_LONG_CASH:
        _require_quantiles(signals)
        interval_width = signals["q90"] - signals["q10"]
        if (interval_width <= 0).any():
            raise StrategyError("Quantile columns produced non-positive confidence intervals.")
        signals["score"] = signals["q50"] / interval_width
        signals["raw_weight"] = ((signals["score"] > strategy_config.threshold) & (signals["q10"] > 0)).astype(float)
        signals["raw_weight"] = signals["raw_weight"] * strategy_config.max_position
    elif strategy_config.name == StrategyName.TOP_K_ROTATION:
        signals["score"] = signals[strategy_config.score_column]
        signals = signals.sort_values(["end_date", "score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
        ranks = signals.groupby("end_date")["score"].rank(method="first", ascending=False)
        signals["raw_weight"] = (ranks <= strategy_config.top_k).astype(float)
        signals["raw_weight"] = signals["raw_weight"] * (strategy_config.max_position / strategy_config.top_k)
    else:
        raise StrategyError(f"Unsupported strategy `{strategy_config.name}`.")

    return signals
