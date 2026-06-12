#!/usr/bin/env python3
"""Validate SimTrading against the explicit real-data contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simtrading.backtest import run_backtest
from simtrading.config import load_app_config
from simtrading.data_loader import load_dataset, prediction_file_supports_setting
from simtrading.strategy import StrategyConfig, StrategyError, StrategyName, get_strategy_availability


def _long_cash_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.LONG_CASH,
        threshold=0.0,
        fee_bps=2.0,
        slippage_bps=2.0,
        top_k=1,
        max_position=1.0,
        score_column="y_pred",
    )


def _confidence_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.CONFIDENCE_LONG_CASH,
        threshold=0.0,
        fee_bps=2.0,
        slippage_bps=2.0,
        top_k=1,
        max_position=1.0,
        score_column="y_pred",
    )


def _topk_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.TOP_K_ROTATION,
        threshold=0.0,
        fee_bps=2.0,
        slippage_bps=2.0,
        top_k=3,
        max_position=1.0,
        score_column="y_pred",
    )


def validate_config(config_path: Path, dataset_filter: str | None = None) -> int:
    issues: list[str] = []
    rows: list[dict[str, object]] = []
    try:
        app_config = load_app_config(config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load config: {exc}")
        return 2

    for dataset_name, dataset_cfg in app_config.datasets.items():
        if dataset_filter and dataset_name != dataset_filter:
            continue
        for setting_key in dataset_cfg.sample_paths:
            supported_models = [
                model_name
                for model_name in dataset_cfg.prediction_result_paths
                if prediction_file_supports_setting(dataset_cfg, model_name, setting_key)
            ]
            if not supported_models:
                issues.append(f"[{dataset_name}/{setting_key}] no model has matching prediction rows.")
                continue

            for model_name in supported_models:
                owner = f"{dataset_name}/{setting_key}/{model_name}"
                try:
                    bundle = load_dataset(app_config, dataset_name, setting_key, model_name)
                    long_result = run_backtest(bundle, _long_cash_config())
                except Exception as exc:  # noqa: BLE001
                    issues.append(f"[{owner}] load or Long/Cash backtest failed: {exc}")
                    continue

                availability = get_strategy_availability(bundle.predictions)
                if availability[StrategyName.CONFIDENCE_LONG_CASH][0]:
                    try:
                        run_backtest(bundle, _confidence_config())
                    except Exception as exc:  # noqa: BLE001
                        issues.append(f"[{owner}] Confidence backtest failed: {exc}")

                if dataset_name == "fnspid":
                    try:
                        run_backtest(bundle, _topk_config())
                    except (StrategyError, Exception) as exc:  # noqa: BLE001
                        issues.append(f"[{owner}] Top-K backtest failed: {exc}")
                elif availability[StrategyName.TOP_K_ROTATION][0]:
                    issues.append(f"[{owner}] Top-K should be unavailable for single-symbol data.")

                rows.append(
                    {
                        "owner": owner,
                        "predictions": len(bundle.predictions),
                        "market": len(bundle.market_data),
                        "samples": len(bundle.sample_data),
                        "trades": len(long_result.trades),
                        "dropped_no_execution": long_result.diagnostics.get("dropped_no_execution", 0),
                    }
                )

    print("SIMTRADING DATA VALIDATION")
    print(f"Config: {config_path}")
    for row in rows:
        print(
            f" - {row['owner']}: predictions={row['predictions']}, samples={row['samples']}, "
            f"market={row['market']}, trades={row['trades']}, "
            f"dropped_no_execution={row['dropped_no_execution']}"
        )

    if issues:
        print(f"[ISSUES] {len(issues)}")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print("No blocking format or backtest issues found.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SimTrading real-data compatibility.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "datasets.yaml",
        help="Path to SimTrading datasets.yaml.",
    )
    parser.add_argument("--dataset", type=str, default="", help="Validate only one dataset key.")
    args = parser.parse_args()
    return validate_config(args.config, args.dataset or None)


if __name__ == "__main__":
    raise SystemExit(main())
