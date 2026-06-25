from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from simtrading.backtest import run_backtest
from simtrading.config import load_app_config
from simtrading.data_loader import load_dataset
from simtrading.strategy import StrategyConfig, StrategyError, StrategyName


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


class StrategyBacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)

        panel = pd.DataFrame(
            {
                "end_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                "symbol": ["AAA", "AAA", "AAA", "AAA", "AAA"],
                "open": [100, 101, 103, 102, 105],
                "close": [101, 103, 102, 105, 106],
            }
        )
        samples = pd.DataFrame(
            {
                "sample_id": ["AAA_2024-01-01_H60_F1", "AAA_2024-01-02_H60_F1", "AAA_2024-01-03_H60_F1"],
                "symbol": ["AAA", "AAA", "AAA"],
                "end_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "H": [60, 60, 60],
                "F": [1, 1, 1],
                "split": ["test", "test", "test"],
            }
        )
        predictions = pd.DataFrame(
            {
                "dataset": ["oil_etf", "oil_etf", "oil_etf"],
                "model": ["DLinear", "DLinear", "DLinear"],
                "symbol": ["AAA", "AAA", "AAA"],
                "end_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "window": [60, 60, 60],
                "horizon": [1, 1, 1],
                "y_true": [0.01, 0.02, -0.01],
                "y_pred": [0.02, 0.03, -0.02],
                "split": ["test", "test", "test"],
            }
        )

        self.panel_path = root / "panel.parquet"
        self.samples_h60 = root / "samples_H60_F1.parquet"
        self.samples_h120 = root / "samples_H120_F5.parquet"
        self.predictions_path = root / "predictions.parquet"
        _write_parquet(panel, self.panel_path)
        _write_parquet(samples, self.samples_h60)
        _write_parquet(samples.iloc[:1].assign(H=120, F=5), self.samples_h120)
        _write_parquet(predictions, self.predictions_path)

        self.config_path = root / "datasets.yaml"
        self.config_path.write_text(
            f"""
datasets:
  oil_etf:
    display_name: "OilETF"
    panel_path: "{self.panel_path.as_posix()}"
    sample_paths:
      H60_F1: "{self.samples_h60.as_posix()}"
      H120_F5: "{self.samples_h120.as_posix()}"
    date_column: "end_date"
    symbol_column: "symbol"
    price_columns:
      execution_price: "open"
      mark_price: "close"
      benchmark_price: "close"
    prediction_result_paths:
      DLinear: "{self.predictions_path.as_posix()}"
    benchmark_symbols: ["AAA"]
""",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_backtest_runs_for_long_cash(self) -> None:
        app_config = load_app_config(self.config_path)
        bundle = load_dataset(app_config, "oil_etf", "H60_F1", "DLinear")
        result = run_backtest(
            bundle,
            StrategyConfig(
                name=StrategyName.LONG_CASH,
                threshold=0.0,
                fee_bps=0.0,
                slippage_bps=0.0,
                top_k=1,
                max_position=1.0,
                score_column="y_pred",
            ),
        )
        self.assertGreater(len(result.trades), 0)
        self.assertIn("equity", result.equity_curve.columns)

    def test_confidence_strategy_requires_quantiles(self) -> None:
        app_config = load_app_config(self.config_path)
        bundle = load_dataset(app_config, "oil_etf", "H60_F1", "DLinear")
        with self.assertRaises(StrategyError):
            run_backtest(
                bundle,
                StrategyConfig(
                    name=StrategyName.CONFIDENCE_LONG_CASH,
                    threshold=0.0,
                    fee_bps=0.0,
                    slippage_bps=0.0,
                    top_k=1,
                    max_position=1.0,
                    score_column="y_pred",
                ),
            )

    def test_backtest_tolerates_duplicate_market_timestamps(self) -> None:
        panel = pd.read_parquet(self.panel_path)
        _write_parquet(pd.concat([panel, panel.iloc[[1]]], ignore_index=True), self.panel_path)
        app_config = load_app_config(self.config_path)
        bundle = load_dataset(app_config, "oil_etf", "H60_F1", "DLinear")

        result = run_backtest(
            bundle,
            StrategyConfig(
                name=StrategyName.LONG_CASH,
                threshold=0.0,
                fee_bps=0.0,
                slippage_bps=0.0,
                top_k=1,
                max_position=1.0,
                score_column="y_pred",
            ),
        )

        self.assertGreater(len(result.trades), 0)


if __name__ == "__main__":
    unittest.main()
