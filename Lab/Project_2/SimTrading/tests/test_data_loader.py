from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import simtrading.data_loader as data_loader
from simtrading.config import DatasetConfig, PriceColumns


class DataLoaderTests(unittest.TestCase):
    def test_filter_predictions_prefers_setting_column(self) -> None:
        predictions = pd.DataFrame(
            {
                "dataset": ["oiletf", "oiletf"],
                "model": ["Chronos-2", "Chronos-2"],
                "symbol": ["OilETF", "OilETF"],
                "end_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "setting": ["H60_F1", "H120_F5"],
                "window": [0, 0],
                "horizon": [0, 0],
                "y_true": [0.1, 0.2],
                "y_pred": [0.1, 0.2],
                "split": ["test", "test"],
            }
        )

        out = data_loader._filter_predictions(predictions, "oiletf", "H60_F1", "Chronos-2")

        self.assertEqual(1, len(out))
        self.assertEqual("2026-01-01", out["end_date"].dt.strftime("%Y-%m-%d").iloc[0])

    def test_normalize_sample_data_accepts_echo_rows_with_end_date(self) -> None:
        samples = pd.DataFrame({"date": ["2026-01-01"], "end_date": ["2026-01-02"], "OT": [0.1]})

        out = data_loader._normalize_sample_data(samples)

        self.assertEqual(pd.Timestamp("2026-01-02"), out["end_date"].iloc[0])

    def test_normalize_sample_data_accepts_manifest_with_legacy_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "SimTrading"
            series_path = project_root.parent / "data" / "fnspid" / "echo" / "H60_F1" / "AAA.csv"
            series_path.parent.mkdir(parents=True)
            pd.DataFrame({"end_date": ["2026-01-03"]}).to_csv(series_path, index=False)
            manifest = pd.DataFrame(
                {
                    "series_id": ["AAA"],
                    "csv_path": ["/home/zhw/Project_2/data/fnspid/echo/H60_F1/AAA.csv"],
                }
            )

            with patch.object(data_loader, "PROJECT_ROOT", project_root):
                out = data_loader._normalize_sample_data(manifest)

        self.assertEqual(["AAA"], out["symbol"].tolist())
        self.assertEqual(pd.Timestamp("2026-01-03"), out["end_date"].iloc[0])

    def test_market_without_symbol_uses_configured_benchmark_symbol(self) -> None:
        dataset_cfg = DatasetConfig(
            name="oiletf",
            display_name="OilETF",
            panel_path=Path("panel.csv"),
            sample_paths={},
            date_column="date",
            symbol_column="symbol",
            price_columns=PriceColumns("open", "close", "close"),
            prediction_result_paths={},
            benchmark_symbols=["OilETF"],
        )
        market = pd.DataFrame({"date": ["2026-01-01"], "open": [1.0], "close": [1.1]})

        out = data_loader._normalize_market_data(market, dataset_cfg)

        self.assertEqual(["OilETF"], out["symbol"].tolist())

    def test_single_symbol_market_aligns_to_prediction_symbol(self) -> None:
        market = pd.DataFrame({"symbol": ["OilETF", "OilETF"], "date": pd.to_datetime(["2026-01-01", "2026-01-02"])})
        predictions = pd.DataFrame({"symbol": ["oiletf"]})

        out = data_loader._align_single_symbol_market(market, predictions, "symbol")

        self.assertEqual(["oiletf"], sorted(out["symbol"].unique()))


if __name__ == "__main__":
    unittest.main()
