from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import simtrading.config as config_module
from simtrading.config import ConfigError, load_app_config


class ConfigTests(unittest.TestCase):
    def test_missing_config_file_raises(self) -> None:
        with self.assertRaises(ConfigError):
            load_app_config(Path("missing-datasets.yaml"))

    def test_missing_path_in_dataset_config_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "datasets.yaml"
            config_path.write_text(
                """
datasets:
  oil_etf:
    display_name: "OilETF"
    panel_path: "missing.parquet"
    sample_paths:
      H60_F1: "missing_h60.parquet"
      H120_F5: "missing_h120.parquet"
    date_column: "end_date"
    symbol_column: "symbol"
    price_columns:
      execution_price: "open"
      mark_price: "close"
      benchmark_price: "close"
    prediction_result_paths:
      DLinear: "missing_predictions.parquet"
    benchmark_symbols: ["USO"]
""",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_app_config(config_path)

    def test_relative_paths_resolve_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for path in ["data/panel.csv", "data/samples.csv", "predictions/model.csv"]:
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x\n", encoding="utf-8")
            config_path = root / "config" / "datasets.yaml"
            config_path.parent.mkdir()
            config_path.write_text(
                """
datasets:
  oiletf:
    display_name: "OilETF"
    panel_path: "data/panel.csv"
    sample_paths:
      H60_F1: "data/samples.csv"
    date_column: "date"
    symbol_column: "symbol"
    price_columns:
      execution_price: "open"
      mark_price: "close"
      benchmark_price: "close"
    prediction_result_paths:
      Chronos-2: "predictions/model.csv"
    benchmark_symbols: ["OilETF"]
""",
                encoding="utf-8",
            )

            with patch.object(config_module, "PROJECT_ROOT", root):
                app_config = load_app_config(config_path)

            dataset = app_config.datasets["oiletf"]
            self.assertEqual(root / "data" / "panel.csv", dataset.panel_path)
            self.assertEqual(root / "predictions" / "model.csv", dataset.prediction_result_paths["Chronos-2"])


if __name__ == "__main__":
    unittest.main()
