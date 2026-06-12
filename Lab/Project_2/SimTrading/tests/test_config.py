from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
    dataset_label: "oiletf"
    panel:
      path: "missing.parquet"
      date_column: "end_date"
      symbol_mode: "constant"
      constant_symbol: "oiletf"
      price_columns:
        execution_price: "open"
        mark_price: "close"
        benchmark_price: "close"
    samples:
      format: "metadata_table"
      symbol_mode: "constant"
      constant_symbol: "oiletf"
      paths:
        H60_F1: "missing_h60.parquet"
        H120_F5: "missing_h120.parquet"
    predictions:
      require_setting: true
      signal_horizons:
        H60_F1: 0
        H120_F5: 4
      result_paths:
        DLinear: "missing_predictions.parquet"
    benchmark_symbols: ["USO"]
""",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_app_config(config_path)


if __name__ == "__main__":
    unittest.main()
