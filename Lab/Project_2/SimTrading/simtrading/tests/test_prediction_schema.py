from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from simtrading.prediction_schema import PredictionSchemaError, load_predictions, validate_prediction_schema


def _base_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["oil_etf"],
            "model": ["DLinear"],
            "symbol": ["USO"],
            "end_date": ["2024-01-02"],
            "window": [60],
            "horizon": [1],
            "y_true": [0.01],
            "y_pred": [0.02],
            "split": ["test"],
        }
    )


class PredictionSchemaTests(unittest.TestCase):
    def test_missing_required_column_raises(self) -> None:
        df = _base_prediction_frame().drop(columns=["y_pred"])
        with self.assertRaises(PredictionSchemaError):
            validate_prediction_schema(df)

    def test_invalid_date_raises(self) -> None:
        df = _base_prediction_frame()
        df.loc[0, "end_date"] = "not-a-date"
        with self.assertRaises(PredictionSchemaError):
            validate_prediction_schema(df)

    def test_quantile_requirement_raises(self) -> None:
        df = _base_prediction_frame()
        with self.assertRaises(PredictionSchemaError):
            validate_prediction_schema(df, require_quantiles=True)

    def test_load_predictions_reads_csv(self) -> None:
        df = _base_prediction_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            df.to_csv(path, index=False)
            loaded = load_predictions(path)
        self.assertEqual(len(loaded), 1)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded["end_date"]))


if __name__ == "__main__":
    unittest.main()
