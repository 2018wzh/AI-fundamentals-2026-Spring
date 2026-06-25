"""Electricity dataset builder (hourly electricity load)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import ensure_dir, save_metadata

from runners.builders.common import _detect_column


def prepare_electricity(source_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(source_csv)
    date_col = _detect_column(df.columns, ["date", "datetime", "timestamp"])
    numeric_cols = [col for col in df.columns if col != date_col and pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        raise ValueError(f"No numeric columns found in {source_csv}")

    target_column = "OT" if "OT" in numeric_cols else numeric_cols[-1]
    ordered_cols = [date_col] + [col for col in numeric_cols if col != target_column] + [target_column]
    df = df[ordered_cols].copy().rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])

    ensure_dir(output_dir)
    csv_path = output_dir / "electricity.csv"
    df.to_csv(csv_path, index=False)

    chronos = df.melt(id_vars="date", var_name="item_id", value_name="target").rename(columns={"date": "timestamp"})
    chronos["timestamp"] = pd.to_datetime(chronos["timestamp"])
    chronos.to_parquet(output_dir / "chronos_df.parquet", index=False)

    metadata = {
        "baseline_mode": "single_csv",
        "baseline_csv": str(csv_path),
        "chronos_df": str(output_dir / "chronos_df.parquet"),
        "target_column": target_column,
        "numeric_columns": numeric_cols,
        "enc_in": len(numeric_cols),
        "dec_in": len(numeric_cols),
        "c_out": len(numeric_cols),
        "features": "M",
        "freq": "h",
    }
    save_metadata(output_dir / "metadata.json", metadata)
