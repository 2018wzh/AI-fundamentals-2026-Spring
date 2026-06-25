#!/usr/bin/env python3
"""Convert Chronos-2-ECHO evaluation predictions to SimTrading-compatible format.

Runner output schema:
  dataset, model, setting, series_id, window_id, horizon_idx, target_idx,
  timestamp, y_true, y_pred, q10, q50, q90

SimTrading expected schema:
  dataset, model, symbol, end_date, window, horizon, y_true, y_pred,
  split[, q10, q50, q90]

Usage:
  python scripts/convert_for_simtrading.py \\
    --predictions results/oiletf/Chronos-2-ECHO/H60_F1_text_only/predictions.parquet \\
    --samples /home/zhw/OilETF-TimeMMD/data/processed/samples_H60_F1.parquet \\
    --panel /home/zhw/Project_2/data/oiletf/processed/baseline_numeric.csv \\
    --date-col date \\
    --output simtrading_predictions.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert runner predictions to SimTrading format")
    parser.add_argument("--predictions", required=True, type=Path, help="Runner predictions.parquet")
    parser.add_argument("--samples", required=True, type=Path, help="Sample metadata parquet (e.g. samples_H60_F1.parquet)")
    parser.add_argument("--panel", type=Path, default=None, help="Panel/market data CSV or parquet (for end_date lookup)")
    parser.add_argument("--date-col", type=str, default="date", help="Date column name in panel (default: date)")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet path for SimTrading")
    parser.add_argument("--model-name", type=str, default=None, help="Override model name in output")
    return parser


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"series_id", "window_id", "horizon_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {missing}")
    return df


def load_samples(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"sample_id", "symbol", "end_date", "H", "F", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sample file missing columns: {missing}")
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def guess_setting(predictions: pd.DataFrame) -> tuple[int, int]:
    setting_val = predictions["setting"].iloc[0]
    parts = setting_val.split("_")
    history = int(parts[0].removeprefix("H"))
    forecast = int(parts[1].removeprefix("F"))
    return history, forecast


def convert(
    predictions: pd.DataFrame,
    samples: pd.DataFrame,
    panel: pd.DataFrame | None = None,
    date_col: str = "date",
    model_name: str | None = None,
) -> pd.DataFrame:
    history, forecast = guess_setting(predictions)

    # Filter samples to match this setting
    matching_samples = samples[(samples["H"] == history) & (samples["F"] == forecast)].copy()
    if matching_samples.empty:
        raise ValueError(f"No sample rows match H={history} F={forecast}")

    # Build series_id → symbol mapping from sample data
    # sample_id format: {SYMBOL}_{end_date}_H{history}_F{forecast}
    # sample_id formats:
    #   daily:   {SYMBOL}_{YYYY-MM-DD}_H{history}_F{forecast}
    #   intraday:{SYMBOL}_{ISO_TIMESTAMP}_H{history}_F{forecast}
    matching_samples["_series_id"] = matching_samples["sample_id"].str.extract(
        rf"^(.+?)_\d{{4}}-\d{{2}}-\d{{2}}(?:T\d{{2}}:\d{{2}}:\d{{2}}[+-]\d{{2}}:\d{{2}})?_H{history}_F{forecast}$"
    )[0]
    series_symbol_map = dict(zip(matching_samples["_series_id"], matching_samples["symbol"]))

    # Map series_id → symbol
    out = predictions.copy()
    out["symbol"] = out["series_id"].map(series_symbol_map).fillna(out["series_id"])

    # Aggregate to per-window-end_date (horizon_idx=0 → end_date)
    out["window"] = out["window_id"].astype(int)
    out["horizon"] = out["horizon_idx"].astype(int) + 1  # 0-indexed → 1-indexed

    # Build end_date lookup from samples
    # samples have unique (symbol, end_date) per window
    sample_dates = matching_samples[["symbol", "end_date"]].drop_duplicates()

    # If panel is provided, build a more complete end_date mapping
    if panel is not None:
        panel_dates = panel.copy()
        panel_dates[date_col] = pd.to_datetime(panel_dates[date_col])
        # For fnspid: panel has symbol column; for oiletf: single series
        if "symbol" in panel_dates.columns:
            date_map = panel_dates[["symbol", date_col]].drop_duplicates()
        else:
            date_map = panel_dates[[date_col]].copy()
            date_map["symbol"] = out["symbol"].iloc[0]

    # Assign split from sample data
    # Each (symbol, end_date) combination maps to a specific window
    # We use the sample data directly
    split_map = dict(
        zip(
            zip(matching_samples["symbol"], matching_samples["end_date"].dt.strftime("%Y-%m-%d")),
            matching_samples["split"],
        )
    )

    # Use sample dates for end_date assignment
    # For each series, the end_date = first test window date + window_id steps
    # Since horizon_idx=0 corresponds to the first prediction step, we need end_date = sample_end_date[window_id]
    for symbol in out["symbol"].unique():
        sym_samples = matching_samples[matching_samples["symbol"] == symbol].sort_values("end_date")
        sym_dates = sym_samples["end_date"].dt.strftime("%Y-%m-%d").tolist()
        sym_splits = sym_samples["split"].tolist()
        mask = out["symbol"] == symbol
        for w in out.loc[mask, "window"].unique():
            widx = int(w)
            if widx < len(sym_dates):
                out.loc[mask & (out["window"] == w), "end_date"] = sym_dates[widx]
                out.loc[mask & (out["window"] == w), "split"] = sym_splits[widx]

    # Fill any missing end_date/split
    out["end_date"] = out["end_date"].fillna("1970-01-01")
    out["split"] = out["split"].fillna("test")

    # Select and rename columns
    model = model_name or out["model"].iloc[0]
    keep = {
        "dataset": "dataset",
        "symbol": "symbol",
        "end_date": "end_date",
        "window": "window",
        "horizon": "horizon",
        "y_true": "y_true",
        "y_pred": "y_pred",
        "split": "split",
    }
    if "q10" in out.columns:
        keep["q10"] = "q10"
    if "q50" in out.columns:
        keep["q50"] = "q50"
    if "q90" in out.columns:
        keep["q90"] = "q90"

    result = out[list(keep.keys())].rename(columns=keep)
    result["model"] = model
    result["end_date"] = pd.to_datetime(result["end_date"], errors="coerce")
    return result.sort_values(["symbol", "end_date", "window", "horizon"]).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()

    predictions = load_predictions(args.predictions)
    samples = load_samples(args.samples)
    panel = None
    if args.panel:
        suffix = args.panel.suffix.lower()
        if suffix == ".csv":
            panel = pd.read_csv(args.panel)
        else:
            panel = pd.read_parquet(args.panel)

    result = convert(predictions, samples, panel=panel, date_col=args.date_col, model_name=args.model_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"Converted {len(result)} rows → {args.output}")
    print(f"  symbols: {sorted(result['symbol'].unique())}")
    print(f"  splits: {result['split'].value_counts().to_dict()}")
    print(f"  horizons: {sorted(result['horizon'].unique())}")


if __name__ == "__main__":
    main()
