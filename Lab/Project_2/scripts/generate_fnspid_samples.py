#!/usr/bin/env python3
"""Generate FNSPID sample index for SimTrading.

Creates sample_H60_F1.parquet and sample_H120_F5.parquet in the FNSPID
processed directory.  These files describe every test/val/train sliding
window so SimTrading can align predictions with the correct end_date
and split.

Produces the same schema as OilETF-TimeMMD's samples_*.parquet:
  sample_id, symbol, end_date, H, F, split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from runners.common import rolling_test_starts


def generate_samples(
    panel_path: Path,
    output_dir: Path,
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> None:
    panel = pd.read_parquet(panel_path)
    panel[date_col] = pd.to_datetime(panel[date_col])
    symbols = sorted(panel[symbol_col].unique())

    for history, forecast in [(60, 1), (120, 5)]:
        rows = []
        for symbol in symbols:
            series = panel[panel[symbol_col] == symbol].sort_values(date_col).reset_index(drop=True)
            dates = series[date_col].tolist()
            n = len(series)
            for w in rolling_test_starts(n, history, forecast):
                if w + history + forecast > n:
                    continue
                end_idx = w + history - 1
                end_date = dates[end_idx]
                # Determine split
                num_train = int(n * 0.7)
                num_val = int(n * 0.2)
                test_start = max(0, n - num_val - num_test_fraction(n))
                val_start = num_train - history
                test_start_adj = max(0, n - int(n * 0.2) - history)
                # Simple train/val/test assignment
                if w >= test_start_adj:
                    split = "test"
                elif w >= max(0, int(n * 0.7) - history):
                    split = "val"
                else:
                    split = "train"

                sample_id = f"{symbol}_{end_date.strftime('%Y-%m-%d')}_H{history}_F{forecast}"
                rows.append({
                    "sample_id": sample_id,
                    "symbol": symbol,
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "H": history,
                    "F": forecast,
                    "split": split,
                })

        df = pd.DataFrame(rows)
        out_path = output_dir / f"samples_H{history}_F{forecast}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Generated {len(df)} samples → {out_path}")
        print(f"  split distribution: {df['split'].value_counts().to_dict()}")


def num_test_fraction(n: int) -> int:
    return int(n * 0.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FNSPID sample index for SimTrading")
    parser.add_argument("--panel", required=True, type=Path, help="FNSPID panel parquet")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--symbol-col", default="symbol")
    parser.add_argument("--date-col", default="date")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_samples(
        panel_path=args.panel,
        output_dir=args.output_dir,
        symbol_col=args.symbol_col,
        date_col=args.date_col,
    )


if __name__ == "__main__":
    main()
