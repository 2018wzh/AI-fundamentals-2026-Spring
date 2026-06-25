"""OilETF dataset builders (daily + intraday)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import ensure_dir, save_metadata

from runners.builders.common import (
    _build_echo_rows,
    _build_long_chronos_df,
    _plot_price_window,
)


def prepare_oiletf(samples_h60: Path, samples_h120: Path, daily_panel: Path, raw_prices: Path, output_dir: Path) -> None:
    frame = pd.read_parquet(daily_panel).copy()
    frame["date"] = pd.to_datetime(frame["end_date"]).dt.normalize()
    frame = frame.dropna(subset=["OT"]).reset_index(drop=True)

    baseline_cols = [
        "date",
        "uso_open",
        "uso_high",
        "uso_low",
        "uso_close",
        "uso_volume",
        "bno_ret_1d",
        "dbo_ret_1d",
        "wti_ret_1d",
        "brent_ret_1d",
        "brent_wti_spread",
        "dxy_change",
        "vix_change",
        "spy_ret_1d",
        "xle_ret_1d",
        "news_count",
        "news_sent_mean",
        "oil_event_count",
        "OT",
    ]
    baseline = frame[baseline_cols].copy().rename(
        columns={
            "uso_open": "open",
            "uso_high": "high",
            "uso_low": "low",
            "uso_close": "close",
            "uso_volume": "volume",
            "news_sent_mean": "sentiment_mean",
        }
    )
    # NOTE: return_1d column intentionally omitted — it was identical to OT,
    # creating data leakage for Chronos-2-ECHO (which uses features="MS").
    baseline["ma_5"] = baseline["close"].rolling(5, min_periods=1).mean()
    baseline["ma_20"] = baseline["close"].rolling(20, min_periods=1).mean()
    baseline["volatility_20"] = baseline["OT"].rolling(20, min_periods=2).std().fillna(0.0)
    baseline["news_agg"] = frame["news_agg"].fillna("No news available.")

    processed_dir = ensure_dir(output_dir / "processed")
    baseline_csv = processed_dir / "baseline_numeric.csv"
    baseline[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ma_5",
            "ma_20",
            "volatility_20",
            "news_count",
            "sentiment_mean",
            "oil_event_count",
            "OT",
        ]
    ].to_csv(baseline_csv, index=False)

    chronos = _build_long_chronos_df(
        baseline[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "ma_5",
                "ma_20",
                "volatility_20",
                "news_count",
                "sentiment_mean",
                "oil_event_count",
                "OT",
            ]
        ],
        item_id="OilETF",
    )
    chronos.to_parquet(processed_dir / "chronos_df.parquet", index=False)

    for history, name in [(60, "echo_H60_F1.csv"), (120, "echo_H120_F5.csv")]:
        echo = _build_echo_rows(
            baseline.assign(symbol="USO"),
            history=history,
            image_rel_dir="images/OilETF",
        )
        echo.to_csv(processed_dir / name, index=False)

    metadata = {
        "baseline_mode": "single_csv",
        "baseline_csv": str(baseline_csv),
        "chronos_df": str(processed_dir / "chronos_df.parquet"),
        "echo_H60_F1": str(processed_dir / "echo_H60_F1.csv"),
        "echo_H120_F5": str(processed_dir / "echo_H120_F5.csv"),
        "image_root": str(output_dir),
        "target_column": "OT",
        "enc_in": 12,
        "dec_in": 12,
        "c_out": 1,
        "features": "MS",
        "freq": "d",
    }
    save_metadata(processed_dir / "metadata.json", metadata)

    image_root = ensure_dir(output_dir / "images" / "OilETF")
    sample_windows: dict[int, set[str]] = {60: set(), 120: set()}
    for sample_path, history in [(samples_h60, 60), (samples_h120, 120)]:
        sample_df = pd.read_parquet(sample_path, columns=["end_date"])
        sample_windows[history] = {
            pd.to_datetime(value).strftime("%Y-%m-%d")
            for value in sample_df["end_date"].dropna().tolist()
        }

    price_df = pd.read_csv(raw_prices)
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
    uso = price_df[price_df["symbol"].astype(str).str.upper() == "USO"].copy()
    uso = uso.rename(columns={"timestamp": "date"}).sort_values("date").reset_index(drop=True)
    uso["ma_5"] = uso["close"].rolling(5, min_periods=1).mean()
    uso["ma_20"] = uso["close"].rolling(20, min_periods=1).mean()
    for history in [60, 120]:
        required_dates = sample_windows[history]
        for idx in range(history - 1, len(uso)):
            end_date = pd.to_datetime(uso.iloc[idx]["date"]).strftime("%Y-%m-%d")
            if end_date not in required_dates:
                continue
            output = image_root / f"USO_{end_date}_H{history}.png"
            if output.exists():
                continue
            _plot_price_window(uso.iloc[idx - history + 1 : idx + 1], output, f"USO {end_date} H{history}")


def prepare_oiletf_intraday(
    hourly_panel: Path,
    output_dir: Path,
) -> None:
    """Build OilETF intraday (hourly) evaluation data from pre-built panel.

    The intraday panel (3476 hourly bars, 2024-05-31 to 2026-05-29) already
    contains OHLCV, derived features, news aggregates, and OT target.  We
    produce:
      - baseline_numeric.csv   (for TSLib baselines)
      - chronos_df.parquet     (for Chronos-2)
      - echo_H60_F1.csv        (for Chronos-2-ECHO, 60h → 1h)
      - echo_H120_F7.csv       (for Chronos-2-ECHO, 120h → 7h)
      - metadata.json
    """
    frame = pd.read_parquet(hourly_panel).copy()
    # Use bar_start_utc as canonical timestamp — already UTC, no tz ambiguity
    frame["date"] = pd.to_datetime(frame["bar_start_utc"], utc=True).dt.tz_localize(None)
    frame = frame.dropna(subset=["OT"]).reset_index(drop=True)

    baseline = frame[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "uso_ret_7h",
            "vol_7h",
            "vol_20h",
            "ma_7h",
            "ma_20h",
            "rsi_14h",
            "news_count_1h",
            "news_count_6h",
            "news_sent_mean_6h",
            "oil_event_count_6h",
            "OT",
        ]
    ].copy().rename(columns={"news_sent_mean_6h": "sentiment_mean", "oil_event_count_6h": "oil_event_count"})

    # NOTE: uso_ret_1h intentionally excluded — it is identical to OT (hourly
    # return), creating data leakage for Chronos-2-ECHO (features="MS").
    baseline["news_agg"] = frame["news_agg_6h"].fillna("No news available.")

    processed_dir = ensure_dir(output_dir / "processed")
    baseline_csv = processed_dir / "baseline_numeric.csv"
    baseline[[
        "date",
        "open", "high", "low", "close", "volume",
        "uso_ret_7h", "vol_7h", "vol_20h",
        "ma_7h", "ma_20h", "rsi_14h",
        "news_count_1h", "news_count_6h",
        "sentiment_mean", "oil_event_count",
        "OT",
    ]].to_csv(baseline_csv, index=False)

    chronos = _build_long_chronos_df(
        baseline[[
            "date",
            "open", "high", "low", "close", "volume",
            "uso_ret_7h", "vol_7h", "vol_20h",
            "ma_7h", "ma_20h", "rsi_14h",
            "news_count_1h", "news_count_6h",
            "sentiment_mean", "oil_event_count",
            "OT",
        ]],
        item_id="OilETF",
    )
    chronos.to_parquet(processed_dir / "chronos_df.parquet", index=False)

    # TimeMMD echo CSVs — no images for intraday, so image_rel_dir=None.
    # _build_echo_rows expects daily-feature column names; map intraday names.
    echo_baseline = baseline.assign(symbol="USO").copy()
    echo_baseline["ma_5"] = echo_baseline["ma_7h"]
    echo_baseline["ma_20"] = echo_baseline["ma_20h"]
    echo_baseline["volatility_20"] = echo_baseline["vol_20h"]
    echo_baseline["news_count"] = echo_baseline["news_count_6h"]
    for history, forecast, name in [
        (60, 1, "echo_H60_F1.csv"),
        (120, 7, "echo_H120_F7.csv"),
    ]:
        echo = _build_echo_rows(
            echo_baseline,
            history=history,
            image_rel_dir=None,
        )
        echo.to_csv(processed_dir / name, index=False)

    n_features = len(baseline.columns) - 2  # minus date and OT
    metadata = {
        "baseline_mode": "single_csv",
        "baseline_csv": str(baseline_csv),
        "chronos_df": str(processed_dir / "chronos_df.parquet"),
        "echo_H60_F1": str(processed_dir / "echo_H60_F1.csv"),
        "echo_H120_F7": str(processed_dir / "echo_H120_F7.csv"),
        "image_root": str(output_dir),
        "target_column": "OT",
        "enc_in": n_features,
        "dec_in": n_features,
        "c_out": 1,
        "features": "MS",
        "freq": "h",
    }
    save_metadata(processed_dir / "metadata.json", metadata)
