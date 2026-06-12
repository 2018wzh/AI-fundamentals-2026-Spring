from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import ensure_dir, rolling_test_starts, save_metadata, write_json


PRICE_COLUMN_CANDIDATES = {
    "date": ["date", "timestamp", "datetime", "time"],
    "open": ["open", "Open", "adj_open"],
    "high": ["high", "High", "adj_high"],
    "low": ["low", "Low", "adj_low"],
    "close": ["close", "Close", "adj_close", "adjusted_close"],
    "volume": ["volume", "Volume", "vol"],
    "symbol": ["symbol", "ticker", "stock", "stock_symbol"],
}

NEWS_COLUMN_CANDIDATES = {
    "symbol": ["symbol", "ticker", "stock_symbol", "stock", "company_symbol", "Stock_symbol"],
    "date": ["date", "publish_date", "datetime", "time", "created_at", "Date"],
    "title": ["title", "headline", "news_title", "article_title", "Article_title"],
    "summary": ["summary", "description", "news_text", "article", "content", "body", "Article"],
    "sentiment": ["sentiment", "sentiment_score", "score", "label_score"],
}


def _maybe_import_hf():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for asset download. Install it in the controller .venv first."
        ) from exc
    return hf_hub_download


def _detect_column(columns: Iterable[str], candidates: list[str], *, required: bool = True) -> str | None:
    lowered = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise ValueError(f"Could not find any of columns {candidates} in {list(columns)}")
    return None


def _prepare_price_frame(df: pd.DataFrame, fallback_symbol: str | None = None) -> pd.DataFrame:
    date_col = _detect_column(df.columns, PRICE_COLUMN_CANDIDATES["date"])
    open_col = _detect_column(df.columns, PRICE_COLUMN_CANDIDATES["open"])
    high_col = _detect_column(df.columns, PRICE_COLUMN_CANDIDATES["high"])
    low_col = _detect_column(df.columns, PRICE_COLUMN_CANDIDATES["low"])
    close_col = _detect_column(df.columns, PRICE_COLUMN_CANDIDATES["close"])
    volume_col = _detect_column(df.columns, PRICE_COLUMN_CANDIDATES["volume"], required=False)
    symbol_col = _detect_column(df.columns, PRICE_COLUMN_CANDIDATES["symbol"], required=False)

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]).dt.normalize(),
            "open": pd.to_numeric(df[open_col], errors="coerce"),
            "high": pd.to_numeric(df[high_col], errors="coerce"),
            "low": pd.to_numeric(df[low_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "volume": pd.to_numeric(df[volume_col], errors="coerce") if volume_col else 0.0,
        }
    )
    if symbol_col:
        frame["symbol"] = df[symbol_col].astype(str).str.upper()
    elif fallback_symbol:
        frame["symbol"] = fallback_symbol.upper()
    else:
        raise ValueError("A ticker/symbol is required for FNSPID price preparation.")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    return frame


def _compute_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy().sort_values("date")
    frame["OT"] = frame["close"].pct_change()
    frame["return_1d"] = frame["OT"]
    frame["ma_5"] = frame["close"].rolling(5, min_periods=1).mean()
    frame["ma_20"] = frame["close"].rolling(20, min_periods=1).mean()
    frame["volatility_20"] = frame["OT"].rolling(20, min_periods=2).std().fillna(0.0)
    return frame


def _plot_price_window(window: pd.DataFrame, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    x = np.arange(len(window))
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=96)
    color = np.where(window["close"].to_numpy() >= window["open"].to_numpy(), "green", "red")
    ax.vlines(x, window["low"], window["high"], color="black", linewidth=0.5, alpha=0.8)
    ax.bar(x, window["close"] - window["open"], bottom=window["open"], color=color, width=0.4, alpha=0.65)
    ax.plot(x, window["close"], color="navy", linewidth=1.0, alpha=0.7)
    if "ma_5" in window:
        ax.plot(x, window["ma_5"], color="orange", linewidth=0.8)
    if "ma_20" in window:
        ax.plot(x, window["ma_20"], color="purple", linewidth=0.8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _build_long_chronos_df(frame: pd.DataFrame, *, item_id: str) -> pd.DataFrame:
    chronos = frame.rename(columns={"date": "timestamp"}).copy()
    if "OT" in chronos.columns and "target" not in chronos.columns:
        chronos = chronos.rename(columns={"OT": "target"})
    chronos.insert(0, "item_id", item_id)
    return chronos


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


def _download_hf_file(repo_id: str, filename: str, repo_type: str, local_dir: Path) -> Path:
    hf_hub_download = _maybe_import_hf()
    ensure_dir(local_dir)
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=local_dir)
    return Path(downloaded)


def _unzip_if_needed(zip_path: Path, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    marker = output_dir / ".extracted"
    if marker.exists():
        return output_dir
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    marker.write_text("ok", encoding="utf-8")
    return output_dir


def _normalize_news_date(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _summarize_news(news_file: Path) -> pd.DataFrame:
    with news_file.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)

    symbol_col = _detect_column(header, NEWS_COLUMN_CANDIDATES["symbol"])
    date_col = _detect_column(header, NEWS_COLUMN_CANDIDATES["date"])
    title_col = _detect_column(header, NEWS_COLUMN_CANDIDATES["title"], required=False)
    summary_col = _detect_column(header, NEWS_COLUMN_CANDIDATES["summary"], required=False)
    sentiment_col = _detect_column(header, NEWS_COLUMN_CANDIDATES["sentiment"], required=False)

    buckets: dict[tuple[str, str], dict[str, object]] = {}

    with news_file.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = str(row.get(symbol_col, "")).strip().upper()
            date_str = _normalize_news_date(row.get(date_col))
            if not symbol or not date_str:
                continue

            title = str(row.get(title_col, "")).strip() if title_col else ""
            summary = str(row.get(summary_col, "")).strip() if summary_col else ""
            text = f"{title} {summary}".strip()

            sentiment = np.nan
            if sentiment_col:
                raw_sentiment = row.get(sentiment_col)
                if raw_sentiment not in (None, ""):
                    try:
                        sentiment = float(raw_sentiment)
                    except (TypeError, ValueError):
                        sentiment = np.nan

            key = (symbol, date_str)
            entry = buckets.setdefault(
                key,
                {
                    "symbol": symbol,
                    "date": date_str,
                    "news_count": 0,
                    "sentiment_sum": 0.0,
                    "sentiment_n": 0,
                    "texts": [],
                },
            )
            entry["news_count"] = int(entry["news_count"]) + 1
            if text and len(entry["texts"]) < 16:
                entry["texts"].append(text)
            if pd.notna(sentiment):
                entry["sentiment_sum"] = float(entry["sentiment_sum"]) + float(sentiment)
                entry["sentiment_n"] = int(entry["sentiment_n"]) + 1

    rows = []
    for value in buckets.values():
        sentiment_n = int(value["sentiment_n"])
        rows.append(
            {
                "symbol": value["symbol"],
                "date": value["date"],
                "news_count": int(value["news_count"]),
                "sentiment_mean": float(value["sentiment_sum"]) / sentiment_n if sentiment_n else 0.0,
                "news_agg": " ".join(value["texts"]) if value["texts"] else "No news available.",
            }
        )
    return pd.DataFrame(rows)


def _choose_top_symbols(news_daily: pd.DataFrame, top_k: int) -> list[str]:
    ranked = news_daily.groupby("symbol")["news_count"].sum().sort_values(ascending=False)
    return ranked.head(top_k).index.tolist()


def _normalize_news_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    normalized["news_count"] = pd.to_numeric(normalized["news_count"], errors="coerce").fillna(0).astype(int)
    normalized["sentiment_mean"] = pd.to_numeric(normalized["sentiment_mean"], errors="coerce").fillna(0.0)
    normalized["news_agg"] = normalized["news_agg"].fillna("No news available.").astype(str)
    normalized = normalized.dropna(subset=["date"]).reset_index(drop=True)
    return normalized


def _find_price_csvs(price_root: Path) -> list[Path]:
    return [path for path in price_root.rglob("*.csv") if path.is_file()]


def _resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve any of: {[str(path) for path in candidates]}")


def _load_symbol_price(price_root: Path, symbol: str) -> pd.DataFrame:
    candidates = [path for path in _find_price_csvs(price_root) if symbol.lower() in path.stem.lower()]
    if not candidates:
        raise FileNotFoundError(f"Could not find price csv for {symbol} under {price_root}")
    for candidate in candidates:
        try:
            raw = pd.read_csv(candidate)
            frame = _prepare_price_frame(raw, fallback_symbol=symbol)
            if not frame.empty:
                return frame[frame["symbol"] == symbol]
        except Exception:
            continue
    raise ValueError(f"Could not normalize any price file for {symbol}")


def _build_start_dates(dates: pd.Series, history: int) -> list[str]:
    values = pd.to_datetime(dates).tolist()
    starts = []
    for idx in range(len(values)):
        start_idx = max(0, idx - history + 1)
        starts.append(values[start_idx].strftime("%Y-%m-%d"))
    return starts


def _build_echo_rows(frame: pd.DataFrame, *, history: int, image_rel_dir: str | None = None) -> pd.DataFrame:
    echo = frame.copy()
    echo["date"] = pd.to_datetime(echo["date"]).dt.strftime("%Y-%m-%d")
    echo["end_date"] = echo["date"]
    echo["start_date"] = _build_start_dates(pd.to_datetime(frame["date"]), history)
    echo["prior_history_avg"] = echo["OT"].rolling(history, min_periods=1).mean().fillna(0.0)
    echo["fact"] = echo["news_agg"].fillna("No news available.").astype(str).replace("", "No news available.")
    if image_rel_dir is not None:
        echo["image_path"] = [
            f"{image_rel_dir}/{symbol}_{date}_H{history}.png" for symbol, date in zip(echo["symbol"], echo["date"])
        ]
    keep_cols = [
        "date",
        "fact",
        "prior_history_avg",
        "start_date",
        "end_date",
        "OT",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "ma_5",
        "ma_20",
        "volatility_20",
        "news_count",
        "sentiment_mean",
    ]
    if "image_path" in echo.columns:
        keep_cols.append("image_path")
    return echo[keep_cols]


def _generate_images(
    frame: pd.DataFrame,
    image_root: Path,
    symbol: str,
    windows: list[int],
    required_dates: dict[int, set[str]] | None = None,
) -> None:
    frame = frame.sort_values("date").copy()
    for history in windows:
        history_required_dates = required_dates.get(history, set()) if required_dates else None
        for idx in range(history - 1, len(frame)):
            end_date = pd.to_datetime(frame.iloc[idx]["date"]).strftime("%Y-%m-%d")
            if history_required_dates is not None and end_date not in history_required_dates:
                continue
            out_path = image_root / f"{symbol}_{end_date}_H{history}.png"
            if out_path.exists():
                continue
            window = frame.iloc[idx - history + 1 : idx + 1]
            _plot_price_window(window, out_path, f"{symbol} {end_date} H{history}")


def prepare_fnspid(
    *,
    repo_root: Path,
    output_dir: Path,
    top_k: int,
    repo_id: str,
    price_filename: str,
    news_filename: str,
    download: bool,
    skip_images: bool,
) -> None:
    raw_dir = ensure_dir(repo_root / "raw")
    processed_dir = ensure_dir(output_dir / "processed")
    if download:
        price_archive = _download_hf_file(repo_id, price_filename, "dataset", raw_dir)
        news_file = _download_hf_file(repo_id, news_filename, "dataset", raw_dir)
    else:
        price_archive = _resolve_existing_path(
            raw_dir / Path(price_filename).name,
            raw_dir / price_filename,
        )
        news_file = _resolve_existing_path(
            raw_dir / Path(news_filename).name,
            raw_dir / news_filename,
        )
    price_root = _unzip_if_needed(price_archive, raw_dir / "prices")

    news_cache = processed_dir / "news_daily.parquet"
    if news_cache.exists():
        news_daily = pd.read_parquet(news_cache)
    else:
        news_daily = _summarize_news(news_file)
        news_daily.to_parquet(news_cache, index=False)
    news_daily = _normalize_news_daily_frame(news_daily)
    top_symbols = _choose_top_symbols(news_daily, top_k)

    baseline_root = ensure_dir(output_dir / "baseline")
    echo_root = ensure_dir(output_dir / "echo")
    image_root = ensure_dir(output_dir / "images")

    panel_frames = []
    chronos_frames = []
    baseline_manifest = []
    echo_manifest_60 = []
    echo_manifest_120 = []

    for symbol in top_symbols:
        try:
            price = _load_symbol_price(price_root, symbol)
        except (FileNotFoundError, ValueError):
            continue
        if price.empty:
            continue
        price = _compute_features(price)
        merged = price.merge(news_daily[news_daily["symbol"] == symbol], how="left", on=["symbol", "date"])
        merged["news_count"] = merged["news_count"].fillna(0).astype(int)
        merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)
        merged["news_agg"] = merged["news_agg"].fillna("No news available.")
        merged = merged.dropna(subset=["OT"]).reset_index(drop=True)
        if len(merged) < 140:
            continue

        baseline_csv = baseline_root / f"{symbol}.csv"
        baseline_frame = merged[
            ["date", "open", "high", "low", "close", "volume", "return_1d", "ma_5", "ma_20", "volatility_20", "news_count", "sentiment_mean", "OT"]
        ].copy()
        baseline_frame.to_csv(baseline_csv, index=False)

        echo60_dir = ensure_dir(echo_root / "H60_F1")
        echo120_dir = ensure_dir(echo_root / "H120_F5")
        echo60_csv = echo60_dir / f"{symbol}.csv"
        echo120_csv = echo120_dir / f"{symbol}.csv"
        _build_echo_rows(merged, history=60, image_rel_dir=f"images/{symbol}").to_csv(echo60_csv, index=False)
        _build_echo_rows(merged, history=120, image_rel_dir=f"images/{symbol}").to_csv(echo120_csv, index=False)

        panel = baseline_frame.copy()
        panel.insert(0, "symbol", symbol)
        panel_frames.append(panel)
        chronos_frames.append(_build_long_chronos_df(baseline_frame, item_id=symbol))

        baseline_manifest.append({"series_id": symbol, "csv_path": str(baseline_csv)})
        echo_manifest_60.append({"series_id": symbol, "csv_path": str(echo60_csv)})
        echo_manifest_120.append({"series_id": symbol, "csv_path": str(echo120_csv)})

        if not skip_images:
            required_image_dates: dict[int, set[str]] = {}
            for history, forecast in [(60, 1), (120, 5)]:
                required_image_dates[history] = {
                    pd.to_datetime(merged.iloc[start + history - 1]["date"]).strftime("%Y-%m-%d")
                    for start in rolling_test_starts(len(merged), history, forecast)
                    if start + history - 1 < len(merged)
                }
            _generate_images(merged, image_root / symbol, symbol, [60, 120], required_dates=required_image_dates)

    if not panel_frames:
        raise RuntimeError("No FNSPID symbol series were prepared. Check the downloaded files and column mappings.")

    panel_df = pd.concat(panel_frames, ignore_index=True)
    panel_df.to_parquet(processed_dir / "panel.parquet", index=False)
    pd.concat(chronos_frames, ignore_index=True).to_parquet(processed_dir / "chronos_df.parquet", index=False)
    write_json(processed_dir / "baseline_manifest.json", {"series": baseline_manifest})
    pd.DataFrame(echo_manifest_60).to_csv(processed_dir / "echo_H60_F1.csv", index=False)
    pd.DataFrame(echo_manifest_120).to_csv(processed_dir / "echo_H120_F5.csv", index=False)

    metadata = {
        "baseline_mode": "manifest",
        "baseline_manifest": str(processed_dir / "baseline_manifest.json"),
        "chronos_df": str(processed_dir / "chronos_df.parquet"),
        "panel": str(processed_dir / "panel.parquet"),
        "echo_H60_F1": str(processed_dir / "echo_H60_F1.csv"),
        "echo_H120_F5": str(processed_dir / "echo_H120_F5.csv"),
        "image_root": str(output_dir),
        "target_column": "OT",
        "enc_in": 12,
        "dec_in": 12,
        "c_out": 1,
        "features": "MS",
        "freq": "d",
        "top_symbols": top_symbols,
    }
    save_metadata(processed_dir / "metadata.json", metadata)


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
    baseline["return_1d"] = baseline["OT"]
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
            "return_1d",
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
                "return_1d",
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
        "enc_in": 13,
        "dec_in": 13,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Project Two datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    electricity = subparsers.add_parser("prepare-electricity")
    electricity.add_argument("--source-csv", required=True)
    electricity.add_argument("--output-dir", required=True)

    fnspid = subparsers.add_parser("prepare-fnspid")
    fnspid.add_argument("--repo-root", required=True)
    fnspid.add_argument("--output-dir", required=True)
    fnspid.add_argument("--top-k", type=int, default=50)
    fnspid.add_argument("--repo-id", default="Zihan1004/FNSPID")
    fnspid.add_argument("--price-filename", default="Stock_price/full_history.zip")
    fnspid.add_argument("--news-filename", default="Stock_news/All_external.csv")
    fnspid.add_argument("--download", action="store_true")
    fnspid.add_argument("--skip-images", action="store_true")

    oiletf = subparsers.add_parser("prepare-oiletf")
    oiletf.add_argument("--samples-h60", required=True)
    oiletf.add_argument("--samples-h120", required=True)
    oiletf.add_argument("--daily-panel", required=True)
    oiletf.add_argument("--raw-prices", required=True)
    oiletf.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-electricity":
        prepare_electricity(Path(args.source_csv), Path(args.output_dir))
    elif args.command == "prepare-fnspid":
        prepare_fnspid(
            repo_root=Path(args.repo_root),
            output_dir=Path(args.output_dir),
            top_k=args.top_k,
            repo_id=args.repo_id,
            price_filename=args.price_filename,
            news_filename=args.news_filename,
            download=args.download,
            skip_images=args.skip_images,
        )
    elif args.command == "prepare-oiletf":
        prepare_oiletf(
            Path(args.samples_h60),
            Path(args.samples_h120),
            Path(args.daily_panel),
            Path(args.raw_prices),
            Path(args.output_dir),
        )
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
