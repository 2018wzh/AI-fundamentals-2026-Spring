"""Shared helpers for dataset builders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import ensure_dir


def _detect_column(columns: Iterable[str], candidates: list[str], *, required: bool = True) -> str | None:
    lowered = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise ValueError(f"Could not find any of columns {candidates} in {list(columns)}")
    return None


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
        "ma_5",
        "ma_20",
        "volatility_20",
        "news_count",
        "sentiment_mean",
    ]
    if "image_path" in echo.columns:
        keep_cols.append("image_path")
    return echo[keep_cols]
