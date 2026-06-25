"""Energy dataset builder (Time-MMD weekly gasoline prices)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import ensure_dir, save_metadata

from runners.builders.common import _build_long_chronos_df, _build_start_dates


def prepare_energy(
    numerical_csv: Path,
    report_csv: Path,
    search_csv: Path,
    output_dir: Path,
) -> None:
    """Build Energy (Time-MMD) evaluation data from numerical + textual CSVs.

    The Energy dataset contains weekly U.S. gasoline price data (1993–2024)
    with 8 regional price features and an OT target.  Textual data (EIA
    reports + news search) is aligned to each weekly window by date-range
    overlap.  No images are available — evaluation is text-only.

    Produces:
      - baseline_numeric.csv   (for TSLib baselines)
      - chronos_df.parquet     (for Chronos-2)
      - echo_H60_F1.csv        (for Chronos-2-ECHO / Aurora, 60w → 1w)
      - echo_H120_F5.csv       (for Chronos-2-ECHO / Aurora, 120w → 5w)
      - metadata.json
    """
    # ── Load numerical data ─────────────────────────────────────────────
    num = pd.read_csv(numerical_csv)
    num["date"] = pd.to_datetime(num["date"])
    # Drop rows where OT is NaN, inf, or -inf — these poison all metrics
    if "OT" in num.columns:
        num["OT"] = pd.to_numeric(num["OT"], errors="coerce")
        num = num[~num["OT"].isin([float("inf"), float("-inf")])]
        num = num.dropna(subset=["OT"])
    num = num.sort_values("date").reset_index(drop=True)

    # Identify feature columns (regional gasoline prices)
    known_cols = {"date", "OT", "start_date", "end_date"}
    feature_cols = [
        c for c in num.columns
        if c not in known_cols and pd.api.types.is_numeric_dtype(num[c])
    ]

    # ── Load & merge text data ──────────────────────────────────────────
    text_frames: list[pd.DataFrame] = []
    for text_path in [report_csv, search_csv]:
        if text_path.exists():
            tf = pd.read_csv(text_path)
            if {"start_date", "end_date", "fact"}.issubset(tf.columns):
                tf["start_date"] = pd.to_datetime(tf["start_date"])
                tf["end_date"] = pd.to_datetime(tf["end_date"])
                text_frames.append(tf[["start_date", "end_date", "fact"]])

    if text_frames:
        all_text = pd.concat(text_frames, ignore_index=True)
        all_text = all_text.dropna(subset=["fact"])
        all_text["fact"] = all_text["fact"].astype(str).str.strip()
        all_text = all_text[all_text["fact"] != ""]
    else:
        all_text = pd.DataFrame(columns=["start_date", "end_date", "fact"])

    # Join text facts onto numerical rows by date-range overlap.
    # Each numerical row covers one week; we pick the most recent matching
    # fact (by end_date) when multiple text entries overlap.
    facts_by_row: list[str] = []
    for _, row in num.iterrows():
        row_start = pd.to_datetime(row["start_date"])
        row_end = pd.to_datetime(row["end_date"])
        matches = all_text[
            (all_text["start_date"] <= row_end)
            & (all_text["end_date"] >= row_start)
        ]
        if not matches.empty:
            best = matches.sort_values("end_date", ascending=False).iloc[0]
            facts_by_row.append(str(best["fact"]))
        else:
            facts_by_row.append("No news available.")

    # ── Build outputs ───────────────────────────────────────────────────
    processed_dir = ensure_dir(output_dir / "processed")

    # Baseline CSV (for TSLib baselines)
    baseline_csv = processed_dir / "baseline_numeric.csv"
    baseline = num[["date"] + feature_cols + ["OT"]].copy()
    baseline.to_csv(baseline_csv, index=False)

    # Chronos long-form parquet (for Chronos-2)
    chronos = _build_long_chronos_df(num[["date", "OT"]], item_id="Energy")
    chronos.to_parquet(processed_dir / "chronos_df.parquet", index=False)

    # Echo CSVs (for Chronos-2-ECHO and Aurora multimodal evaluation).
    # TimeMMD format requires: date, OT, fact, start_date, end_date,
    # prior_history_avg, plus any numeric feature columns.
    echo_base = num[["date", "OT", "start_date", "end_date"] + feature_cols].copy()
    echo_base["date"] = echo_base["date"].dt.strftime("%Y-%m-%d")
    echo_base["fact"] = facts_by_row

    for history, _forecast, name in [
        (60, 1, "echo_H60_F1.csv"),
        (120, 5, "echo_H120_F5.csv"),
    ]:
        echo = echo_base.copy()
        echo["start_date"] = _build_start_dates(pd.to_datetime(num["date"]), history)
        echo["prior_history_avg"] = (
            echo["OT"].rolling(history, min_periods=1).mean().fillna(0.0)
        )
        echo.to_csv(processed_dir / name, index=False)

    # ── Metadata ────────────────────────────────────────────────────────
    n_features = len(feature_cols)
    metadata = {
        "baseline_mode": "single_csv",
        "baseline_csv": str(baseline_csv),
        "chronos_df": str(processed_dir / "chronos_df.parquet"),
        "echo_H60_F1": str(processed_dir / "echo_H60_F1.csv"),
        "echo_H120_F5": str(processed_dir / "echo_H120_F5.csv"),
        "image_root": str(output_dir),
        "target_column": "OT",
        "enc_in": n_features,
        "dec_in": n_features,
        "c_out": 1,
        "features": "MS",
        "freq": "w",
    }
    save_metadata(processed_dir / "metadata.json", metadata)
