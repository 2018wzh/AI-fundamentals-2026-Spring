"""Generic Time-MMD dataset builder — one function for all domains.

Each Time-MMD domain follows the same layout:
  numerical/{Domain}/{Domain}.csv
  textual/{Domain}/{Domain}_report.csv
  textual/{Domain}/{Domain}_search.csv

By default uses Time-MMD standard protocol: features="S" (univariate).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import ensure_dir, save_metadata

from runners.builders.common import _build_long_chronos_df, _build_start_dates

# All Time-MMD domains
TIMEMMD_DOMAINS = [
    "Agriculture",
    "Climate",
    "Economy",
    "Energy",
    "Environment",
    "Health_AFR",
    "Security",
    "SocialGood",
    "Traffic",
]

# Time-MMD standard seq_len per domain (from Aurora/TimeMMD reference script)
TIMEMMD_SEQ_LEN = {
    "Agriculture": 192,
    "Climate": 192,
    "Economy": 192,
    "Energy": 1056,
    "Environment": 528,
    "Health_AFR": 96,
    "Security": 220,
    "SocialGood": 192,
    "Traffic": 96,
}


def prepare_timemmd(
    domain: str,
    numerical_csv: Path,
    report_csv: Path,
    search_csv: Path,
    output_dir: Path,
    *,
    features: str = "S",
) -> None:
    """Build a single Time-MMD domain for evaluation.

    ``features="S"`` (default) → Time-MMD standard: univariate, no covariates.
    ``features="MS"`` → legacy: include numeric feature columns as covariates.

    Produces:
      - baseline_numeric.csv   (for TSLib baselines)
      - chronos_df.parquet     (for Chronos-2)
      - echo_H{seq}_F*.csv     (for Chronos-2-ECHO / Aurora)
      - metadata.json
    """
    seq_len = TIMEMMD_SEQ_LEN.get(domain, 60)

    # ── Load numerical data ─────────────────────────────────────────────
    num = pd.read_csv(numerical_csv)
    if "date" not in num.columns:
        if "start_date" in num.columns:
            num["date"] = num["start_date"]
        elif "end_date" in num.columns:
            num["date"] = num["end_date"]
        else:
            raise ValueError(f"Numerical CSV missing date column: {numerical_csv}")
    num["date"] = pd.to_datetime(num["date"])
    # Drop rows where OT is NaN, inf, or -inf
    if "OT" in num.columns:
        num["OT"] = pd.to_numeric(num["OT"], errors="coerce")
        num = num[~num["OT"].isin([float("inf"), float("-inf")])]
        num = num.dropna(subset=["OT"])
    num = num.sort_values("date").reset_index(drop=True)

    # Identify feature columns (only for MS mode)
    known_cols = {"date", "OT", "start_date", "end_date"}
    feature_cols = []
    if features == "MS":
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

    has_dates = "start_date" in num.columns and "end_date" in num.columns
    facts_by_row: list[str] = []
    for _, row in num.iterrows():
        if has_dates:
            row_start = pd.to_datetime(row["start_date"])
            row_end = pd.to_datetime(row["end_date"])
        else:
            row_dt = pd.to_datetime(row["date"])
            row_start = row_dt
            row_end = row_dt
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

    # Baseline CSV
    baseline_csv = processed_dir / "baseline_numeric.csv"
    baseline = num[["date"] + feature_cols + ["OT"]].copy()
    baseline.to_csv(baseline_csv, index=False)

    # Chronos long-form parquet
    chronos = _build_long_chronos_df(num[["date", "OT"]], item_id=domain)
    chronos.to_parquet(processed_dir / "chronos_df.parquet", index=False)

    # Echo CSV (Time-MMD standard: only target + text, no covariates)
    if has_dates:
        echo_base = num[["date", "OT", "start_date", "end_date"] + feature_cols].copy()
    else:
        echo_base = num[["date", "OT"] + feature_cols].copy()
        echo_base["start_date"] = echo_base["date"]
        echo_base["end_date"] = echo_base["date"]
    echo_base["date"] = echo_base["date"].dt.strftime("%Y-%m-%d")
    echo_base["fact"] = facts_by_row

    # Build one echo CSV per standard horizon
    horizons = _timemmd_horizons(domain)
    for history, forecast, name in horizons:
        echo = echo_base.copy()
        echo["start_date"] = _build_start_dates(pd.to_datetime(num["date"]), history)
        echo["prior_history_avg"] = (
            echo["OT"].rolling(seq_len, min_periods=1).mean().fillna(0.0)
        )
        echo.to_csv(processed_dir / name, index=False)

    # ── Metadata ────────────────────────────────────────────────────────
    metadata = {
        "baseline_mode": "single_csv",
        "baseline_csv": str(baseline_csv),
        "chronos_df": str(processed_dir / "chronos_df.parquet"),
        "image_root": str(output_dir),
        "target_column": "OT",
        "enc_in": len(feature_cols),
        "dec_in": len(feature_cols),
        "c_out": 1,
        "features": features,
        "freq": "w",
        "timemmd_seq_len": seq_len,
    }
    # Add echo CSV paths (key: echo_H{seq}_F{pred}, value: H{seq}_F{pred}.csv)
    for _, _, name in horizons:
        key = f"echo_{name.replace('.csv','')}"
        metadata[key] = str(processed_dir / name)
    save_metadata(processed_dir / "metadata.json", metadata)


def _timemmd_horizons(domain: str) -> list[tuple[int, int, str]]:
    """Return (history, forecast, echo_filename) for a domain's standard horizons."""
    seq = TIMEMMD_SEQ_LEN.get(domain, 60)
    if domain in ("Energy", "Health_AFR"):
        preds = [12, 24, 36, 48]
    elif domain == "Environment":
        preds = [48, 96, 192, 336]
    else:
        preds = [6, 8, 10, 12]
    return [(seq, p, f"H{seq}_F{p}.csv") for p in preds]
