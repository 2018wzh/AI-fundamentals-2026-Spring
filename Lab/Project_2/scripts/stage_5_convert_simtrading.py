#!/usr/bin/env python3
"""Stage 5: Convert evaluation predictions to explicit SimTrading format.

Handles ALL datasets generically — no hardcoded dataset or setting names.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
for _p in [str(_project_root), str(_project_root / "runners")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_script_dir) in sys.path:
    sys.path.remove(str(_script_dir))

from scripts.common import ensure_dir, load_json

SIMTRADING_ROOT = _project_root / "SimTrading"
PREDICTIONS_OUTPUT = SIMTRADING_ROOT / "predictions"
CONFIG_OUTPUT = SIMTRADING_ROOT / "config" / "datasets.yaml"


def _parse_setting(setting: str) -> tuple[int, int]:
    history = int(setting.split("_")[0].removeprefix("H"))
    forecast = int(setting.split("_")[1].removeprefix("F"))
    return history, forecast


def _resolve_known_path(path: str | Path) -> Path:
    raw = str(path).replace("\\", "/")
    if raw.startswith("/home/zhw/Project_2/"):
        return _project_root / raw.removeprefix("/home/zhw/Project_2/")
    return Path(raw).expanduser()


def _model_label(model: str, raw_setting_dir: str) -> tuple[str, str] | None:
    """Parse model name and mode from setting directory name.

    Examples:
      "H60_F1"         → ("H60_F1", "Chronos-2")
      "H192_F6_zero_shot" → ("H192_F6", "Chronos-2-ECHO-zero_shot")
      "H528_F48_training" → ("H528_F48", None)  # skip training dirs
    """
    clean = raw_setting_dir
    mode_suffix = ""
    for suffix in ["_text_only", "_text_image", "_image_only", "_zero_shot"]:
        if raw_setting_dir.endswith(suffix):
            clean = raw_setting_dir[: -len(suffix)]
            mode_suffix = raw_setting_dir[-len(suffix) + 1 :]
            break
    if raw_setting_dir.endswith("_training"):
        return None  # skip training dirs
    # Validate it looks like a setting: H{number}_F{number}
    try:
        _parse_setting(clean)
    except (ValueError, AttributeError):
        return None
    return clean, f"{model}-{mode_suffix}" if mode_suffix else model


def _generic_end_dates(dataset_dir: Path, setting: str) -> pd.DataFrame | None:
    """Build end_date map from the echo CSV for a dataset.

    For Time-MMD single-series datasets, reads the echo CSV directly.
    For multi-series (fnspid), reads the manifest.
    """
    metadata_path = _project_root / "data" / dataset_dir.name / "processed" / "metadata.json"
    if not metadata_path.exists():
        return None

    metadata = load_json(str(metadata_path))
    echo_key = f"echo_{setting}"
    echo_path_raw = metadata.get(echo_key)
    if not echo_path_raw:
        return None

    echo_path = _resolve_known_path(echo_path_raw)
    if not echo_path.exists():
        return None

    history, forecast = _parse_setting(setting)

    # Check if it's a manifest CSV (has series_id, csv_path columns)
    manifest = pd.read_csv(echo_path)
    if {"series_id", "csv_path"}.issubset(manifest.columns):
        # Multi-series: fnspid style
        pieces = []
        for _, row in manifest.iterrows():
            series_csv = _resolve_known_path(row["csv_path"])
            if not series_csv.exists():
                continue
            df = pd.read_csv(series_csv, usecols=["end_date"]).reset_index(drop=True)
            df["end_date"] = pd.to_datetime(df["end_date"])
            n = len(df)
            train_rows = int(n * 0.7)
            val_rows = int(n * 0.2)
            test_start = max(0, train_rows + val_rows - history)
            max_windows = n - test_start - (history + forecast) + 1
            if max_windows <= 0:
                continue
            window_df = df.iloc[test_start : test_start + max_windows].copy()
            window_df["window_id"] = range(len(window_df))
            window_df["symbol"] = row["series_id"]
            pieces.append(window_df)
        return pd.concat(pieces, ignore_index=True) if pieces else None
    else:
        # Single-series: Time-MMD
        df = manifest.copy()
        df["end_date"] = pd.to_datetime(df["end_date"] if "end_date" in df.columns else df["date"])
        n = len(df)
        train_rows = int(n * 0.7)
        test_start = max(0, train_rows - history)
        max_windows = n - test_start - (history + forecast) + 1
        if max_windows <= 0:
            return None
        window_df = df.iloc[test_start : test_start + max_windows].copy()
        window_df["window_id"] = range(len(window_df))
        window_df["symbol"] = dataset_dir.name
        return window_df[["window_id", "end_date", "symbol"]]


def _convert_one(
    predictions_path: Path,
    *,
    dataset_label: str,
    model_label: str,
    setting: str,
    end_date_map: pd.DataFrame,
) -> pd.DataFrame:
    preds = pd.read_parquet(predictions_path)
    required = {"series_id", "window_id", "horizon_idx", "y_true", "y_pred"}
    missing = required.difference(preds.columns)
    if missing:
        raise ValueError(f"{predictions_path} missing columns: {sorted(missing)}")

    symbol_col = end_date_map["symbol"].iloc[0] if "symbol" in end_date_map.columns else dataset_label
    out = pd.DataFrame(
        {
            "dataset": dataset_label,
            "model": model_label,
            "setting": setting,
            "symbol": preds["series_id"].astype(str) if "series_id" in preds.columns else str(symbol_col),
            "window": preds["window_id"].astype(int),
            "horizon": preds["horizon_idx"].astype(int),
            "y_true": preds["y_true"].astype(float),
            "y_pred": preds["y_pred"].astype(float),
            "split": "test",
        }
    )
    for col in ["q10", "q50", "q90"]:
        if col in preds.columns:
            out[col] = preds[col].astype(float)

    out = out.merge(end_date_map[["window_id", "end_date"]], left_on="window", right_on="window_id", how="left")
    out = out.drop(columns=["window_id"], errors="ignore")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    out = out.dropna(subset=["end_date"])
    out = out.drop_duplicates(["dataset", "model", "setting", "symbol", "end_date", "window", "horizon"])
    return out.sort_values(["symbol", "setting", "window", "horizon"]).reset_index(drop=True)


def _dataset_config_key(dataset_name: str) -> str:
    """Map dataset directory names to SimTrading config keys."""
    mapping = {
        "oiletf": "oil_etf",
        "oiletf_intraday": "oil_etf_intraday",
    }
    return mapping.get(dataset_name, dataset_name)


def _generate_config(prediction_files: dict[str, dict[str, Path]]) -> str:
    """Generate datasets.yaml from collected prediction files."""
    lines = ["# Auto-generated by stage_5_convert_simtrading", "# Do not edit manually", "datasets:"]

    # Dataset metadata
    meta = {
        "fnspid": {
            "display": "FNSPID",
            "panel_path": _project_root / "data" / "fnspid" / "processed" / "panel.parquet",
            "date_column": "date",
            "symbol_column": "symbol",
            "exec_price": "open",
            "mark_price": "close",
            "benchmark_price": "close",
            "benchmarks": ["SPY"],
        },
        "oil_etf": {
            "display": "OilETF-TimeMMD",
            "panel_path": Path("/home/zhw/OilETF-TimeMMD/data/processed/daily_panel.parquet"),
            "date_column": "end_date",
            "symbol_column": "symbol",
            "exec_price": "uso_open",
            "mark_price": "uso_close",
            "benchmark_price": "uso_close",
            "benchmarks": ["USO"],
        },
        "oil_etf_intraday": {
            "display": "OilETF-Intraday",
            "panel_path": Path("/home/zhw/OilETF-TimeMMD/data/processed/hourly_panel.parquet"),
            "date_column": "end_date",
            "symbol_column": "symbol",
            "exec_price": "uso_open",
            "mark_price": "uso_close",
            "benchmark_price": "uso_close",
            "benchmarks": ["USO"],
        },
    }

    # Add Time-MMD domains
    timemmd_domains = ["agriculture", "climate", "economy", "energy", "environment",
                       "health_afr", "security", "socialgood", "traffic"]
    for ds in timemmd_domains:
        meta[ds] = {
            "display": ds.replace("_", " ").title(),
            "panel_path": _project_root / "data" / ds / "processed" / "chronos_df.parquet",
            "date_column": "timestamp",
            "symbol_column": "item_id",
            "exec_price": "target",
            "mark_price": "target",
            "benchmark_price": "target",
            "benchmarks": ["target"],
        }

    for config_key in sorted(prediction_files.keys()):
        m = meta.get(config_key)
        if not m:
            continue
        lines.extend([
            f"  {config_key}:",
            f'    display_name: "{m["display"]}"',
            f'    panel_path: "{m["panel_path"]}"',
            f'    date_column: "{m["date_column"]}"',
            f'    symbol_column: "{m["symbol_column"]}"',
            "    price_columns:",
            f'      execution_price: "{m["exec_price"]}"',
            f'      mark_price: "{m["mark_price"]}"',
            f'      benchmark_price: "{m["benchmark_price"]}"',
            "    sample_paths:",
        ])
        # Read actual settings from prediction files
        settings_seen = set()
        for model_label, path in sorted(prediction_files[config_key].items()):
            try:
                df = pd.read_parquet(path)
                for s in df["setting"].unique():
                    settings_seen.add(str(s))
            except Exception:
                pass
        for s in sorted(settings_seen):
            echo_path = _project_root / "data" / config_key.replace("oil_etf", "oiletf") / "processed" / f"echo_{s}.csv"
            # For Time-MMD, the echo file might use different naming
            if not echo_path.exists():
                echo_path = _project_root / "data" / config_key / "processed" / f"H{_parse_setting(s)[0]}_F{_parse_setting(s)[1]}.csv"
            lines.append(f'      {s}: "{echo_path}"')

        lines.append("    prediction_result_paths:")
        for model_label, path in sorted(prediction_files[config_key].items()):
            lines.append(f'      {model_label}: "{path}"')
        lines.append("    benchmark_symbols:")
        for b in m["benchmarks"]:
            lines.append(f'      - "{b}"')
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    results_root = _project_root / "results"
    if not results_root.exists():
        print("No results/ found. Run stage 3 first.")
        return 1

    converted: dict[Path, list[pd.DataFrame]] = {}
    prediction_files: dict[str, dict[str, Path]] = {}

    # Only these datasets are supported by SimTrading
    SIMTRADING_DATASETS = {"fnspid", "oiletf", "oiletf_intraday"}

    for dataset_dir in sorted(results_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name not in SIMTRADING_DATASETS:
            continue
        if args.datasets and dataset_dir.name not in args.datasets:
            continue

        dataset_label = dataset_dir.name
        config_key = _dataset_config_key(dataset_label)

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if args.models and model not in args.models:
                continue

            for setting_dir in sorted(model_dir.iterdir()):
                if not setting_dir.is_dir():
                    continue
                label = _model_label(model, setting_dir.name)
                if label is None:
                    continue
                setting, model_label = label
                preds_path = setting_dir / "predictions.parquet"
                if not preds_path.is_file():
                    continue

                end_date_map = _generic_end_dates(dataset_dir, setting)
                if end_date_map is None or end_date_map.empty:
                    print(f"  ⚠ Skipping {dataset_label}/{setting_dir.name}: no end_date map")
                    continue

                try:
                    out = _convert_one(
                        preds_path,
                        dataset_label=dataset_label,
                        model_label=model_label,
                        setting=setting,
                        end_date_map=end_date_map,
                    )
                except Exception as exc:
                    print(f"  ✗ {dataset_label}/{setting_dir.name}: {exc}")
                    continue

                if out.empty:
                    continue
                output_path = PREDICTIONS_OUTPUT / dataset_dir.name / f"{model_label}.parquet"
                converted.setdefault(output_path, []).append(out)
                prediction_files.setdefault(config_key, {})[model_label] = output_path
                print(f"  ✓ {dataset_label}/{setting_dir.name} → {output_path.name} ({len(out)} rows)")

    if not converted:
        print("No predictions converted.")
        return 1

    for output_path, pieces in converted.items():
        ensure_dir(output_path.parent)
        result = pd.concat(pieces, ignore_index=True)
        result = result.drop_duplicates(["dataset", "model", "setting", "symbol", "end_date", "window", "horizon"])
        result = result.sort_values(["symbol", "setting", "window", "horizon"]).reset_index(drop=True)
        result.to_parquet(output_path, index=False)
        print(f"Wrote: {output_path}")

    if args.write_config:
        ensure_dir(CONFIG_OUTPUT.parent)
        CONFIG_OUTPUT.write_text(_generate_config(prediction_files), encoding="utf-8")
        print(f"Config written: {CONFIG_OUTPUT}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 5: Convert predictions to explicit SimTrading format")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--write-config", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
