#!/usr/bin/env python3
"""Stage 5: Convert evaluation predictions to explicit SimTrading format."""

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
    if raw.startswith("/home/zhw/OilETF-TimeMMD/"):
        return Path("D:/Workspace/OilETF-TimeMMD") / raw.removeprefix("/home/zhw/OilETF-TimeMMD/")
    return Path(raw).expanduser()


def _oiletf_end_dates(setting: str) -> pd.DataFrame:
    sample_path = {
        "H60_F1": "D:/Workspace/OilETF-TimeMMD/data/processed/samples_H60_F1.parquet",
        "H120_F5": "D:/Workspace/OilETF-TimeMMD/data/processed/samples_H120_F5.parquet",
    }[setting]
    samples = pd.read_parquet(sample_path)
    test = samples[samples["split"].astype(str).str.lower() == "test"].sort_values("end_date").reset_index(drop=True)
    test["window_id"] = range(len(test))
    return test[["window_id", "end_date", "symbol"]]


def _fnspid_end_dates(metadata_path: Path, setting: str) -> dict[str, pd.DataFrame]:
    metadata = load_json(str(metadata_path))
    manifest_path = metadata.get(f"echo_{setting}")
    if not manifest_path:
        return {}

    manifest = pd.read_csv(_resolve_known_path(manifest_path))
    history, forecast = _parse_setting(setting)
    result: dict[str, pd.DataFrame] = {}

    for symbol, csv_path in zip(manifest["series_id"], manifest["csv_path"]):
        resolved = _resolve_known_path(csv_path)
        if not resolved.exists():
            continue
        df = pd.read_csv(resolved, usecols=["end_date"]).reset_index(drop=True)
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
        result[str(symbol)] = window_df[["window_id", "end_date"]]
    return result


def _convert_one(
    predictions_path: Path,
    *,
    dataset_label: str,
    model_label: str,
    setting: str,
    end_date_map: pd.DataFrame | None = None,
    fnspid_end_dates: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    preds = pd.read_parquet(predictions_path)
    required = {"series_id", "window_id", "horizon_idx", "y_true", "y_pred"}
    missing = required.difference(preds.columns)
    if missing:
        raise ValueError(f"{predictions_path} missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "dataset": dataset_label,
            "model": model_label,
            "setting": setting,
            "symbol": "oiletf" if dataset_label == "oiletf" else preds["series_id"].astype(str),
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

    if end_date_map is not None:
        out = out.merge(end_date_map[["window_id", "end_date"]], left_on="window", right_on="window_id", how="left")
    elif fnspid_end_dates is not None:
        pieces = []
        for symbol, subset in out.groupby("symbol", sort=False):
            dates = fnspid_end_dates.get(symbol)
            if dates is None:
                continue
            pieces.append(subset.merge(dates, left_on="window", right_on="window_id", how="left"))
        out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=list(out.columns) + ["end_date"])
    else:
        raise ValueError("Missing end_date resolver.")

    out = out.drop(columns=["window_id"], errors="ignore")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    out = out.dropna(subset=["end_date"])
    out = out.drop_duplicates(["dataset", "model", "setting", "symbol", "end_date", "window", "horizon"])
    return out.sort_values(["symbol", "setting", "window", "horizon"]).reset_index(drop=True)


def _model_label(model: str, raw_setting_dir: str) -> tuple[str, str] | None:
    clean = raw_setting_dir
    mode_suffix = ""
    for suffix in ["_text_only", "_text_image", "_image_only", "_training"]:
        if raw_setting_dir.endswith(suffix):
            clean = raw_setting_dir[: -len(suffix)]
            mode_suffix = raw_setting_dir[-len(suffix) + 1 :]
            break
    if clean not in {"H60_F1", "H120_F5"}:
        return None
    return clean, f"{model}-{mode_suffix}" if mode_suffix else model


def _generate_config(prediction_files: dict[str, dict[str, Path]]) -> str:
    meta = {
        "fnspid": {
            "display": "FNSPID",
            "dataset_label": "fnspid",
            "panel_path": _project_root / "data" / "fnspid" / "processed" / "panel.parquet",
            "date_column": "date",
            "symbol_mode": "column",
            "symbol_column": "symbol",
            "constant_symbol": None,
            "sample_format": "series_manifest",
            "sample_symbol_mode": "column",
            "sample_symbol_column": "series_id",
            "sample_constant_symbol": None,
            "sample_h60": _project_root / "data" / "fnspid" / "processed" / "echo_H60_F1.csv",
            "sample_h120": _project_root / "data" / "fnspid" / "processed" / "echo_H120_F5.csv",
            "execution_price": "open",
            "mark_price": "close",
            "benchmark_price": "close",
            "benchmarks": ["SPY"],
        },
        "oil_etf": {
            "display": "OilETF-TimeMMD",
            "dataset_label": "oiletf",
            "panel_path": Path("D:/Workspace/OilETF-TimeMMD/data/processed/daily_panel.parquet"),
            "date_column": "end_date",
            "symbol_mode": "constant",
            "symbol_column": None,
            "constant_symbol": "oiletf",
            "sample_format": "metadata_table",
            "sample_symbol_mode": "constant",
            "sample_symbol_column": None,
            "sample_constant_symbol": "oiletf",
            "sample_h60": Path("D:/Workspace/OilETF-TimeMMD/data/processed/samples_H60_F1.parquet"),
            "sample_h120": Path("D:/Workspace/OilETF-TimeMMD/data/processed/samples_H120_F5.parquet"),
            "execution_price": "uso_open",
            "mark_price": "uso_close",
            "benchmark_price": "uso_close",
            "benchmarks": ["USO"],
        },
    }

    lines = ["# Explicit real-data contract for SimTrading.", "datasets:"]
    for dataset_key in ["fnspid", "oil_etf"]:
        m = meta[dataset_key]
        lines.extend(
            [
                f"  {dataset_key}:",
                f'    display_name: "{m["display"]}"',
                f'    dataset_label: "{m["dataset_label"]}"',
                "    panel:",
                f'      path: "{m["panel_path"]}"',
                f'      date_column: "{m["date_column"]}"',
                f'      symbol_mode: "{m["symbol_mode"]}"',
            ]
        )
        if m["symbol_column"]:
            lines.append(f'      symbol_column: "{m["symbol_column"]}"')
        if m["constant_symbol"]:
            lines.append(f'      constant_symbol: "{m["constant_symbol"]}"')
        lines.extend(
            [
                "      price_columns:",
                f'        execution_price: "{m["execution_price"]}"',
                f'        mark_price: "{m["mark_price"]}"',
                f'        benchmark_price: "{m["benchmark_price"]}"',
                "    samples:",
                f'      format: "{m["sample_format"]}"',
                f'      symbol_mode: "{m["sample_symbol_mode"]}"',
            ]
        )
        if m["sample_symbol_column"]:
            lines.append(f'      symbol_column: "{m["sample_symbol_column"]}"')
        if m["sample_constant_symbol"]:
            lines.append(f'      constant_symbol: "{m["sample_constant_symbol"]}"')
        lines.extend(
            [
                "      paths:",
                f'        H60_F1: "{m["sample_h60"]}"',
                f'        H120_F5: "{m["sample_h120"]}"',
                "    predictions:",
                "      require_setting: true",
                "      signal_horizons:",
                "        H60_F1: 0",
                "        H120_F5: 4",
                "      result_paths:",
            ]
        )
        for model_label, path in sorted(prediction_files.get(dataset_key, {}).items()):
            lines.append(f'        {model_label}: "{path}"')
        lines.append("    benchmark_symbols:")
        for benchmark in m["benchmarks"]:
            lines.append(f'      - "{benchmark}"')
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    results_root = _project_root / "results"
    if not results_root.exists():
        print("No results/ found. Run stage 3 first.")
        return 1

    converted: dict[Path, list[pd.DataFrame]] = {}
    prediction_files: dict[str, dict[str, Path]] = {}

    for dataset_dir in sorted(results_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name not in {"fnspid", "oiletf"}:
            continue
        if args.datasets and dataset_dir.name not in args.datasets:
            continue

        dataset_key = "oil_etf" if dataset_dir.name == "oiletf" else dataset_dir.name
        dataset_label = dataset_dir.name
        fnspid_maps: dict[str, dict[str, pd.DataFrame]] = {}

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

                end_date_map = None
                fnspid_map = None
                if dataset_dir.name == "oiletf":
                    end_date_map = _oiletf_end_dates(setting)
                else:
                    if setting not in fnspid_maps:
                        fnspid_maps[setting] = _fnspid_end_dates(
                            _project_root / "data" / "fnspid" / "processed" / "metadata.json",
                            setting,
                        )
                    fnspid_map = fnspid_maps[setting]

                out = _convert_one(
                    preds_path,
                    dataset_label=dataset_label,
                    model_label=model_label,
                    setting=setting,
                    end_date_map=end_date_map,
                    fnspid_end_dates=fnspid_map,
                )
                if out.empty:
                    continue
                output_path = PREDICTIONS_OUTPUT / dataset_dir.name / f"{model_label}.parquet"
                converted.setdefault(output_path, []).append(out)
                prediction_files.setdefault(dataset_key, {})[model_label] = output_path
                print(f"{dataset_dir.name}/{setting_dir.name} -> {output_path} ({len(out)} rows)")

    for output_path, pieces in converted.items():
        ensure_dir(output_path.parent)
        result = pd.concat(pieces, ignore_index=True)
        result = result.drop_duplicates(["dataset", "model", "setting", "symbol", "end_date", "window", "horizon"])
        result = result.sort_values(["symbol", "setting", "window", "horizon"]).reset_index(drop=True)
        result.to_parquet(output_path, index=False)

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
