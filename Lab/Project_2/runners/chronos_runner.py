from __future__ import annotations

import os as _os

# Route HF requests through mirror for network accessibility
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    compute_metrics,
    elapsed_sec,
    load_metadata,
    make_prediction_frame,
    peak_vram_mb,
    reset_peak_vram,
    rolling_test_starts,
    timer,
    write_outputs,
)


def parse_setting(setting: str) -> tuple[int, int]:
    history = int(setting.split("_")[0].removeprefix("H"))
    forecast = int(setting.split("_")[1].removeprefix("F"))
    return history, forecast


def build_inputs(frame: pd.DataFrame, history: int, forecast: int) -> tuple[list[dict], list[dict]]:
    records = []
    metadata = []
    covariate_cols = [col for col in frame.columns if col not in {"item_id", "timestamp", "target"}]

    for series_id, series in frame.groupby("item_id"):
        series = series.sort_values("timestamp").reset_index(drop=True)
        values = series["target"].to_numpy(dtype=np.float32)
        timestamps = pd.to_datetime(series["timestamp"]).tolist()
        covariates = {col: series[col].to_numpy(dtype=np.float32) for col in covariate_cols}
        for window_id, start in enumerate(rolling_test_starts(len(series), history, forecast)):
            end = start + history
            future_end = end + forecast
            if future_end > len(series):
                continue
            payload = {"target": values[start:end]}
            if covariates:
                payload["past_covariates"] = {col: arr[start:end] for col, arr in covariates.items()}
                payload["future_covariates"] = {col: arr[end:future_end] for col, arr in covariates.items()}
            records.append(payload)
            metadata.append(
                {
                    "series_id": series_id,
                    "window_id": window_id,
                    "future_timestamps": timestamps[end:future_end],
                    "y_true": values[end:future_end],
                }
            )
    return records, metadata


def to_numpy_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def normalize_quantiles(value) -> np.ndarray:
    array = to_numpy_array(value)
    if array.ndim == 4 and array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    return array


def normalize_means(value) -> np.ndarray:
    array = to_numpy_array(value)
    if array.ndim == 3 and array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    return array


def cmd_evaluate(args: argparse.Namespace) -> None:

    from chronos import Chronos2Pipeline

    metadata = load_metadata(args.metadata)
    df = pd.read_parquet(metadata["chronos_df"])
    history, forecast = parse_setting(args.setting)
    inputs, input_meta = build_inputs(df, history, forecast)
    if args.max_windows and args.max_windows > 0:
        inputs = inputs[: args.max_windows]
        input_meta = input_meta[: args.max_windows]
    if not inputs:
        raise RuntimeError(f"No evaluation windows built for {args.dataset} {args.setting}")

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    reset_peak_vram()
    started = timer()
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device_map)

    quantiles_all = []
    means_all = []
    for start in range(0, len(inputs), args.batch_size):
        batch = inputs[start : start + args.batch_size]
        quantiles, means = pipeline.predict_quantiles(
            batch,
            prediction_length=forecast,
            quantile_levels=[0.1, 0.5, 0.9],
            limit_prediction_length=False,
        )
        quantiles_all.append(normalize_quantiles(quantiles))
        means_all.append(normalize_means(means))

    quantiles_np = np.concatenate(quantiles_all, axis=0)
    means_np = np.concatenate(means_all, axis=0)
    runtime_sec = elapsed_sec(started)
    vram_mb = peak_vram_mb()

    records = []
    y_true_batches = []
    y_pred_batches = []
    q10_batches = []
    q50_batches = []
    q90_batches = []
    for idx, meta in enumerate(input_meta):
        q10 = quantiles_np[idx, :, 0]
        q50 = quantiles_np[idx, :, 1]
        q90 = quantiles_np[idx, :, 2]
        y_pred = means_np[idx]
        y_true = np.asarray(meta["y_true"], dtype=np.float32)
        y_true_batches.append(y_true)
        y_pred_batches.append(y_pred)
        q10_batches.append(q10)
        q50_batches.append(q50)
        q90_batches.append(q90)
        for horizon_idx, timestamp in enumerate(meta["future_timestamps"]):
            records.append(
                {
                    "series_id": meta["series_id"],
                    "window_id": meta["window_id"],
                    "horizon_idx": horizon_idx,
                    "target_idx": 0,
                    "timestamp": timestamp,
                    "y_true": float(y_true[horizon_idx]),
                    "y_pred": float(y_pred[horizon_idx]),
                    "q10": float(q10[horizon_idx]),
                    "q50": float(q50[horizon_idx]),
                    "q90": float(q90[horizon_idx]),
                }
            )

    predictions = make_prediction_frame(
        dataset=args.dataset,
        model="Chronos-2",
        setting=args.setting,
        records=records,
    )
    metrics = compute_metrics(
        y_true=np.stack(y_true_batches),
        y_pred=np.stack(y_pred_batches),
        q10=np.stack(q10_batches),
        q50=np.stack(q50_batches),
        q90=np.stack(q90_batches),
        dataset=args.dataset,
        runtime_sec=runtime_sec,
        peak_vram_mb=vram_mb,
        mode="zero_shot",
    )
    runtime = {
        "mode": "zero_shot",
        "runtime_sec": runtime_sec,
        "peak_vram_mb": vram_mb,
        "num_windows": len(input_meta),
        "device_map": device_map,
        "model_id": args.model_id,
    }
    write_outputs(
        output_dir=args.output_dir,
        dataset=args.dataset,
        model="Chronos-2",
        setting=args.setting,
        metrics=metrics,
        predictions=predictions,
        runtime=runtime,
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Fine-tune Chronos-2 on chronos_df data."""
    from chronos import Chronos2Pipeline

    metadata = load_metadata(args.metadata)
    df = pd.read_parquet(metadata["chronos_df"])
    history, forecast = parse_setting(args.setting)

    # Build inputs as list of per-series dicts (format expected by Chronos2Pipeline.fit)
    covariate_cols = [c for c in df.columns if c not in {"item_id", "timestamp", "target"}]
    inputs = []
    for item_id, grp in df.groupby("item_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        entry = {"target": grp["target"].to_numpy(dtype=np.float32)}
        if covariate_cols:
            entry["past_covariates"] = {c: grp[c].to_numpy(dtype=np.float32) for c in covariate_cols}
        inputs.append(entry)

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    reset_peak_vram()
    started = timer()

    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device_map)
    print(f"Training on {len(inputs)} series, {sum(len(e['target']) for e in inputs)} total samples")
    pipeline.fit(
        inputs,
        prediction_length=forecast,
        finetune_mode=args.finetune_mode,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
    )

    runtime_sec = elapsed_sec(started)
    vram_mb = peak_vram_mb()
    print(f"Training completed in {runtime_sec:.1f}s, peak VRAM {vram_mb} MB")

    import json as _json
    runtime = {
        "mode": "training",
        "runtime_sec": runtime_sec,
        "peak_vram_mb": vram_mb,
        "num_steps": args.num_steps,
        "device_map": device_map,
        "model_id": args.model_id,
    }
    (Path(args.output_dir) / "runtime.json").write_text(
        _json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chronos-2 runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--dataset", required=True)
    eval_parser.add_argument("--setting", required=True)
    eval_parser.add_argument("--metadata", required=True)
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.add_argument("--model-id", default="amazon/chronos-2")
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--max-windows", type=int, default=0)

    # Train subcommand
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--setting", required=True)
    train_parser.add_argument("--metadata", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--model-id", default="amazon/chronos-2")
    train_parser.add_argument("--finetune-mode", default="lora", choices=["lora", "full"])
    train_parser.add_argument("--learning-rate", type=float, default=5e-6)
    train_parser.add_argument("--num-steps", type=int, default=20000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
