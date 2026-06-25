from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import compute_metrics, ensure_dir, write_json, write_outputs


def load_json(path: str | Path) -> dict:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_baseline_chunks(
    pred_np: np.ndarray,
    true_np: np.ndarray,
    *,
    dataset: str,
    model: str,
    setting: str,
    series_id: str,
    window_chunk_size: int = 8,
    target_chunk_size: int = 32,
):
    num_windows, num_horizons, num_targets = pred_np.shape
    for window_start in range(0, num_windows, window_chunk_size):
        window_end = min(window_start + window_chunk_size, num_windows)
        window_count = window_end - window_start
        for target_start in range(0, num_targets, target_chunk_size):
            target_end = min(target_start + target_chunk_size, num_targets)
            target_count = target_end - target_start
            pred_chunk = np.asarray(pred_np[window_start:window_end, :, target_start:target_end], dtype=np.float64)
            true_chunk = np.asarray(true_np[window_start:window_end, :, target_start:target_end], dtype=np.float64)
            row_count = pred_chunk.size
            window_ids = np.repeat(np.arange(window_start, window_end, dtype=np.int64), num_horizons * target_count)
            horizon_ids = np.tile(np.repeat(np.arange(num_horizons, dtype=np.int64), target_count), window_count)
            target_ids = np.tile(np.arange(target_start, target_end, dtype=np.int64), window_count * num_horizons)
            yield {
                "dataset": pa.array([dataset] * row_count, type=pa.string()),
                "model": pa.array([model] * row_count, type=pa.string()),
                "setting": pa.array([setting] * row_count, type=pa.string()),
                "series_id": pa.array([series_id] * row_count, type=pa.string()),
                "window_id": pa.array(window_ids, type=pa.int64()),
                "horizon_idx": pa.array(horizon_ids, type=pa.int64()),
                "target_idx": pa.array(target_ids, type=pa.int64()),
                "timestamp": pa.array([None] * row_count, type=pa.string()),
                "y_true": pa.array(true_chunk.reshape(-1), type=pa.float64()),
                "y_pred": pa.array(pred_chunk.reshape(-1), type=pa.float64()),
                "q10": pa.array([None] * row_count, type=pa.float64()),
                "q50": pa.array([None] * row_count, type=pa.float64()),
                "q90": pa.array([None] * row_count, type=pa.float64()),
            }


def normalize_baseline_func(
    *,
    dataset: str,
    model: str,
    setting: str,
    source_dir: Path | str,
    output_dir: Path | str,
    runtime_sec: float,
    series_id: str = "series",
    mode: str = "baseline_train",
) -> None:
    """Convert raw TSLib .npy outputs to normalized metrics.csv + predictions.parquet."""
    source_dir = Path(source_dir)
    pred_np = np.load(source_dir / "pred.npy", mmap_mode="r")
    true_np = np.load(source_dir / "true.npy", mmap_mode="r")

    schema = pa.schema(
        [
            ("dataset", pa.string()),
            ("model", pa.string()),
            ("setting", pa.string()),
            ("series_id", pa.string()),
            ("window_id", pa.int64()),
            ("horizon_idx", pa.int64()),
            ("target_idx", pa.int64()),
            ("timestamp", pa.string()),
            ("y_true", pa.float64()),
            ("y_pred", pa.float64()),
            ("q10", pa.float64()),
            ("q50", pa.float64()),
            ("q90", pa.float64()),
        ]
    )
    output_dir = ensure_dir(output_dir)
    pq_writer = pq.ParquetWriter(str(output_dir / "predictions.parquet"), schema=schema)
    try:
        for batch in _iter_baseline_chunks(
            pred_np,
            true_np,
            dataset=dataset,
            model=model,
            setting=setting,
            series_id=series_id,
        ):
            pq_writer.write_table(pa.Table.from_arrays(list(batch.values()), names=list(batch.keys())))
    finally:
        pq_writer.close()

    # Compute metrics via common.compute_metrics so baseline rows share the same
    # standardized metric space as zero-shot rows in the summary CSVs.
    pred_df = pd.read_parquet(output_dir / "predictions.parquet", columns=["y_true", "y_pred"])
    metrics = compute_metrics(
        y_true=pred_df["y_true"].to_numpy(dtype=float),
        y_pred=pred_df["y_pred"].to_numpy(dtype=float),
        dataset=dataset,
        runtime_sec=float(runtime_sec),
        peak_vram_mb=None,
        mode=mode,
    )
    runtime = {"runtime_sec": runtime_sec, "mode": mode, "source_dir": str(source_dir)}
    metrics_row = {"dataset": dataset, "model": model, "setting": setting}
    metrics_row.update(metrics)
    pd.DataFrame([metrics_row]).to_csv(output_dir / "metrics.csv", index=False)
    write_json(output_dir / "runtime.json", runtime)


def normalize_baseline(args) -> None:
    normalize_baseline_func(
        dataset=args.dataset,
        model=args.model,
        setting=args.setting,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        runtime_sec=args.runtime_sec,
        series_id=args.series_id,
        mode=args.mode,
    )


def merge_run_dirs_func(
    *,
    dataset: str,
    model: str,
    setting: str,
    run_root: Path | str,
    output_dir: Path | str,
    mode: str = "baseline_train",
) -> None:
    """Merge per-series normalized outputs into a single metrics.csv + predictions.parquet."""
    run_root = Path(run_root)
    prediction_files = [path for path in run_root.rglob("predictions.parquet") if path.is_file()]
    if not prediction_files:
        raise RuntimeError(f"No normalized run outputs found under {run_root}")

    predictions = pd.concat([pd.read_parquet(path) for path in prediction_files], ignore_index=True)
    q10 = predictions["q10"].to_numpy(dtype=float) if "q10" in predictions and predictions["q10"].notna().any() else None
    q50 = predictions["q50"].to_numpy(dtype=float) if "q50" in predictions and predictions["q50"].notna().any() else None
    q90 = predictions["q90"].to_numpy(dtype=float) if "q90" in predictions and predictions["q90"].notna().any() else None
    runtimes = [load_json(path.parent / "runtime.json") for path in prediction_files if (path.parent / "runtime.json").exists()]
    runtime_sec = sum(float(item.get("runtime_sec", 0.0) or 0.0) for item in runtimes)
    metrics = compute_metrics(
        y_true=predictions["y_true"].to_numpy(dtype=float),
        y_pred=predictions["y_pred"].to_numpy(dtype=float),
        q10=q10,
        q50=q50,
        q90=q90,
        dataset=dataset,
        runtime_sec=runtime_sec,
        peak_vram_mb=None,
        mode=mode,
    )
    runtime = {"runtime_sec": runtime_sec, "mode": mode, "num_children": len(prediction_files)}
    write_outputs(
        output_dir=output_dir,
        dataset=dataset,
        model=model,
        setting=setting,
        metrics=metrics,
        predictions=predictions,
        runtime=runtime,
    )


def merge_run_dirs(args) -> None:
    merge_run_dirs_func(
        dataset=args.dataset,
        model=args.model,
        setting=args.setting,
        run_root=args.run_root,
        output_dir=args.output_dir,
        mode=args.mode,
    )


def summarize_func(results_root: Path | str) -> None:
    """Aggregate all metrics.csv files under results_root into summary CSVs."""
    results_root = Path(results_root)
    metrics_files = [path for path in results_root.rglob("metrics.csv") if path.is_file()]
    if not metrics_files:
        raise RuntimeError(f"No metrics.csv files found under {results_root}")
    frame = pd.concat([pd.read_csv(path) for path in metrics_files], ignore_index=True)
    frame.to_csv(results_root / "summary_main.csv", index=False)
    frame[frame["pinball_q50"].notna()].to_csv(results_root / "summary_probabilistic.csv", index=False)
    frame[frame["mode"].isin(["text_only", "image_only", "text_image"])].to_csv(
        results_root / "summary_ablation.csv",
        index=False,
    )


def summarize(args) -> None:
    summarize_func(args.results_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and summarize Project Two outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("normalize-baseline")
    baseline.add_argument("--dataset", required=True)
    baseline.add_argument("--model", required=True)
    baseline.add_argument("--setting", required=True)
    baseline.add_argument("--source-dir", required=True)
    baseline.add_argument("--output-dir", required=True)
    baseline.add_argument("--runtime-sec", type=float, default=0.0)
    baseline.add_argument("--series-id", default="series")
    baseline.add_argument("--mode", default="baseline_train")

    merge = subparsers.add_parser("merge-run-dirs")
    merge.add_argument("--dataset", required=True)
    merge.add_argument("--model", required=True)
    merge.add_argument("--setting", required=True)
    merge.add_argument("--run-root", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--mode", default="baseline_train")

    summarize_cmd = subparsers.add_parser("summarize")
    summarize_cmd.add_argument("--results-root", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "normalize-baseline":
        normalize_baseline(args)
    elif args.command == "merge-run-dirs":
        merge_run_dirs(args)
    elif args.command == "summarize":
        summarize(args)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
