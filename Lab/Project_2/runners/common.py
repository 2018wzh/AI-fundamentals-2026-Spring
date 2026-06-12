from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


FINANCIAL_DATASETS = {"fnspid", "oiletf"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def flatten_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_up = y_true > 0
    pred_up = y_pred > 0
    tp = float(np.sum(true_up & pred_up))
    fp = float(np.sum(~true_up & pred_up))
    fn = float(np.sum(true_up & ~pred_up))
    if tp == 0.0:
        return 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2.0 * precision * recall / (precision + recall + 1e-8))


def compute_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset: str,
    q10: np.ndarray | None = None,
    q50: np.ndarray | None = None,
    q90: np.ndarray | None = None,
    runtime_sec: float | None = None,
    peak_vram_mb: float | None = None,
    mode: str = "default",
) -> dict[str, Any]:
    y_true_f = flatten_array(y_true)
    y_pred_f = flatten_array(y_pred)
    mse = float(np.mean((y_true_f - y_pred_f) ** 2))
    mae = float(np.mean(np.abs(y_true_f - y_pred_f)))
    rmse = float(np.sqrt(mse))
    result = {
        "mode": mode,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "smape": smape(y_true_f, y_pred_f),
        "pinball_q10": safe_float(pinball_loss(y_true_f, flatten_array(q10), 0.1)) if q10 is not None else None,
        "pinball_q50": safe_float(pinball_loss(y_true_f, flatten_array(q50), 0.5)) if q50 is not None else None,
        "pinball_q90": safe_float(pinball_loss(y_true_f, flatten_array(q90), 0.9)) if q90 is not None else None,
        "directional_accuracy": None,
        "f1_up_down": None,
        "runtime_sec": safe_float(runtime_sec),
        "peak_vram_mb": safe_float(peak_vram_mb),
    }
    if dataset.lower() in FINANCIAL_DATASETS:
        result["directional_accuracy"] = directional_accuracy(y_true_f, y_pred_f)
        result["f1_up_down"] = binary_f1(y_true_f, y_pred_f)
    return result


def peak_vram_mb() -> float | None:
    if torch is None or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def reset_peak_vram() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def timer() -> float:
    return time.perf_counter()


def elapsed_sec(start_time: float) -> float:
    return float(time.perf_counter() - start_time)


def write_outputs(
    *,
    output_dir: str | Path,
    dataset: str,
    model: str,
    setting: str,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    runtime: dict[str, Any],
) -> None:
    output_dir = ensure_dir(output_dir)
    metrics_row = {"dataset": dataset, "model": model, "setting": setting}
    metrics_row.update(metrics)
    pd.DataFrame([metrics_row]).to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    write_json(output_dir / "runtime.json", runtime)


def make_prediction_frame(
    *,
    dataset: str,
    model: str,
    setting: str,
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "setting",
                "series_id",
                "window_id",
                "horizon_idx",
                "target_idx",
                "timestamp",
                "y_true",
                "y_pred",
                "q10",
                "q50",
                "q90",
            ]
        )
    frame.insert(0, "setting", setting)
    frame.insert(0, "model", model)
    frame.insert(0, "dataset", dataset)
    return frame


def rolling_test_starts(length: int, seq_len: int, pred_len: int) -> range:
    num_train = int(length * 0.7)
    num_test = int(length * 0.2)
    test_start = max(0, length - num_test - seq_len)
    return range(test_start, max(test_start, length - pred_len + 1))


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def save_metadata(metadata_path: str | Path, metadata: dict[str, Any]) -> None:
    ensure_dir(Path(metadata_path).parent)
    write_json(metadata_path, metadata)
