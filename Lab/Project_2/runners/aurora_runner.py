from __future__ import annotations

import argparse
import sys
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


def load_image_tensor(image_path: Path, image_size: int = 224) -> torch.Tensor:
    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor.unsqueeze(0)


def evaluate_unimodal(model, df: pd.DataFrame, history: int, forecast: int, device: torch.device,
                       batch_size: int = 1024):
    date_col = "date"
    numeric_cols = [col for col in df.columns if col != date_col and pd.api.types.is_numeric_dtype(df[col])]
    records = []
    y_true_all = []
    y_pred_all = []
    q10_all = []
    q50_all = []
    q90_all = []

    max_windows = getattr(evaluate_unimodal, "_max_windows", 0) or 0
    window_counter = 0

    # Temporary batch buffers
    batch_contexts: list[np.ndarray] = []
    batch_meta: list[tuple[str, int, np.ndarray, list]] = []  # (series_id, window_id, truth, timestamps)

    def _flush_batch() -> None:
        """Run generate on the accumulated batch and record results."""
        nonlocal window_counter
        if not batch_contexts:
            return
        B = len(batch_contexts)
        inputs = torch.tensor(np.stack(batch_contexts, axis=0), dtype=torch.float32, device=device)
        samples = model.generate(
            inputs=inputs,
            max_output_length=forecast,
            num_samples=20,
            inference_token_len=min(48, history),
        ).detach().cpu().numpy()  # shape: (B, num_samples, forecast)
        for i in range(B):
            series_id, window_id, truth, timestamps = batch_meta[i]
            s = samples[i]  # (num_samples, forecast)
            q10 = np.quantile(s, 0.1, axis=0)
            q50 = np.quantile(s, 0.5, axis=0)
            q90 = np.quantile(s, 0.9, axis=0)
            mean = s.mean(axis=0)
            y_true_all.append(truth)
            y_pred_all.append(mean)
            q10_all.append(q10)
            q50_all.append(q50)
            q90_all.append(q90)
            for horizon_idx, timestamp in enumerate(timestamps):
                records.append({
                    "series_id": series_id,
                    "window_id": window_id,
                    "horizon_idx": horizon_idx,
                    "target_idx": 0,
                    "timestamp": timestamp,
                    "y_true": float(truth[horizon_idx]),
                    "y_pred": float(mean[horizon_idx]),
                    "q10": float(q10[horizon_idx]),
                    "q50": float(q50[horizon_idx]),
                    "q90": float(q90[horizon_idx]),
                })
            window_counter += 1
        batch_contexts.clear()
        batch_meta.clear()

    for series_id in numeric_cols:
        values = df[series_id].to_numpy(dtype=np.float32)
        timestamps = pd.to_datetime(df[date_col]).tolist()
        for window_id, start in enumerate(rolling_test_starts(len(df), history, forecast)):
            end = start + history
            future_end = end + forecast
            if future_end > len(df):
                continue
            if max_windows > 0 and window_counter >= max_windows:
                _flush_batch()
                return records, np.stack(y_true_all), np.stack(y_pred_all), np.stack(q10_all), np.stack(q50_all), np.stack(q90_all)
            batch_contexts.append(values[start:end])
            batch_meta.append((series_id, window_id, values[end:future_end], timestamps[end:future_end]))
            if len(batch_contexts) >= batch_size:
                _flush_batch()

    # Flush remaining
    _flush_batch()
    return records, np.stack(y_true_all), np.stack(y_pred_all), np.stack(q10_all), np.stack(q50_all), np.stack(q90_all)


def evaluate_multimodal_single_csv(
    model,
    csv_path: Path,
    *,
    history: int,
    forecast: int,
    device: torch.device,
    image_root: Path,
    series_id: str,
    batch_size: int = 1024,
):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = [col for col in df.columns if col not in {"date", "fact", "start_date", "end_date", "image_path", "prior_history_avg"}]
    numeric_cols = [col for col in numeric_cols if pd.api.types.is_numeric_dtype(df[col])]
    numeric_cols = [col for col in numeric_cols if col != "OT"] + ["OT"]

    text_tokenizer = model.tokenizer
    target_idx = len(numeric_cols) - 1
    records = []
    y_true_all = []
    y_pred_all = []
    q10_all = []
    q50_all = []
    q90_all = []

    max_windows = getattr(evaluate_multimodal_single_csv, "_max_windows", 0) or 0
    nf = len(numeric_cols)
    window_counter = 0

    # Batch buffers: each entry is a window's pre-computed data
    batch_inputs: list[torch.Tensor] = []      # each (nf, history)
    batch_text_ids: list[torch.Tensor] = []    # each (nf, text_len)
    batch_text_mask: list[torch.Tensor] = []
    batch_text_tti: list[torch.Tensor] = []
    batch_vision: list[torch.Tensor] = []      # each (nf, 3, 224, 224) or None
    batch_meta: list[tuple[int, np.ndarray, list]] = []  # (window_id, truth, timestamps)

    def _flush_batch() -> None:
        nonlocal window_counter
        if not batch_inputs:
            return
        B = len(batch_inputs)
        inputs = torch.cat(batch_inputs, dim=0)                     # (B*nf, history)
        text_ids = torch.cat(batch_text_ids, dim=0)                  # (B*nf, text_len)
        text_mask = torch.cat(batch_text_mask, dim=0)
        text_tti = torch.cat(batch_text_tti, dim=0)
        # Vision: build uniform tensor — None → zeros
        vision_list = []
        for vt in batch_vision:
            if vt is not None:
                vision_list.append(vt)
            else:
                vision_list.append(torch.zeros(nf, 3, 224, 224, device=device))
        vision = torch.cat(vision_list, dim=0) if any(vt is not None for vt in batch_vision) else None

        outputs = model.generate(
            inputs=inputs,
            text_input_ids=text_ids,
            text_attention_mask=text_mask,
            text_token_type_ids=text_tti,
            vision_inputs=vision,
            max_output_length=forecast,
            num_samples=20,
            inference_token_len=min(48, history),
        )
        # outputs dict: each key corresponds to one feature per window
        # target_idx = last feature (OT). For window i, OT output is at index i*nf + target_idx
        for i in range(B):
            window_id, truth, timestamps = batch_meta[i]
            target_samples = outputs[i * nf + target_idx].detach().cpu().numpy()  # (num_samples, forecast)
            q10_v = np.quantile(target_samples, 0.1, axis=0)
            q50_v = np.quantile(target_samples, 0.5, axis=0)
            q90_v = np.quantile(target_samples, 0.9, axis=0)
            mean = target_samples.mean(axis=0)
            y_true_all.append(truth)
            y_pred_all.append(mean)
            q10_all.append(q10_v)
            q50_all.append(q50_v)
            q90_all.append(q90_v)
            for horizon_idx, timestamp in enumerate(timestamps):
                records.append({
                    "series_id": series_id,
                    "window_id": window_id,
                    "horizon_idx": horizon_idx,
                    "target_idx": 0,
                    "timestamp": timestamp,
                    "y_true": float(truth[horizon_idx]),
                    "y_pred": float(mean[horizon_idx]),
                    "q10": float(q10_v[horizon_idx]),
                    "q50": float(q50_v[horizon_idx]),
                    "q90": float(q90_v[horizon_idx]),
                })
            window_counter += 1
        batch_inputs.clear()
        batch_text_ids.clear()
        batch_text_mask.clear()
        batch_text_tti.clear()
        batch_vision.clear()
        batch_meta.clear()

    for window_id, start in enumerate(rolling_test_starts(len(df), history, forecast)):
        end = start + history
        future_end = end + forecast
        if future_end > len(df):
            continue
        if max_windows > 0 and window_counter >= max_windows:
            _flush_batch()
            return records, np.stack(y_true_all), np.stack(y_pred_all), np.stack(q10_all), np.stack(q50_all), np.stack(q90_all)

        seq = torch.tensor(df.iloc[start:end][numeric_cols].to_numpy(dtype=np.float32), device=device).unsqueeze(0)
        batch_x = seq.permute(0, 2, 1).reshape(nf, history)  # (nf, history)

        text = str(df.iloc[end - 1]["fact"])
        tokenized = text_tokenizer(
            text, padding="max_length", truncation=True, max_length=200, return_tensors="pt",
        )
        ids = tokenized["input_ids"].to(device).repeat(nf, 1)
        mask = tokenized["attention_mask"].to(device).repeat(nf, 1)
        tti = tokenized.get("token_type_ids", torch.zeros_like(tokenized["input_ids"])).to(device).repeat(nf, 1)

        image_tensor = None
        if "image_path" in df.columns and pd.notna(df.iloc[end - 1]["image_path"]):
            image_path = Path(str(df.iloc[end - 1]["image_path"]))
            if not image_path.is_absolute():
                image_path = image_root / image_path
            if image_path.exists():
                image_tensor = load_image_tensor(image_path).to(device).repeat(nf, 1, 1, 1)

        batch_inputs.append(batch_x)
        batch_text_ids.append(ids)
        batch_text_mask.append(mask)
        batch_text_tti.append(tti)
        batch_vision.append(image_tensor)
        batch_meta.append((window_id, df.iloc[end:future_end]["OT"].to_numpy(dtype=np.float32),
                           df.iloc[end:future_end]["date"].tolist()))

        if len(batch_inputs) >= batch_size:
            _flush_batch()

    _flush_batch()
    return records, np.stack(y_true_all), np.stack(y_pred_all), np.stack(q10_all), np.stack(q50_all), np.stack(q90_all)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurora evaluator")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Inference batch size for window batching (default: 256)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from aurora.modeling_aurora import AuroraForPrediction

    tied_keys = getattr(AuroraForPrediction, "_tied_weights_keys", None)
    if not isinstance(tied_keys, dict):
        tied_keys = {}
    AuroraForPrediction.all_tied_weights_keys = tied_keys

    metadata = load_metadata(args.metadata)
    history, forecast = parse_setting(args.setting)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reset_peak_vram()
    started = timer()

    model = AuroraForPrediction.from_pretrained(args.model_path).to(device)
    model.eval()

    if args.dataset == "electricity":
        df = pd.read_csv(metadata["baseline_csv"])
        # Electricity is multivariate (321 features) but Aurora is univariate —
        # evaluate only the target column to avoid 321x the computation
        target_col = metadata.get("target_column", "OT")
        df = df[["date", target_col]]
        evaluate_unimodal._max_windows = args.max_windows
        records, y_true, y_pred, q10, q50, q90 = evaluate_unimodal(
            model, df, history, forecast, device, batch_size=args.batch_size
        )
        mode = "unimodal_zero_shot"
    else:
        echo_key = f"echo_{args.setting}"
        if echo_key not in metadata:
            # Fallback: try legacy hardcoded keys for backward compatibility
            if args.setting == "H60_F1":
                echo_key = "echo_H60_F1"
            elif args.setting == "H120_F5":
                echo_key = "echo_H120_F5"
            elif args.setting == "H120_F7":
                echo_key = "echo_H120_F7"
            else:
                raise KeyError(f"Metadata missing echo key for setting {args.setting} (tried {echo_key})")
        manifest_path = metadata[echo_key]
        manifest = pd.read_csv(manifest_path)
        if {"series_id", "csv_path"}.issubset(manifest.columns):
            series_rows = manifest.to_dict(orient="records")
        else:
            series_rows = [{"series_id": args.dataset, "csv_path": manifest_path}]

        image_root = Path(metadata.get("image_root", Path(manifest_path).parent))
        all_records = []
        y_true_list = []
        y_pred_list = []
        q10_list = []
        q50_list = []
        q90_list = []
        for row in series_rows:
            evaluate_multimodal_single_csv._max_windows = args.max_windows
            result = evaluate_multimodal_single_csv(
                model,
                Path(row["csv_path"]),
                history=history,
                forecast=forecast,
                device=device,
                image_root=image_root,
                series_id=row["series_id"],
                batch_size=args.batch_size,
            )
            recs, y_true_s, y_pred_s, q10_s, q50_s, q90_s = result
            all_records.extend(recs)
            y_true_list.append(y_true_s)
            y_pred_list.append(y_pred_s)
            q10_list.append(q10_s)
            q50_list.append(q50_s)
            q90_list.append(q90_s)
        records = all_records
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        q10 = np.concatenate(q10_list, axis=0)
        q50 = np.concatenate(q50_list, axis=0)
        q90 = np.concatenate(q90_list, axis=0)
        mode = "multimodal_zero_shot"

    runtime_sec = elapsed_sec(started)
    vram_mb = peak_vram_mb()
    predictions = make_prediction_frame(
        dataset=args.dataset,
        model="Aurora",
        setting=args.setting,
        records=records,
    )
    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        q10=q10,
        q50=q50,
        q90=q90,
        dataset=args.dataset,
        runtime_sec=runtime_sec,
        peak_vram_mb=vram_mb,
        mode=mode,
    )
    runtime = {
        "mode": mode,
        "runtime_sec": runtime_sec,
        "peak_vram_mb": vram_mb,
        "device": str(device),
        "model_path": args.model_path,
    }
    write_outputs(
        output_dir=args.output_dir,
        dataset=args.dataset,
        model="Aurora",
        setting=args.setting,
        metrics=metrics,
        predictions=predictions,
        runtime=runtime,
    )


if __name__ == "__main__":
    main()
