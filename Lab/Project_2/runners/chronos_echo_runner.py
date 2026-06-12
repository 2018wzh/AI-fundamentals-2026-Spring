from __future__ import annotations

import os as _os

# Route HF requests through mirror for network accessibility
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (
    compute_metrics,
    elapsed_sec,
    load_metadata,
    make_prediction_frame,
    peak_vram_mb,
    reset_peak_vram,
    timer,
    write_outputs,
)


def parse_setting(setting: str) -> tuple[int, int]:
    history = int(setting.split("_")[0].removeprefix("H"))
    forecast = int(setting.split("_")[1].removeprefix("F"))
    return history, forecast


def _images_available(csv_path: Path, image_root: Path) -> bool:
    """Check whether image files referenced in a TimeMMD CSV actually exist."""
    try:
        df = pd.read_csv(csv_path, nrows=200)
    except Exception:
        return False
    if "image_path" not in df.columns:
        return False
    for p_str in df["image_path"].dropna():
        p = Path(p_str)
        if not p.is_absolute():
            p = image_root / p
        if p.exists():
            return True
    return False


def _resolve_defaults(echo_config: dict[str, Any], mode: str) -> dict[str, Any]:
    """Inject offline defaults for local tokenizer / ViT resources."""
    # Tokenizer path — always needed because predict_timemmd creates a tokenizer.
    echo_config.setdefault(
        "text_tokenizer_name_or_path",
        "/home/zhw/Aurora/aurora/bert_config",
    )
    if mode in ("image_only", "text_image"):
        echo_config.setdefault(
            "vision_model_name_or_path",
            "/home/zhw/Aurora/local_vit_model",
        )
        echo_config.setdefault("vision_image_size", 224)
        echo_config.setdefault("freeze_vision_backbone", True)
    return echo_config


def load_pipeline(
    model_path: str,
    echo_config: dict[str, Any] | None = None,
    device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    from pathlib import Path
    from chronos import Chronos2EchoConfig, Chronos2EchoPipeline

    checkpoint_path = Path(model_path)
    is_peft_ckpt = (
        checkpoint_path.is_dir()
        and (checkpoint_path / "adapter_config.json").is_file()
        and not (checkpoint_path / "model.safetensors").is_file()
    )

    if echo_config:
        cfg = Chronos2EchoConfig(**echo_config)
    else:
        cfg = None

    if is_peft_ckpt:
        # LoRA adapter checkpoint: load base model + wrap with PEFT adapter.
        # Read the base model id from the adapter_config so evaluation always
        # uses the SAME base model that training used (avoids shape mismatches).
        import json as _json
        from peft import PeftModel

        adapter_config = _json.loads(
            (checkpoint_path / "adapter_config.json").read_text(encoding="utf-8")
        )
        base_model_id = adapter_config.get("base_model_name_or_path", "amazon/chronos-2")

        # Resolve to the local HuggingFace cache when the network is
        # unavailable — the snapshot directory contains config.json +
        # model.safetensors and works as a drop-in model path.
        import os as _os

        _hf_cache = Path(
            _os.path.expanduser(_os.environ.get("HF_HOME", "~/.cache/huggingface")),
            "hub",
        )
        _repo_dir = _hf_cache / ("models--" + base_model_id.replace("/", "--"))
        if _repo_dir.is_dir():
            _snapshots = sorted((_repo_dir / "snapshots").iterdir(), reverse=True)
            if _snapshots:
                base_model_id = str(_snapshots[0])
                print(f"Using cached base model: {base_model_id}")

        base_cfg = cfg if cfg is not None else Chronos2EchoConfig()
        base_pipeline = Chronos2EchoPipeline.from_pretrained(
            base_model_id, echo_config=base_cfg, device_map=device_map,
        )
        model = PeftModel.from_pretrained(base_pipeline.model, str(checkpoint_path))
        return Chronos2EchoPipeline(model=model)
    else:
        if cfg is not None:
            return Chronos2EchoPipeline.from_pretrained(
                model_path, echo_config=cfg, device_map=device_map,
            )
        return Chronos2EchoPipeline.from_pretrained(model_path, device_map=device_map)


# ── Inference helpers ─────────────────────────────────────────────────────


def _eval_predict(
    pipeline,
    csv_path: Path,
    *,
    history: int,
    forecast: int,
    image_root_path: Path | None,
    batch_size: int,
    image_column: str | None = "image_path",
) -> dict[str, np.ndarray]:
    """Plain text + image inference via ``predict_timemmd`` (text_only / text_image).

    Set ``image_column=None`` for text-only inference — avoids loading images and
    prevents ViT image size mismatches when the model config still references a
    vision backbone.
    """
    output = pipeline.predict_timemmd(
        root_path=csv_path.parent,
        data_path=csv_path.name,
        target="OT",
        seq_len=history,
        pred_len=forecast,
        features="MS",
        batch_size=batch_size,
        flag="test",
        image_column=image_column,
        image_root_path=image_root_path,
    )
    quantiles = output["quantiles"].numpy()
    predictions = output["predictions"].numpy()
    targets = output["targets"].numpy()
    quantile_levels = list(pipeline.quantiles)
    q10_idx = min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - 0.1))
    q50_idx = min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - 0.5))
    q90_idx = min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - 0.9))
    return {
        "targets": targets,
        "predictions": predictions,
        "q10": quantiles[..., q10_idx],
        "q50": quantiles[..., q50_idx],
        "q90": quantiles[..., q90_idx],
    }


def _eval_image_only(
    pipeline,
    csv_path: Path,
    *,
    history: int,
    forecast: int,
    image_root_path: Path | None,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Image-only inference — manual batching, **no text** passed to the model."""
    window_dataset, batch_dataset = pipeline._create_timemmd_dataset(
        root_path=csv_path.parent,
        data_path=csv_path.name,
        flag="test",
        seq_len=history,
        pred_len=forecast,
        target="OT",
        features="MS",
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        image_column="image_path",
        image_root_path=image_root_path,
    )
    loader = DataLoader(batch_dataset, batch_size=None, pin_memory=pipeline.model.device.type == "cuda")

    quantile_windows = []
    target_windows = []
    for batch in loader:
        target_idx_ranges = batch.pop("target_idx_ranges")
        future_target = batch.pop("future_target")
        model_inputs = {}
        for key, value in batch.items():
            # Drop ALL text keys — this is the image-only path
            if key.startswith("text_"):
                continue
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(pipeline.model.device)
            else:
                model_inputs[key] = value
        output = pipeline.model(**model_inputs)
        quantile_preds = output.quantile_preds[..., :forecast].cpu()
        for start, end in target_idx_ranges:
            quantile_windows.append(quantile_preds[start:end].permute(2, 0, 1))
            target_windows.append(future_target[start:end].permute(1, 0))

    quantiles = torch.stack(quantile_windows, dim=0).numpy()
    targets = torch.stack(target_windows, dim=0).numpy()
    quantile_levels = list(pipeline.quantiles)
    median_idx = min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - 0.5))
    predictions = quantiles[..., median_idx]
    q10_idx = min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - 0.1))
    q90_idx = min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - 0.9))
    return {
        "targets": targets,
        "predictions": predictions,
        "q10": quantiles[..., q10_idx],
        "q50": quantiles[..., median_idx],
        "q90": quantiles[..., q90_idx],
    }


def collect_records(
    preds: np.ndarray,
    targets: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    *,
    series_id: str,
) -> list[dict[str, Any]]:
    records = []
    for w in range(preds.shape[0]):
        for h in range(preds.shape[1]):
            records.append({
                "series_id": series_id,
                "window_id": w,
                "horizon_idx": h,
                "target_idx": 0,
                "timestamp": None,
                "y_true": float(targets[w, h, 0]),
                "y_pred": float(preds[w, h, 0]),
                "q10": float(q10[w, h, 0]),
                "q50": float(q50[w, h, 0]),
                "q90": float(q90[w, h, 0]),
            })
    return records


# ── Evaluate subcommand ───────────────────────────────────────────────────


def cmd_evaluate(args: argparse.Namespace) -> None:
    metadata = load_metadata(args.metadata)
    history, forecast = parse_setting(args.setting)

    echo_setting = args.setting
    echo_data_key = f"echo_{echo_setting}"
    manifest_path = metadata.get(echo_data_key)
    if not manifest_path:
        raise RuntimeError(f"Metadata missing {echo_data_key} for {args.dataset}")

    manifest = pd.read_csv(manifest_path)
    if {"series_id", "csv_path"}.issubset(manifest.columns):
        series_rows = manifest.to_dict(orient="records")
    else:
        series_rows = [{"series_id": args.dataset, "csv_path": manifest_path}]

    echo_config = json.loads(args.echo_config) if args.echo_config else {}
    echo_config = _resolve_defaults(echo_config, args.mode)

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    reset_peak_vram()
    started = timer()
    pipeline = load_pipeline(args.model_path, echo_config=echo_config, device_map=device_map)

    image_root = Path(metadata.get("image_root", Path(manifest_path).parent))
    first_csv = Path(series_rows[0]["csv_path"])
    has_images = _images_available(first_csv, image_root)
    eval_mode = args.mode
    if eval_mode == "image_only" and not has_images:
        raise FileNotFoundError(
            f"No images found in {image_root} for {args.dataset}/{args.setting}"
        )
    if eval_mode == "text_image" and not has_images:
        print(f"Warning: images not found, falling back to text_only for {args.dataset}/{args.setting}")
        eval_mode = "text_only"

    all_tensors = {"targets": [], "predictions": [], "q10": [], "q50": [], "q90": []}
    records = []

    for row in series_rows:
        csv_path = Path(row["csv_path"])
        series_id = row["series_id"]
        if eval_mode in ("text_only", "text_image"):
            outputs = _eval_predict(
                pipeline, csv_path,
                history=history, forecast=forecast,
                image_root_path=image_root,
                batch_size=args.batch_size,
                image_column="image_path" if eval_mode == "text_image" else None,
            )
        else:  # image_only
            outputs = _eval_image_only(
                pipeline, csv_path,
                history=history, forecast=forecast,
                image_root_path=image_root, batch_size=args.batch_size,
            )
        for key in all_tensors:
            all_tensors[key].append(outputs[key][..., 0])
        records.extend(collect_records(
            outputs["predictions"], outputs["targets"],
            outputs["q10"], outputs["q50"], outputs["q90"],
            series_id=series_id,
        ))

    runtime_sec = elapsed_sec(started)
    vram_mb = peak_vram_mb()
    predictions = make_prediction_frame(
        dataset=args.dataset, model="Chronos-2-ECHO",
        setting=echo_setting, records=records,
    )
    metrics = compute_metrics(
        y_true=np.concatenate(all_tensors["targets"], axis=0),
        y_pred=np.concatenate(all_tensors["predictions"], axis=0),
        q10=np.concatenate(all_tensors["q10"], axis=0),
        q50=np.concatenate(all_tensors["q50"], axis=0),
        q90=np.concatenate(all_tensors["q90"], axis=0),
        dataset=args.dataset, runtime_sec=runtime_sec,
        peak_vram_mb=vram_mb, mode=eval_mode,
    )
    write_outputs(
        output_dir=args.output_dir, dataset=args.dataset,
        model="Chronos-2-ECHO", setting=echo_setting,
        metrics=metrics, predictions=predictions, runtime={
            "mode": eval_mode, "runtime_sec": runtime_sec,
            "peak_vram_mb": vram_mb, "num_series": len(series_rows),
            "device_map": device_map, "model_path": args.model_path,
        },
    )


# ── Train subcommand ──────────────────────────────────────────────────────


def cmd_train(args: argparse.Namespace) -> None:
    """Fine-tune Chronos-2-ECHO on multimodal time series data.

    Supports multi-dataset ConcatDataset, warmup + constant LR, frequency-masked
    reconstruction loss, and frozen ViT vision backbone.
    """
    metadata = load_metadata(args.metadata) if args.metadata else {}
    echo_config = json.loads(args.echo_config) if args.echo_config else {}
    echo_config = _resolve_defaults(echo_config, mode="text_image")

    # ── Resolve data sources ──────────────────────────────────────────────
    data_paths = args.data_path.split(",") if "," in args.data_path else [args.data_path]

    first_path = Path(data_paths[0])
    is_manifest = False
    if first_path.suffix == ".csv" and first_path.is_file():
        manifest = pd.read_csv(first_path)
        if {"series_id", "csv_path"}.issubset(manifest.columns):
            is_manifest = True
            csv_paths = [row["csv_path"] for row in manifest.to_dict(orient="records")]
            first_abs = Path(csv_paths[0])
            root_path = str(first_abs.parent)
            data_paths = [Path(p).name for p in csv_paths]

    if not is_manifest:
        root_path = args.root_path or str(Path(data_paths[0]).parent)
    else:
        root_path = args.root_path or root_path  # keep the resolved root_path

    history, forecast = parse_setting(args.setting)

    from chronos import Chronos2EchoConfig, Chronos2EchoPipeline

    cfg = Chronos2EchoConfig(**echo_config)
    pipeline = Chronos2EchoPipeline.from_pretrained(
        args.model_path, echo_config=cfg,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    reset_peak_vram()
    started = timer()

    pipeline.fit_timemmd(
        root_path=root_path,
        data_path=data_paths if len(data_paths) > 1 else data_paths[0],
        target="OT",
        seq_len=history,
        pred_len=forecast,
        features=args.features,
        batch_size=args.batch_size,
        flag="train",
        validation_flag=args.validation_flag,
        finetune_mode=args.finetune_mode,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        image_column="image_path",
        image_root_path=Path(metadata.get("image_root", ".")) if args.metadata else None,
        image_size=echo_config.get("vision_image_size", 64),
    )

    runtime_sec = elapsed_sec(started)
    vram_mb = peak_vram_mb()
    print(f"Training completed in {runtime_sec:.1f}s, peak VRAM {vram_mb} MB")
    print(f"Fine-tuned model saved to: {args.output_dir}")

    # Write runtime manifest so stage_3 resume detection works
    import json as _json
    runtime = {
        "mode": "training",
        "runtime_sec": runtime_sec,
        "peak_vram_mb": vram_mb,
        "num_steps": args.num_steps,
        "device_map": "cuda" if torch.cuda.is_available() else "cpu",
        "model_path": args.model_path,
        "checkpoint_path": str(Path(args.output_dir) / "finetuned-ckpt"),
    }
    (Path(args.output_dir) / "runtime.json").write_text(
        _json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Argument parser ───────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chronos-2-ECHO runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Zero-shot inference on TimeMMD CSVs")
    eval_parser.add_argument("--dataset", required=True)
    eval_parser.add_argument("--setting", required=True)
    eval_parser.add_argument("--metadata", required=True)
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.add_argument("--mode", required=True, choices=["text_only", "image_only", "text_image"])
    eval_parser.add_argument("--model-path", default="amazon/chronos-2")
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument("--echo-config", type=str, default=None)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Fine-tune Echo adapter on multimodal data")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--setting", required=True)
    train_parser.add_argument("--metadata", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--model-path", default="amazon/chronos-2")
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--echo-config", type=str, default=None)
    train_parser.add_argument("--root-path", type=str, default=None)
    train_parser.add_argument("--data-path", type=str, required=True)
    train_parser.add_argument("--features", type=str, default="MS", choices=["S", "M", "MS"])
    train_parser.add_argument("--validation-flag", type=str, default=None)
    train_parser.add_argument("--finetune-mode", type=str, default="lora",
                              choices=["echo_only", "lora", "full"])
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--num-steps", type=int, default=50000)
    train_parser.add_argument("--warmup-steps", type=int, default=5000)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.0)
    train_parser.add_argument("--lr-scheduler-type", type=str, default="constant",
                              choices=["constant", "linear", "cosine", "constant_with_warmup"])
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

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
