#!/usr/bin/env python3
"""Stage 4: Collect and summarize evaluation metrics.

Scans all results/**/metrics.csv files and produces:
  - summary_main.csv          — all tasks
  - summary_probabilistic.csv — tasks with pinball quantile predictions
  - summary_ablation.csv      — Chronos-2-ECHO ablation modes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path, remove auto-added scripts/ to avoid import conflicts
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_script_dir) in sys.path:
    sys.path.remove(str(_script_dir))

import pandas as pd
from tqdm import tqdm

from scripts.common import (
    load_config,
    resolve_config,
    ensure_dir,
    write_json,
)


def run(args: argparse.Namespace) -> None:
    config = resolve_config(load_config(args.config_path), args)
    project_root = Path(config["paths"]["project_root"])
    results_root = args.results_root or (project_root / "results")
    results_root = Path(results_root)

    print(f"\n{'='*80}")
    print(f"Stage 4: Collect Metrics")
    print(f"Results root: {results_root}")
    print(f"{'='*80}\n")

    # Find all metrics.csv files
    metrics_files = sorted(results_root.rglob("metrics.csv"))
    if not metrics_files:
        print(f"No metrics.csv files found under {results_root}")
        return

    print(f"Found {len(metrics_files)} metrics.csv files")

    if args.dry_run:
        print("[dry-run] Would aggregate and write summary CSVs")
        print(f"  Files:")
        for path in metrics_files:
            rel = path.relative_to(results_root)
            print(f"    • {rel}")
        return

    # Read and concatenate all metrics
    frames: list[pd.DataFrame] = []
    with tqdm(total=len(metrics_files), desc="Reading metrics", unit="file", ncols=100) as pbar:
        for path in metrics_files:
            try:
                df = pd.read_csv(path)
                frames.append(df)
            except Exception as exc:
                pbar.write(f"  ⚠ Failed to read {path}: {exc}")
            pbar.update(1)

    if not frames:
        print("No valid metrics data found.")
        return

    all_metrics = pd.concat(frames, ignore_index=True)
    print(f"\nTotal metric rows: {len(all_metrics)}")

    # Write summaries
    ensure_dir(results_root)

    all_metrics.to_csv(results_root / "summary_main.csv", index=False)
    print(f"  ✓ summary_main.csv ({len(all_metrics)} rows)")

    prob = all_metrics[all_metrics["pinball_q50"].notna()]
    if not prob.empty:
        prob.to_csv(results_root / "summary_probabilistic.csv", index=False)
        print(f"  ✓ summary_probabilistic.csv ({len(prob)} rows)")

    ablation = all_metrics[all_metrics["mode"].isin(["text_only", "image_only", "text_image"])]
    if not ablation.empty:
        ablation.to_csv(results_root / "summary_ablation.csv", index=False)
        print(f"  ✓ summary_ablation.csv ({len(ablation)} rows)")

    # Print per-dataset, per-model summary
    print(f"\n{'─'*80}")
    print("Per-dataset / Per-model summary (MAE / RMSE):")
    print(f"{'─'*80}")
    grouped = all_metrics.groupby(["dataset", "model", "setting"])
    for (ds, mdl, stg), group in grouped:
        modes = group["mode"].unique()
        for mode_val in modes:
            subset = group[group["mode"] == mode_val]
            mae = subset["mae"].mean()
            rmse = subset["rmse"].mean()
            mode_tag = f" / mode={mode_val}" if mode_val not in ("baseline_train", "zero_shot", "") else ""
            print(f"  {ds:15s} / {mdl:20s} / {stg:12s}{mode_tag}  MAE={mae:.6f}  RMSE={rmse:.6f}")

    print(f"\n{'─'*80}")
    print("Stage 4 complete. Results in:", results_root)
    print(f"{'─'*80}")


def build_parser() -> argparse.ArgumentParser:
    from scripts.common import build_common_parent_parser
    parent = build_common_parent_parser()
    parser = argparse.ArgumentParser(
        description="Stage 4: Collect and summarize evaluation metrics",
        parents=[parent],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Override results root (default: <project_root>/results)")
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
