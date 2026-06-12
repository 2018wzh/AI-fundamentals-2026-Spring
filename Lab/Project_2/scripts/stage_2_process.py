#!/usr/bin/env python3
"""Stage 2: Process raw data into evaluation-ready features.

Processes:
  - Electricity: CSV → metadata.json + chronos_df.parquet
  - FNSPID: price + news → per-series baseline CSVs + echo CSVs + images
  - OilETF: daily_panel → baseline CSV + chronos_df + echo CSVs + images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root + runners/ on sys.path; remove auto-added scripts/ to avoid import conflicts
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
for _p in [str(_project_root), str(_project_root / "runners")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_script_dir) in sys.path:
    sys.path.remove(str(_script_dir))

from tqdm import tqdm

from scripts.common import (
    build_common_parent_parser,
    ensure_dir,
    load_config,
    resolve_config,
    should_prepare_dataset,
)

# Direct import of fnspid_builder functions (same Python env)
from runners.fnspid_builder import (
    prepare_electricity,
    prepare_fnspid,
    prepare_oiletf,
)

DEFAULT_OILETF_SAMPLES_H60 = Path("~/OilETF-TimeMMD/data/processed/samples_H60_F1.parquet").expanduser()
DEFAULT_OILETF_SAMPLES_H120 = Path("~/OilETF-TimeMMD/data/processed/samples_H120_F5.parquet").expanduser()
DEFAULT_OILETF_DAILY_PANEL = Path("~/OilETF-TimeMMD/data/processed/daily_panel.parquet").expanduser()
DEFAULT_OILETF_RAW_PRICES = Path("~/OilETF-TimeMMD/data/raw/prices/raw_prices.csv").expanduser()


def run(args: argparse.Namespace) -> None:
    config = resolve_config(load_config(args.config_path), args)
    paths = config["paths"]
    project_root = Path(paths["project_root"])
    builder = project_root / "runners" / "fnspid_builder.py"
    fnspid_top_k = config.get("runtime", {}).get("fnspid_top_k") or 50

    steps: list[tuple[str, callable]] = []

    if should_prepare_dataset("electricity", args):
        elec_dir = Path(config["datasets"]["electricity"]["data_dir"])
        # The download stage puts electricity.csv in the data dir
        candidates = list(elec_dir.glob("*.csv")) + list(elec_dir.glob("**/*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV found in {elec_dir}. Run stage_1_download first.")
        source_csv = candidates[0]
        steps.append((f"Process electricity → {elec_dir}", lambda src=source_csv: (
            prepare_electricity(source_csv=src, output_dir=elec_dir)
        )))

    if should_prepare_dataset("fnspid", args):
        fnspid_dir = Path(config["datasets"]["fnspid"]["data_dir"])
        steps.append((f"Process FNSPID → {fnspid_dir}", lambda: (
            prepare_fnspid(
                repo_root=Path(paths["fnspid_root"]),
                output_dir=fnspid_dir,
                top_k=fnspid_top_k,
                repo_id=config["external_assets"]["fnspid_hf_repo"],
                price_filename=config["external_assets"]["fnspid_price_file"],
                news_filename=config["external_assets"]["fnspid_news_file"],
                download=False,
                skip_images=True,
            )
        )))

    if should_prepare_dataset("oiletf", args):
        oiletf_dir = Path(config["datasets"]["oiletf"]["data_dir"])
        steps.append((f"Process OilETF → {oiletf_dir}", lambda: (
            prepare_oiletf(
                samples_h60=DEFAULT_OILETF_SAMPLES_H60,
                samples_h120=DEFAULT_OILETF_SAMPLES_H120,
                daily_panel=DEFAULT_OILETF_DAILY_PANEL,
                raw_prices=DEFAULT_OILETF_RAW_PRICES,
                output_dir=oiletf_dir,
            )
        )))

    if args.dry_run:
        print("[dry-run] Processing steps that would execute:")
        for label, _ in steps:
            print(f"  • {label}")
        return

    with tqdm(total=len(steps), desc="Processing datasets", unit="step", ncols=100) as pbar:
        for label, fn in steps:
            pbar.write(f"\n[{pbar.n + 1}/{len(steps)}] {label}")
            try:
                fn()
            except Exception as exc:
                pbar.write(f"  ✗ FAILED: {exc}")
                raise
            pbar.update(1)

    print("\n✓ All data processing complete.")


def build_parser() -> argparse.ArgumentParser:
    parent = build_common_parent_parser()
    parser = argparse.ArgumentParser(
        description="Stage 2: Process raw data into evaluation-ready features",
        parents=[parent],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
