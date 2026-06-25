#!/usr/bin/env python3
"""Stage 1: Download external assets (datasets, checkpoints, repos).

Downloads:
  - Electricity CSV from HuggingFace
  - FNSPID price + news data from HuggingFace (git clone repo first)
  - Chronos-2 model snapshot from HuggingFace
  - Aurora model snapshot from HuggingFace
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

import tqdm

from scripts.common import (
    build_common_parent_parser,
    ensure_git_clone,
    ensure_hf_snapshot,
    ensure_python_packages,
    load_config,
    resolve_config,
    should_prepare_dataset,
    should_prepare_model,
)


def download_hf_file(
    python_exe: str | Path,
    *,
    repo_id: str,
    filename: str,
    local_dir: str | Path,
    repo_type: str,
) -> Path:
    """Download a single file from HuggingFace Hub."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import hf_hub_download; "
        f"src = hf_hub_download(repo_id={repr(repo_id)}, filename={repr(filename)}, "
        f"repo_type={repr(repo_type)}, local_dir={repr(str(local_dir))}); print(src)"
    )
    import subprocess
    completed = subprocess.run(
        [str(python_exe), "-c", code],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    for line in output.splitlines():
        if line.strip():
            pass  # could log
    if completed.returncode != 0:
        raise RuntimeError(f"HF download failed: {output}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Failed to resolve downloaded Hugging Face file path")
    return Path(lines[-1])


def run(args: argparse.Namespace) -> None:
    config = resolve_config(load_config(args.config_path), args)
    paths = config["paths"]
    assets = config["external_assets"]
    project_root = Path(paths["project_root"])
    controller_python = paths["controller_python"]
    chronos_python = paths["chronos_python"]
    aurora_python = paths["aurora_python"]
    fnspid_top_k = config.get("runtime", {}).get("fnspid_top_k") or 50

    steps = []
    steps.append(("huggingface_hub & deps", lambda: ensure_python_packages(
        controller_python, ["huggingface_hub", "pandas", "pyarrow", "pillow", "matplotlib"], logger=None
    )))
    steps.append(("chronos deps", lambda: ensure_python_packages(
        chronos_python, ["pandas", "pyarrow"], logger=None
    )))
    if should_prepare_model("Aurora", args):
        steps.append(("aurora deps", lambda: ensure_python_packages(
            aurora_python, ["transformers", "huggingface_hub", "pandas", "pyarrow", "pillow", "matplotlib"], logger=None
        )))

    if should_prepare_model("Chronos-2", args) or should_prepare_model("Chronos-2-ECHO", args):
        ckpt = assets["chronos2_checkpoint"]
        steps.append((f"Chronos-2 snapshot → {ckpt}", lambda: ensure_hf_snapshot(
            controller_python, assets["chronos2_model_id"], ckpt, logger=None
        )))

    if should_prepare_dataset("electricity", args):
        elec_dir = Path(config["datasets"]["electricity"]["data_dir"])
        steps.append((f"Electricity CSV → {elec_dir}", lambda: download_hf_file(
            controller_python,
            repo_id=assets["electricity_hf_repo"],
            filename=assets["electricity_filename"],
            local_dir=elec_dir,
            repo_type="dataset",
        )))

    if should_prepare_dataset("fnspid", args):
        steps.append((f"FNSPID git clone → {paths['fnspid_root']}", lambda: ensure_git_clone(
            assets["fnspid_git"], paths["fnspid_root"], logger=None
        )))
        raw_dir = Path(paths["fnspid_root"]) / "raw"
        steps.append((f"FNSPID price data", lambda: download_hf_file(
            controller_python,
            repo_id=assets["fnspid_hf_repo"],
            filename=assets["fnspid_price_file"],
            local_dir=raw_dir,
            repo_type="dataset",
        )))
        steps.append((f"FNSPID news data", lambda: download_hf_file(
            controller_python,
            repo_id=assets["fnspid_hf_repo"],
            filename=assets["fnspid_news_file"],
            local_dir=raw_dir,
            repo_type="dataset",
        )))

    if should_prepare_model("Aurora", args):
        steps.append((f"Aurora checkpoint → {assets['aurora_checkpoint']}", lambda: ensure_hf_snapshot(
            aurora_python, assets["aurora_hf_repo"], assets["aurora_checkpoint"], logger=None
        )))

    if args.dry_run:
        print("[dry-run] Download steps that would execute:")
        for label, _ in steps:
            print(f"  • {label}")
        return

    with tqdm.tqdm(total=len(steps), desc="Downloading assets", unit="step", ncols=100) as pbar:
        for label, fn in steps:
            pbar.write(f"\n[{pbar.n + 1}/{len(steps)}] {label}")
            try:
                fn()
            except Exception as exc:
                pbar.write(f"  ✗ FAILED: {exc}")
                raise
            pbar.update(1)

    print("\n✓ All downloads complete.")


def build_parser() -> argparse.ArgumentParser:
    parent = build_common_parent_parser()
    parser = argparse.ArgumentParser(
        description="Stage 1: Download external assets (datasets, checkpoints, repos)",
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
