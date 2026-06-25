#!/usr/bin/env python3
"""Project Two evaluation orchestrator.

Runs evaluation stages sequentially or picks a specific stage.

Usage:
    python scripts/run_all.py                          # run all stages 1→4
    python scripts/run_all.py --stage 3                # evaluate only
    python scripts/run_all.py --stage 1 --dry-run      # preview downloads
    python scripts/run_all.py --stage 3 --datasets electricity --models DLinear
    python scripts/run_all.py prepare-assets           # legacy shortcut for stage 1+2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import tqdm

# ── sys.path bootstrap (same pattern as stage scripts) ────────────────────
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
for _p in [str(_project_root), str(_project_root / "runners")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_script_dir) in sys.path:
    sys.path.remove(str(_script_dir))

from scripts.common import (
    build_common_parent_parser,
    load_config,
    resolve_config,
)

STAGES = {
    "1": ("stage_1_download", "Download external assets"),
    "2": ("stage_2_process", "Process raw data into features"),
    "3": ("stage_3_evaluate", "Run evaluation across all datasets/models"),
    "4": ("stage_4_collect", "Collect and summarize metrics"),
    "5": ("stage_5_convert_simtrading", "Convert predictions → SimTrading format"),
}

STAGE_ORDER = ["1", "2", "3", "4", "5"]


def run_stage(stage_id: str, argv: list[str], pbar: tqdm.tqdm | None = None) -> int:
    """Run a single stage script as a subprocess (isolated Python env)."""
    module, desc = STAGES[stage_id]
    script = _script_dir / f"{module}.py"
    if not script.exists():
        print(f"ERROR: Stage script not found: {script}", file=sys.stderr)
        return 1

    if pbar is not None:
        pbar.write(f"\n{'=' * 80}")
        pbar.write(f"Stage {stage_id}: {desc}")
        pbar.write(f"{'=' * 80}")

    import subprocess
    completed = subprocess.run(
        [sys.executable, str(script)] + argv,
        cwd=_project_root,
    )
    return completed.returncode


def run(args: argparse.Namespace) -> int:
    config = resolve_config(load_config(args.config_path), args)
    project_root = Path(config["paths"]["project_root"])

    print(f"\n{'=' * 80}")
    print(f"Project Two Evaluation Orchestrator")
    print(f"Config: {args.config_path}")
    print(f"Project root: {project_root}")
    print(f"{'=' * 80}")

    if args.stage:
        stages_to_run = [s for s in STAGE_ORDER if s == args.stage or (args.stage == "prepare" and s in ("1", "2"))]
    else:
        stages_to_run = STAGE_ORDER

    print(f"Stages to run: {', '.join(s.upper() for s in stages_to_run)}")
    if args.dry_run:
        print("[dry-run mode — no commands will execute]\n")
    print()

    shared_args = []
    if args.config_path is not None:
        shared_args.extend(["--config-path", str(args.config_path)])
    if args.datasets:
        shared_args.extend(["--datasets"] + list(args.datasets))
    if args.models:
        shared_args.extend(["--models"] + list(args.models))
    if args.limit:
        shared_args.extend(["--limit", str(args.limit)])
    if args.resume:
        shared_args.append("--resume")
    if args.dry_run:
        shared_args.append("--dry-run")
    if args.list_tasks:
        shared_args.append("--list-tasks")

    started = time.perf_counter()

    with tqdm.tqdm(total=len(stages_to_run), desc="Overall progress", unit="stage", ncols=100, mininterval=2.0) as pbar:
        for stage_id in stages_to_run:
            module, desc = STAGES[stage_id]
            pbar.set_description(f"Stage {stage_id}: {desc[:40]}")
            rc = run_stage(stage_id, shared_args, pbar=pbar)
            if rc != 0:
                pbar.write(f"\n✗ Stage {stage_id} FAILED with exit code {rc}")
                return rc
            pbar.update(1)

    elapsed = time.perf_counter() - started
    print(f"\n{'=' * 80}")
    print(f"All stages completed in {elapsed:.1f}s.")
    print(f"{'=' * 80}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parent = build_common_parent_parser()
    parser = argparse.ArgumentParser(
        description="Project Two evaluation orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parent],
    )
    parser.add_argument(
        "--stage", "-s", choices=["1", "2", "3", "4", "5", "prepare"],
        help="Run a single stage (1=download, 2=process, 3=evaluate, 4=collect, prepare=1+2)",
    )
    # Legacy command mode
    parser.add_argument("command", nargs="?", default=None,
                        help="Legacy: prepare-assets | run-experiment | collect-metrics | run-all")
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Handle legacy subcommand mode
    if argv and argv[0] in ("prepare-assets",):
        print("→ Legacy 'prepare-assets' maps to --stage prepare")
        parser = build_parser()
        rest = argv[1:]
        args = parser.parse_args(["--stage", "prepare"] + rest)
        return run(args)

    parser = build_parser()
    args = parser.parse_args(argv)

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
