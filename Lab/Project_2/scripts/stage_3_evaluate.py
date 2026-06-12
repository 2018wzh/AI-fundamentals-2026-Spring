#!/usr/bin/env python3
"""Stage 3: Evaluate all models across all datasets.

Iterates the execution_plan from config, dispatching each task to the
appropriate model runner (TSLib for baselines, chronos_python for Chronos-2/ECHO,
aurora_python for Aurora). Uses tqdm for structured progress display and
RunAllStateTracker for resume support.
"""

from __future__ import annotations

import argparse
import sys
import time
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
    BASELINE_MODELS,
    build_common_parent_parser,
    build_selected_tasks,
    ensure_dir,
    format_task_label,
    invoke_normalize_baseline,
    invoke_merge_run_dirs,
    is_task_complete,
    load_config,
    load_json,
    resolve_config,
    run_aurora_experiment,
    run_chronos_experiment,
    run_echo_experiment,
    run_command,
    run_tslib,
    RunAllStateTracker,
    build_tslib_args,
    get_tslib_raw_source,
    task_output_dir,
    task_run_key,
    short_config_desc,
)


def run_baseline_single_csv(
    *,
    args: argparse.Namespace,
    config: dict,
    task: dict,
    final_output_dir: Path,
    pbar: tqdm.tqdm,
    state_tracker: RunAllStateTracker | None = None,
) -> None:
    """Run a baseline model on a single-CSV dataset (electricity, oiletf)."""
    paths = config["paths"]
    project_root = Path(paths["project_root"])
    controller_python = paths["controller_python"]
    tslib_python = paths["tslib_python"]
    tslib_root = Path(paths["tslib_root"])
    dataset_config = config["datasets"][task["dataset"]]
    setting_config = dataset_config["settings"][task["setting"]]
    metadata = load_json(dataset_config["metadata"])
    defaults = config["baseline_defaults"][task["model"]]
    csv_path = Path(metadata["baseline_csv"])

    model_id, ts_args = build_tslib_args(
        dataset=task["dataset"],
        run_key=task_run_key(task),
        model=task["model"],
        setting_config=setting_config,
        metadata=metadata,
        defaults=defaults,
        csv_path=csv_path,
    )

    pbar.write(f"  TSLib args: model_id={model_id}, seq_len={ts_args['seq_len']}, pred_len={ts_args['pred_len']}")
    runtime_sec = run_tslib(tslib_python, tslib_root, ts_args, logger=pbar)

    raw_source = get_tslib_raw_source(tslib_root, model_id=model_id, model_name=task["model"], ts_args=ts_args)
    pbar.write(f"  Normalizing baseline outputs from {raw_source}")
    invoke_normalize_baseline(
        controller_python=controller_python,
        project_root=project_root,
        dataset=task["dataset"],
        model=task["model"],
        setting=task_run_key(task),
        source_dir=str(raw_source),
        output_dir=str(final_output_dir),
        runtime_sec=runtime_sec,
        series_id=task["dataset"],
        mode="baseline_train",
    )
    pbar.write(f"  ✓ Baseline outputs → {final_output_dir}")


def run_baseline_manifest(
    *,
    args: argparse.Namespace,
    config: dict,
    task: dict,
    final_output_dir: Path,
    pbar: tqdm.tqdm,
    state_tracker: RunAllStateTracker | None = None,
) -> None:
    """Run a baseline model on a multi-series manifest dataset (fnspid)."""
    paths = config["paths"]
    project_root = Path(paths["project_root"])
    controller_python = paths["controller_python"]
    tslib_python = paths["tslib_python"]
    tslib_root = Path(paths["tslib_root"])
    dataset_config = config["datasets"][task["dataset"]]
    setting_config = dataset_config["settings"][task["setting"]]
    metadata = load_json(dataset_config["metadata"])
    defaults = config["baseline_defaults"][task["model"]]
    manifest = load_json(metadata["baseline_manifest"])
    series_entries = list(manifest["series"])
    temp_root = ensure_dir(final_output_dir / "by_series")

    pbar.write(f"  Manifest baseline with {len(series_entries)} series")

    for index, entry in enumerate(series_entries, start=1):
        series_id = str(entry["series_id"])
        series_output = temp_root / series_id
        if args.resume and is_task_complete(series_output):
            pbar.write(f"  Resume skip [{index}/{len(series_entries)}] series {series_id}: already complete")
            continue

        csv_path = Path(entry["csv_path"])
        model_id, ts_args = build_tslib_args(
            dataset=task["dataset"],
            run_key=task_run_key(task),
            model=task["model"],
            setting_config=setting_config,
            metadata=metadata,
            defaults=defaults,
            csv_path=csv_path,
            model_id_suffix=series_id,
        )
        pbar.write(f"  [{index}/{len(series_entries)}] Series {series_id}: model_id={model_id}")
        runtime_sec = run_tslib(tslib_python, tslib_root, ts_args, logger=pbar)

        raw_source = get_tslib_raw_source(tslib_root, model_id=model_id, model_name=task["model"], ts_args=ts_args)
        series_output = ensure_dir(series_output)
        invoke_normalize_baseline(
            controller_python=controller_python,
            project_root=project_root,
            dataset=task["dataset"],
            model=task["model"],
            setting=task_run_key(task),
            source_dir=str(raw_source),
            output_dir=str(series_output),
            runtime_sec=runtime_sec,
            series_id=series_id,
            mode="baseline_train",
        )

    if args.resume and is_task_complete(final_output_dir):
        pbar.write(f"  Resume skip merge: final outputs already complete")
        return

    pbar.write("  Merging per-series outputs...")
    invoke_merge_run_dirs(
        controller_python=controller_python,
        project_root=project_root,
        dataset=task["dataset"],
        model=task["model"],
        setting=task_run_key(task),
        run_root=str(temp_root),
        output_dir=str(final_output_dir),
        mode="baseline_train",
    )
    pbar.write(f"  ✓ Merged baseline outputs → {final_output_dir}")


def run_evaluate(args: argparse.Namespace) -> None:
    config = resolve_config(load_config(args.config_path), args)
    paths = config["paths"]
    project_root = Path(paths["project_root"])

    # Determine which tasks to run
    selected_tasks = build_selected_tasks(config, args)
    if not selected_tasks:
        print("No tasks matched the current filters.")
        return

    print(f"\n{'='*80}")
    print(f"Stage 3: Evaluation")
    print(f"Config: {args.config_path}")
    print(f"Tasks selected: {len(selected_tasks)} — {short_config_desc(config)}")
    print(f"Resume: {args.resume}")
    print(f"{'='*80}\n")

    if args.list_tasks:
        print("Selected tasks:")
        for idx, task in enumerate(selected_tasks, start=1):
            print(f"  [{idx:3d}] {format_task_label(task)}")
        return

    if args.dry_run:
        print("[dry-run] Tasks that would execute:")
        for idx, task in enumerate(selected_tasks, start=1):
            print(f"  [{idx:3d}] {format_task_label(task)}")
        return

    # Initialize state tracker
    state_tracker = RunAllStateTracker(
        project_root=project_root,
        config_path=args.config_path,
        args=args,
        selected_tasks=selected_tasks,
    )
    state_tracker.set_summary_status("running")

    executed_count = 0
    skipped_count = 0
    failed_count = 0
    started_at = time.perf_counter()

    with tqdm.tqdm(total=len(selected_tasks), desc="Evaluating", unit="task", ncols=100) as pbar:
        for index, task in enumerate(selected_tasks, start=1):
            task_label = format_task_label(task)
            output_dir = task_output_dir(project_root, task)

            # Resume skip
            if args.resume and is_task_complete(output_dir):
                skipped_count += 1
                state_tracker.set_task_status(task, "skipped", message="outputs already complete")
                pbar.write(f"[{index}/{len(selected_tasks)}] ⏭  {task_label} (already complete)")
                pbar.update(1)
                continue

            # Prepare task_args Namespace for downstream functions
            task_args = argparse.Namespace(
                dataset=task["dataset"],
                model=task["model"],
                setting=task["setting"],
                mode=task.get("mode", ""),
                config_path=args.config_path,
                resume=bool(args.resume),
            )
            ensure_dir(output_dir)
            state_tracker.set_task_status(task, "running", message="task started")
            pbar.write(f"\n[{index}/{len(selected_tasks)}] ▶  {task_label}")
            pbar.write(f"    Output: {output_dir}")

            task_started = time.perf_counter()
            model = task["model"]

            try:
                if model in BASELINE_MODELS:
                    dataset_config = config["datasets"][task["dataset"]]
                    metadata = load_json(dataset_config["metadata"])
                    baseline_mode = metadata.get("baseline_mode", "single_csv")
                    if baseline_mode == "manifest":
                        run_baseline_manifest(
                            args=task_args, config=config, task=task,
                            final_output_dir=output_dir, pbar=pbar,
                            state_tracker=state_tracker,
                        )
                    else:
                        run_baseline_single_csv(
                            args=task_args, config=config, task=task,
                            final_output_dir=output_dir, pbar=pbar,
                            state_tracker=state_tracker,
                        )

                elif model == "Chronos-2":
                    chronos_model = config["external_assets"]["chronos2_model_id"]
                    ckpt_key = "chronos2_checkpoint"
                    if ckpt_key in config["external_assets"] and Path(config["external_assets"][ckpt_key]).exists():
                        chronos_model = config["external_assets"][ckpt_key]
                    run_chronos_experiment(
                        chronos_python=paths["chronos_python"],
                        runner_path=project_root / "runners" / "chronos_runner.py",
                        dataset=task["dataset"],
                        setting=task_run_key(task),
                        metadata_path=config["datasets"][task["dataset"]]["metadata"],
                        output_dir=output_dir,
                        model_id=chronos_model,
                        batch_size=128,
                        logger=pbar,
                    )

                elif model == "Chronos-2-ECHO":
                    mode = task.get("mode", "text_only")
                    echo_model = config["external_assets"].get("chronos2_echo_model_path")
                    echo_cfg = config["external_assets"].get("chronos2_echo_config")

                    if mode == "training":
                        # Fine-tune Echo adapter; the runner resolves manifest CSVs internally.
                        dataset_config = config["datasets"][task["dataset"]]
                        metadata = load_json(dataset_config["metadata"])
                        echo_setting = task["setting"]
                        echo_data_key = f"echo_{echo_setting}"
                        data_path = metadata.get(echo_data_key)
                        if not data_path:
                            raise RuntimeError(f"Metadata missing {echo_data_key} for {task['dataset']}")

                        echo_model = echo_model or "amazon/chronos-2"
                        run_command([
                            str(paths["chronos_python"]),
                            str(project_root / "runners" / "chronos_echo_runner.py"),
                            "train",
                            "--dataset", task["dataset"],
                            "--setting", echo_setting,
                            "--metadata", str(dataset_config["metadata"]),
                            "--output-dir", str(output_dir),
                            "--model-path", echo_model,
                            "--data-path", str(data_path),
                            "--batch-size", "8",
                            "--num-steps", "20000",
                            "--warmup-steps", "2000",
                            "--echo-config", echo_cfg if echo_cfg else "{}",
                            "--finetune-mode", "lora",
                            "--learning-rate", "2e-5",
                        ], logger=pbar)
                    else:
                        # Auto-detect finetuned checkpoint: if training has completed
                        # for this dataset/setting, use the finetuned adapter weights
                        # instead of the base zero-shot model.
                        ft_checkpoint = (
                            project_root / "results" / task["dataset"] / "Chronos-2-ECHO"
                            / f"{task['setting']}_training" / "finetuned-ckpt"
                        )
                        model_path = echo_model
                        if ft_checkpoint.is_dir() and (ft_checkpoint / "adapter_model.safetensors").is_file():
                            model_path = str(ft_checkpoint)
                            pbar.write(f"  Using finetuned checkpoint: {model_path}")
                        else:
                            model_path = echo_model or "amazon/chronos-2"
                            pbar.write(f"  Using base model: {model_path}")

                        run_echo_experiment(
                            chronos_python=paths["chronos_python"],
                            runner_path=project_root / "runners" / "chronos_echo_runner.py",
                            dataset=task["dataset"],
                            setting=task["setting"],
                            metadata_path=config["datasets"][task["dataset"]]["metadata"],
                            output_dir=output_dir,
                            mode=mode,
                            model_path=model_path,
                            batch_size=64,
                            echo_config=echo_cfg,
                            logger=pbar,
                        )

                elif model == "Aurora":
                    run_aurora_experiment(
                        aurora_python=paths["aurora_python"],
                        runner_path=project_root / "runners" / "aurora_runner.py",
                        dataset=task["dataset"],
                        setting=task["setting"],
                        metadata_path=config["datasets"][task["dataset"]]["metadata"],
                        output_dir=output_dir,
                        aurora_root=paths["aurora_root"],
                        model_path=config["external_assets"]["aurora_checkpoint"],
                        batch_size=256,
                        logger=pbar,
                    )

                else:
                    raise RuntimeError(f"Unsupported model: {model}")

                elapsed = time.perf_counter() - task_started
                executed_count += 1
                state_tracker.set_task_status(task, "completed", message="task completed")
                pbar.write(f"  ✓ Completed in {elapsed:.1f}s")

            except Exception as exc:
                failed_count += 1
                state_tracker.set_task_status(task, "failed", message=str(exc))
                pbar.write(f"  ✗ FAILED: {exc}")
                # Don't stop the whole batch — continue with remaining tasks
                # Update pbar so task count is accurate
                pbar.update(1)
                continue

            pbar.update(1)

    # Summary
    elapsed_total = time.perf_counter() - started_at
    print(f"\n{'─'*80}")
    print(f"Evaluation summary: {executed_count} executed, {skipped_count} skipped, {failed_count} failed")
    print(f"Total time: {elapsed_total:.1f}s")
    print(f"{'─'*80}")

    state_tracker.set_summary_status(
        "completed" if failed_count == 0 else "partial_failure",
        message=f"{executed_count} executed, {skipped_count} skipped, {failed_count} failed, {elapsed_total:.1f}s",
    )


def build_parser() -> argparse.ArgumentParser:
    parent = build_common_parent_parser()
    parser = argparse.ArgumentParser(
        description="Stage 3: Evaluate all models across all datasets (with tqdm progress)",
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
        run_evaluate(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
