from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import tqdm


# ── Constants ──────────────────────────────────────────────────────────────

BASELINE_MODELS = {"DLinear", "PatchTST", "TimesNet"}
FINANCIAL_DATASETS = {"fnspid", "oiletf"}
DEFAULT_CONFIG_PATH = Path("./configs/experiments.yaml")


# ── Config & JSON helpers ──────────────────────────────────────────────────

def load_config(config_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text + "\n", encoding="utf-8")
    tmp_path.replace(path)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    paths = resolved.setdefault("paths", {})
    for key in (
        "project_root", "controller_python", "tslib_root", "tslib_python",
        "chronos_root", "chronos_python", "aurora_root", "aurora_python",
    ):
        value = getattr(args, key, None)
        if value is not None:
            paths[key] = str(value)
    if getattr(args, "fnspid_top_k", None) is not None:
        resolved.setdefault("runtime", {})["fnspid_top_k"] = int(args.fnspid_top_k)
    return resolved


# ── CLI arg parser (shared) ────────────────────────────────────────────────

def build_common_parent_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--controller-python", type=Path, default=None)
    parser.add_argument("--tslib-root", type=Path, default=None)
    parser.add_argument("--tslib-python", type=Path, default=None)
    parser.add_argument("--chronos-root", type=Path, default=None)
    parser.add_argument("--chronos-python", type=Path, default=None)
    parser.add_argument("--aurora-root", type=Path, default=None)
    parser.add_argument("--aurora-python", type=Path, default=None)
    parser.add_argument("--fnspid-top-k", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0, help="Limit selected tasks before execution")
    parser.add_argument("--list-tasks", action="store_true", help="Print selected tasks and stop")
    parser.add_argument("--dry-run", action="store_true")
    return parser


# ── Task utilities ─────────────────────────────────────────────────────────

def task_run_key(task: dict[str, Any] | argparse.Namespace) -> str:
    setting = str(task.setting) if isinstance(task, argparse.Namespace) else str(task["setting"])
    mode = str(task.mode) if isinstance(task, argparse.Namespace) else str(task.get("mode", ""))
    return f"{setting}_{mode}" if mode else setting


def task_output_dir(project_root: Path, task: dict[str, Any] | argparse.Namespace) -> Path:
    dataset = str(task.dataset) if isinstance(task, argparse.Namespace) else str(task["dataset"])
    model = str(task.model) if isinstance(task, argparse.Namespace) else str(task["model"])
    return project_root / "results" / dataset / model / task_run_key(task)


def is_task_complete(output_dir: str | Path) -> bool:
    output_dir = Path(output_dir)
    task_name = output_dir.name
    if task_name.endswith("_training") or task_name == "training":
        # Training tasks only produce a runtime manifest + checkpoint
        return (output_dir / "runtime.json").is_file()
    required_files = ("metrics.csv", "predictions.parquet", "runtime.json")
    return all((output_dir / name).is_file() for name in required_files)


def format_task_label(task: dict[str, Any]) -> str:
    mode_part = f" / mode={task['mode']}" if "mode" in task and task["mode"] else ""
    return f"{task['dataset']} / {task['model']} / {task['setting']}{mode_part}"


def build_selected_tasks(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task in config["execution_plan"]:
        if args.datasets and str(task["dataset"]) not in args.datasets:
            continue
        if args.models and str(task["model"]) not in args.models:
            continue
        selected.append({
            "dataset": str(task["dataset"]),
            "model": str(task["model"]),
            "setting": str(task["setting"]),
            "mode": str(task.get("mode", "")),
        })
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]
    return selected


def should_prepare_dataset(dataset_name: str, args: argparse.Namespace) -> bool:
    return not args.datasets or dataset_name in args.datasets


def should_prepare_model(model_name: str, args: argparse.Namespace) -> bool:
    return not args.models or model_name in args.models


# ── State tracker (for resume support) ─────────────────────────────────────

def build_state_file_path(
    project_root: Path, args: argparse.Namespace, selected_tasks: Sequence[dict[str, Any]]
) -> Path:
    selection_payload = {
        "config_path": str(args.config_path),
        "datasets": list(args.datasets),
        "models": list(args.models),
        "limit": int(args.limit or 0),
        "tasks": selected_tasks,
    }
    selection_hash = hashlib.sha256(
        json.dumps(selection_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return project_root / "results" / "state" / "run_all" / f"{selection_hash}.json"


def _task_state_key(task: dict[str, Any]) -> str:
    return "|".join([task["dataset"], task["model"], task["setting"], task.get("mode", "")])


class RunAllStateTracker:
    """Persistent state tracker with resume support for batch evaluation."""

    def __init__(
        self,
        *,
        project_root: Path,
        config_path: str | Path,
        args: argparse.Namespace,
        selected_tasks: Sequence[dict[str, Any]],
    ) -> None:
        self.project_root = project_root
        self.path = build_state_file_path(project_root, args, selected_tasks)
        self.payload: dict[str, Any] = {
            "config_path": str(config_path),
            "selection": {
                "datasets": list(args.datasets),
                "models": list(args.models),
                "limit": int(args.limit or 0),
                "resume": bool(args.resume),
                "dry_run": bool(args.dry_run),
            },
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "tasks": [],
            "summary": {"status": "pending", "updated_at": utc_now_iso()},
        }
        if self.path.exists():
            try:
                existing = load_json(self.path)
                if isinstance(existing, dict):
                    self.payload.update(existing)
            except Exception:
                pass
        existing_tasks = {
            _task_state_key(item): item
            for item in self.payload.get("tasks", [])
            if isinstance(item, dict)
        }
        tasks_payload: list[dict[str, Any]] = []
        for task in selected_tasks:
            output_dir = task_output_dir(project_root, task)
            existing = existing_tasks.get(_task_state_key(task), {})
            existing_complete = is_task_complete(output_dir)
            status = "completed" if existing_complete else str(existing.get("status", "pending"))
            last_event = "completed" if existing_complete else existing.get("last_event")
            message = "outputs detected on disk" if existing_complete else existing.get("message")
            completed_at = existing.get("completed_at") or (utc_now_iso() if existing_complete else None)
            tasks_payload.append({
                "dataset": task["dataset"],
                "model": task["model"],
                "setting": task["setting"],
                "mode": task.get("mode", ""),
                "label": format_task_label(task),
                "output_dir": str(output_dir),
                "status": status,
                "last_event": last_event,
                "updated_at": existing.get("updated_at"),
                "started_at": existing.get("started_at"),
                "completed_at": completed_at,
                "message": message,
            })
        self.payload["tasks"] = tasks_payload
        self.flush()

    def _find_task_entry(self, task: dict[str, Any]) -> dict[str, Any]:
        key = _task_state_key(task)
        for entry in self.payload["tasks"]:
            if _task_state_key(entry) == key:
                return entry
        raise KeyError(key)

    def set_task_status(self, task: dict[str, Any], status: str, *, message: str | None = None) -> None:
        entry = self._find_task_entry(task)
        now = utc_now_iso()
        entry["status"] = status
        entry["last_event"] = status
        entry["updated_at"] = now
        if status == "running":
            entry["started_at"] = now
        if status in {"completed", "skipped", "failed"}:
            entry["completed_at"] = now
        if message is not None:
            entry["message"] = message
        self.flush()

    def set_summary_status(self, status: str, *, message: str | None = None) -> None:
        summary = self.payload.setdefault("summary", {})
        summary["status"] = status
        summary["updated_at"] = utc_now_iso()
        if message is not None:
            summary["message"] = message
        self.flush()

    def flush(self) -> None:
        self.payload["updated_at"] = utc_now_iso()
        write_json(self.path, self.payload)


# ── Command runner ─────────────────────────────────────────────────────────

def format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_command(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    logger: tqdm.tqdm | None = None,
) -> None:
    """Run a subprocess, streaming output to logger (or tqdm.write)."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    if logger is not None:
        logger.write(f"Running: {format_command(command)}")
        if cwd is not None:
            logger.write(f"CWD: {Path(cwd)}")

    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for chunk in process.stdout:
        if logger is not None:
            logger.write(chunk)
        else:
            sys.stdout.write(chunk)
            sys.stdout.flush()
    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"Command failed with exit code {exit_code}: {format_command(command)}")


def capture_command_output(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run a subprocess, capture output, return as string."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {format_command(command)}")
    return output


# ── Asset helpers (git, pip, HF) ──────────────────────────────────────────

def ensure_git_clone(repo_url: str, target_dir: str | Path, *, logger: tqdm.tqdm | None = None) -> None:
    target_dir = Path(target_dir)
    if target_dir.exists():
        if logger:
            logger.write(f"Git repo already present: {target_dir}")
        return
    if logger:
        logger.write(f"Cloning {repo_url} -> {target_dir}")
    run_command(["git", "clone", repo_url, str(target_dir)], logger=logger)


def ensure_hf_snapshot(
    python_exe: str | Path,
    repo_id: str,
    local_dir: str | Path,
    *,
    logger: tqdm.tqdm | None = None,
) -> None:
    local_dir = Path(local_dir)
    if (local_dir / "config.json").exists():
        if logger:
            logger.write(f"HF snapshot already present: {local_dir}")
        return
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={json.dumps(repo_id)}, local_dir={json.dumps(str(local_dir))}, "
        "local_dir_use_symlinks=False)"
    )
    run_command([str(python_exe), "-c", code], logger=logger)


def ensure_python_packages(
    python_exe: str | Path,
    packages: Sequence[str],
    *,
    logger: tqdm.tqdm | None = None,
) -> None:
    for package in packages:
        module_name = package.split("[", 1)[0].replace("-", "_")
        if module_name == "pillow":
            module_name = "PIL"
        check_code = f"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)"
        completed = subprocess.run([str(python_exe), "-c", check_code], capture_output=True, text=True)
        if completed.returncode == 0:
            if logger:
                logger.write(f"Package present: {package} ({python_exe})")
            continue
        if logger:
            logger.write(f"Installing missing package: {package} ({python_exe})")
        run_command([str(python_exe), "-m", "pip", "install", package], logger=logger)


# ── TSLib helpers ──────────────────────────────────────────────────────────

def get_tslib_setting_name(
    *,
    model_id: str,
    model_name: str,
    data_name: str,
    features: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    d_model: int,
    n_heads: int,
    e_layers: int,
    d_layers: int,
    d_ff: int,
    expand: int,
    d_conv: int,
    factor: int,
    embed: str,
    distil: bool,
    description: str,
    index: int = 0,
) -> str:
    return (
        "long_term_forecast_{0}_{1}_{2}_ft{3}_sl{4}_ll{5}_pl{6}_dm{7}_nh{8}_el{9}_dl{10}_df{11}_expand{12}_dc{13}_fc{14}_eb{15}_dt{16}_{17}_{18}"
    ).format(
        model_id, model_name, data_name, features,
        seq_len, label_len, pred_len, d_model, n_heads,
        e_layers, d_layers, d_ff, expand, d_conv, factor,
        embed, distil, description, index,
    )


def build_tslib_args(
    *,
    dataset: str,
    run_key: str,
    model: str,
    setting_config: dict[str, Any],
    metadata: dict[str, Any],
    defaults: dict[str, Any],
    csv_path: Path,
    model_id_suffix: str = "",
) -> tuple[str, dict[str, Any]]:
    model_id = f"{dataset.upper()}_{model_id_suffix}_{run_key}" if model_id_suffix else f"{dataset.upper()}_{run_key}"
    model_id = model_id.replace("__", "_")
    ts_args = {
        "model_id": model_id,
        "model": model,
        "root_path": str(csv_path.parent) + os.sep,
        "data_path": csv_path.name,
        "features": str(metadata["features"]),
        "target": str(metadata["target_column"]),
        "freq": str(metadata["freq"]),
        "seq_len": int(setting_config["history"]),
        "label_len": int(setting_config["label_len"]),
        "pred_len": int(setting_config["forecast"]),
        "enc_in": int(metadata["enc_in"]),
        "dec_in": int(metadata["dec_in"]),
        "c_out": int(metadata["c_out"]),
        "e_layers": int(defaults["e_layers"]),
        "d_layers": int(defaults["d_layers"]),
        "d_model": int(defaults["d_model"]),
        "d_ff": int(defaults["d_ff"]),
        "factor": int(defaults["factor"]),
        "batch_size": int(defaults["batch_size"]),
        "train_epochs": int(defaults["train_epochs"]),
        "learning_rate": float(defaults["learning_rate"]),
    }
    if "top_k" in defaults:
        ts_args["top_k"] = int(defaults["top_k"])
    return model_id, ts_args


def get_tslib_raw_source(tslib_root: Path, *, model_id: str, model_name: str, ts_args: dict[str, Any]) -> Path:
    setting_name = get_tslib_setting_name(
        model_id=model_id,
        model_name=model_name,
        data_name="custom",
        features=ts_args["features"],
        seq_len=ts_args["seq_len"],
        label_len=ts_args["label_len"],
        pred_len=ts_args["pred_len"],
        d_model=ts_args["d_model"],
        n_heads=8,
        e_layers=ts_args["e_layers"],
        d_layers=ts_args["d_layers"],
        d_ff=ts_args["d_ff"],
        expand=2,
        d_conv=4,
        factor=ts_args["factor"],
        embed="timeF",
        distil=True,
        description="Proj2",
    )
    return tslib_root / "results" / setting_name


def run_tslib(
    python_exe: str | Path,
    tslib_root: Path,
    ts_args: dict[str, Any],
    *,
    logger: tqdm.tqdm | None = None,
) -> float:
    """Run a TSLib baseline model, return wall-clock seconds."""
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    command = [
        str(python_exe), "-u",
        str(tslib_root / "run.py"),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", str(ts_args["model_id"]),
        "--model", str(ts_args["model"]),
        "--data", "custom",
        "--root_path", str(ts_args["root_path"]),
        "--data_path", str(ts_args["data_path"]),
        "--features", str(ts_args["features"]),
        "--target", str(ts_args["target"]),
        "--freq", str(ts_args["freq"]),
        "--seq_len", str(ts_args["seq_len"]),
        "--label_len", str(ts_args["label_len"]),
        "--pred_len", str(ts_args["pred_len"]),
        "--enc_in", str(ts_args["enc_in"]),
        "--dec_in", str(ts_args["dec_in"]),
        "--c_out", str(ts_args["c_out"]),
        "--e_layers", str(ts_args["e_layers"]),
        "--d_layers", str(ts_args["d_layers"]),
        "--d_model", str(ts_args["d_model"]),
        "--d_ff", str(ts_args["d_ff"]),
        "--factor", str(ts_args["factor"]),
        "--batch_size", str(ts_args["batch_size"]),
        "--train_epochs", str(ts_args["train_epochs"]),
        "--patience", "2",
        "--learning_rate", str(ts_args["learning_rate"]),
        "--itr", "1",
        "--num_workers", "0",
        "--des", "Proj2",
        "--gpu", "0",
    ]
    if "top_k" in ts_args:
        command.extend(["--top_k", str(ts_args["top_k"])])
    started = time.perf_counter()
    run_command(command, cwd=tslib_root, env=env, logger=logger)
    return float(time.perf_counter() - started)


# ── Model runner wrappers (cross-env subprocess) ──────────────────────────

def run_chronos_experiment(
    *,
    chronos_python: str | Path,
    runner_path: Path,
    dataset: str,
    setting: str,
    metadata_path: Path | str,
    output_dir: Path | str,
    model_id: str,
    batch_size: int = 128,
    logger: tqdm.tqdm | None = None,
) -> None:
    run_command([
        str(chronos_python),
        str(runner_path),
        "--dataset", dataset,
        "--setting", setting,
        "--metadata", str(metadata_path),
        "--output-dir", str(output_dir),
        "--model-id", str(model_id),
        "--batch-size", str(batch_size),
    ], logger=logger)


def run_echo_experiment(
    *,
    chronos_python: str | Path,
    runner_path: Path,
    dataset: str,
    setting: str,
    metadata_path: Path | str,
    output_dir: Path | str,
    mode: str,
    model_path: str | None,
    batch_size: int = 64,
    echo_config: str | None = None,
    logger: tqdm.tqdm | None = None,
) -> None:
    cmd = [
        str(chronos_python),
        str(runner_path),
        "evaluate",
        "--dataset", dataset,
        "--setting", setting,
        "--metadata", str(metadata_path),
        "--output-dir", str(output_dir),
        "--mode", mode,
        "--batch-size", str(batch_size),
    ]
    if model_path:
        cmd.extend(["--model-path", str(model_path)])
    if echo_config:
        cmd.extend(["--echo-config", echo_config])
    run_command(cmd, logger=logger)


def run_aurora_experiment(
    *,
    aurora_python: str | Path,
    runner_path: Path,
    dataset: str,
    setting: str,
    metadata_path: Path | str,
    output_dir: Path | str,
    aurora_root: Path | str,
    model_path: Path | str,
    batch_size: int = 256,
    logger: tqdm.tqdm | None = None,
) -> None:
    run_command([
        str(aurora_python),
        str(runner_path),
        "--dataset", dataset,
        "--setting", setting,
        "--metadata", str(metadata_path),
        "--output-dir", str(output_dir),
        "--repo-root", str(aurora_root),
        "--model-path", str(model_path),
        "--batch-size", str(batch_size),
    ], logger=logger)


# ── Aggregate results helpers (same env, direct import) ───────────────────

def invoke_normalize_baseline(
    *,
    controller_python: str | Path,
    project_root: Path,
    dataset: str,
    model: str,
    setting: str,
    source_dir: Path | str,
    output_dir: Path | str,
    runtime_sec: float,
    series_id: str = "series",
    mode: str = "baseline_train",
) -> None:
    """Call aggregate_results.normalize_baseline via subprocess (cross-env fallback)."""
    runner = project_root / "runners" / "aggregate_results.py"
    run_command([
        str(controller_python), str(runner),
        "normalize-baseline",
        "--dataset", dataset,
        "--model", model,
        "--setting", setting,
        "--source-dir", str(source_dir),
        "--output-dir", str(output_dir),
        "--runtime-sec", str(runtime_sec),
        "--series-id", series_id,
        "--mode", mode,
    ])


def invoke_merge_run_dirs(
    *,
    controller_python: str | Path,
    project_root: Path,
    dataset: str,
    model: str,
    setting: str,
    run_root: Path | str,
    output_dir: Path | str,
    mode: str = "baseline_train",
) -> None:
    runner = project_root / "runners" / "aggregate_results.py"
    run_command([
        str(controller_python), str(runner),
        "merge-run-dirs",
        "--dataset", dataset,
        "--model", model,
        "--setting", setting,
        "--run-root", str(run_root),
        "--output-dir", str(output_dir),
        "--mode", mode,
    ])


# ── tqdm progress helpers ──────────────────────────────────────────────────

class TqdmLogger:
    """Bridge between subprocess output and tqdm progress bars."""

    def __init__(self, progress: tqdm.tqdm) -> None:
        self.progress = progress

    def write(self, msg: str) -> None:
        if msg and msg.strip():
            self.progress.write(msg.strip())


def make_progress(
    total: int,
    desc: str = "",
    unit: str = "task",
) -> tqdm.tqdm:
    return tqdm.tqdm(total=total, desc=desc, unit=unit, ncols=100, leave=True)


def short_config_desc(config: dict[str, Any]) -> str:
    """Return a brief summary of the execution plan."""
    plan = config.get("execution_plan", [])
    datasets = sorted({t["dataset"] for t in plan})
    models = sorted({t["model"] for t in plan})
    return f"{len(plan)} tasks across {len(datasets)} datasets x {len(models)} models"
