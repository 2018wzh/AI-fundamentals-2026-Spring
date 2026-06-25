from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "report_figures"
REPORT_PATH = PROJECT_ROOT / "BENCHMARK_REPORT.md"
TIMEMMD_STANDARD = {
    "agriculture",
    "climate",
    "economy",
    "energy",
    "environment",
    "health_afr",
    "security",
    "socialgood",
    "traffic",
}
TIMEMMD_CROSS_DOMAIN = [
    "agriculture",
    "climate",
    "economy",
    "environment",
    "health_afr",
    "security",
    "socialgood",
    "traffic",
]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue": "#A3BEFA",
    "blue_dark": "#2E4780",
    "gold": "#FFE15B",
    "gold_dark": "#736422",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "olive": "#A3D576",
    "olive_dark": "#386411",
    "pink": "#F390CA",
    "pink_dark": "#8A3A6F",
    "neutral": "#C5CAD3",
    "neutral_dark": "#464C55",
}

MODEL_COLORS = {
    "Aurora": COLORS["gold"],
    "Chronos-2": COLORS["blue"],
    "Chronos-2-ECHO": COLORS["orange"],
    "DLinear": COLORS["olive"],
    "PatchTST": COLORS["pink"],
    "TimesNet": COLORS["neutral"],
}


def read_summary(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_report_text() -> str:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(REPORT_PATH)
    return REPORT_PATH.read_text(encoding="utf-8")


def parse_markdown_table_after_heading(text: str, heading: str) -> pd.DataFrame:
    start = text.find(heading)
    if start < 0:
        raise ValueError(f"Heading not found: {heading}")
    lines = text[start:].splitlines()
    rows = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            rows.append(stripped)
        elif rows:
            break
    rows = [row for row in rows if not re.fullmatch(r"\|[\s:|\-]+\|", row)]
    if len(rows) < 2:
        raise ValueError(f"No markdown table after heading: {heading}")
    header = [cell.strip() for cell in rows[0].strip("|").split("|")]
    data = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows[1:]]
    return pd.DataFrame(data, columns=header)


def safe_plot(fn, *args) -> None:
    try:
        fn(*args)
    except Exception as exc:
        print(f"WARNING: skipped {fn.__name__}: {exc}")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "setting", "mode"]
    numeric = df.select_dtypes(include="number").columns
    return df.groupby(keys, as_index=False)[list(numeric)].mean()


def setup_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": ["DejaVu Sans", "Segoe UI", "Arial", "sans-serif"],
            "font.size": 9,
            "savefig.dpi": 220,
        }
    )


def chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, 82, break_long_words=False)
    subtitle = textwrap.fill(subtitle, 112, break_long_words=False)
    fig.subplots_adjust(top=0.84)
    left = ax.get_position().x0
    fig.text(left, 0.985, title, ha="left", va="top", fontsize=13, weight="bold", color=TOKENS["ink"])
    fig.text(left, 0.93, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def label_from_row(row: pd.Series) -> str:
    return f"{row['dataset']} | {row['setting']} | {row['model']} | {row['mode']}"


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, COLORS["neutral"])


def model_short(model: str, mode: str = "") -> str:
    if model == "Chronos-2-ECHO" and mode == "text_only":
        return "ECHO(text)"
    if model == "Chronos-2-ECHO":
        return "ECHO(zs)"
    return model


def horizon_num(setting: str) -> int:
    match = re.search(r"_F(\d+)$", str(setting))
    return int(match.group(1)) if match else 0


def setting_h_num(setting: str) -> int:
    match = re.search(r"H(\d+)_", str(setting))
    return int(match.group(1)) if match else 0


def clean_number(value: str) -> float:
    text = re.sub(r"[*`✅❌⏳,MBs~]|grad_accum=\d+|\([^)]*\)", "", str(value)).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else math.nan


def plot_main_mae(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["mae"]).copy()
    plot_df["label"] = plot_df.apply(label_from_row, axis=1)
    plot_df = plot_df.sort_values("mae", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 9))
    colors = [model_color(model) for model in plot_df["model"]]
    ax.barh(plot_df["label"], plot_df["mae"], color=colors, edgecolor=COLORS["neutral_dark"], linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("MAE (log scale)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    chart_header(
        fig,
        ax,
        "Main MAE comparison across datasets and models",
        "Rows are aggregated by dataset, model, setting, and mode; FNSPID per-series baselines are averaged before plotting.",
    )
    save(fig, "fig_main_mae_by_dataset")


def plot_finance_direction(df: pd.DataFrame) -> None:
    plot_df = df[df["dataset"].isin(["fnspid", "oiletf"])].dropna(subset=["directional_accuracy", "f1_up_down"]).copy()
    plot_df["label"] = plot_df.apply(lambda r: f"{r['dataset']} | {r['setting']} | {r['model']} | {r['mode']}", axis=1)
    plot_df = plot_df.sort_values("directional_accuracy")

    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.hlines(y, plot_df["f1_up_down"], plot_df["directional_accuracy"], color=COLORS["neutral"], linewidth=1.2)
    ax.scatter(plot_df["directional_accuracy"], y, s=40, color=COLORS["orange"], edgecolor=COLORS["orange_dark"], label="DA")
    ax.scatter(plot_df["f1_up_down"], y, s=40, color=COLORS["blue"], edgecolor=COLORS["blue_dark"], label="F1")
    ax.axvline(0.5, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.set_yticks(y, plot_df["label"])
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Score")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    chart_header(
        fig,
        ax,
        "Directional metrics separate trading signal quality from point error",
        "Financial tasks report Directional Accuracy and up/down F1; the dotted line marks random-direction accuracy.",
    )
    save(fig, "fig_finance_direction_metrics")


def plot_tradeoff(df: pd.DataFrame) -> None:
    plot_df = df[df["dataset"].isin(["fnspid", "oiletf"])].dropna(subset=["mae", "directional_accuracy"]).copy()
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for model, part in plot_df.groupby("model"):
        ax.scatter(
            part["mae"],
            part["directional_accuracy"],
            s=80,
            color=model_color(model),
            edgecolor=COLORS["neutral_dark"],
            label=model,
            alpha=0.85,
        )
    ax.set_xscale("log")
    ax.set_xlabel("MAE (log scale, lower is better)")
    ax.set_ylabel("Directional Accuracy (higher is better)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    chart_header(
        fig,
        ax,
        "MAE and direction accuracy expose different winners",
        "Chronos-2 tends to minimize point error, while Chronos-2-ECHO is optimized toward directional decisions.",
    )
    save(fig, "fig_metric_tradeoff_mae_vs_da")


def plot_echo_ablation(ablation: pd.DataFrame, dataset: str, name: str) -> None:
    plot_df = ablation[(ablation["dataset"] == dataset) & (ablation["model"] == "Chronos-2-ECHO")].copy()
    plot_df = plot_df.dropna(subset=["directional_accuracy", "f1_up_down"])
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.axis("off")
        ax.text(
            0.02,
            0.52,
            "No valid text_only ablation metrics available",
            transform=ax.transAxes,
            fontsize=13,
            color=TOKENS["ink"],
            weight="bold",
        )
        ax.text(
            0.02,
            0.38,
            "The report records this as NaN/missing rather than imputing a score.",
            transform=ax.transAxes,
            fontsize=9,
            color=TOKENS["muted"],
        )
        chart_header(
            fig,
            ax,
            f"Chronos-2-ECHO ablation on {dataset}",
            "Missing or invalid ablation metrics are rendered explicitly so the figure set matches the report inventory.",
        )
        save(fig, name)
        return
    labels = [f"{row.setting} | {row.mode}" for row in plot_df.itertuples()]
    x = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(x - width / 2, plot_df["directional_accuracy"], width, label="DA", color=COLORS["orange"], edgecolor=COLORS["orange_dark"])
    ax.bar(x + width / 2, plot_df["f1_up_down"], width, label="F1", color=COLORS["blue"], edgecolor=COLORS["blue_dark"])
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    chart_header(
        fig,
        ax,
        f"Chronos-2-ECHO ablation on {dataset}",
        "Text, image, and fused modes are compared with trading-oriented direction metrics.",
    )
    save(fig, name)


def plot_runtime(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["runtime_sec"]).copy()
    plot_df["label"] = plot_df.apply(label_from_row, axis=1)
    plot_df = plot_df.sort_values("runtime_sec", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(plot_df["label"], plot_df["runtime_sec"], color=[model_color(m) for m in plot_df["model"]], edgecolor=COLORS["neutral_dark"])
    ax.set_xscale("log")
    ax.set_xlabel("Runtime seconds (log scale)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    chart_header(
        fig,
        ax,
        "Runtime comparison across experiment runs",
        "Log scale keeps fast zero-shot inference and slower full-shot training visible in the same figure.",
    )
    save(fig, "fig_runtime_comparison")


def plot_vram(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["peak_vram_mb"]).copy()
    if plot_df.empty:
        return
    plot_df["label"] = plot_df.apply(label_from_row, axis=1)
    plot_df = plot_df.sort_values("peak_vram_mb")

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.barh(plot_df["label"], plot_df["peak_vram_mb"], color=[model_color(m) for m in plot_df["model"]], edgecolor=COLORS["neutral_dark"])
    ax.set_xlabel("Peak VRAM (MB)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    chart_header(
        fig,
        ax,
        "Peak VRAM where runtime logs recorded memory",
        "Only runs with non-empty peak_vram_mb are included; missing rows are omitted instead of imputed.",
    )
    save(fig, "fig_vram_comparison")


def plot_pinball(prob: pd.DataFrame) -> None:
    cols = ["pinball_q10", "pinball_q50", "pinball_q90"]
    plot_df = prob.dropna(subset=cols, how="all").copy()
    plot_df["label"] = plot_df.apply(label_from_row, axis=1)
    plot_df = plot_df.sort_values(["dataset", "pinball_q50"])
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    width = 0.24
    for offset, col, color, edge in [
        (-width, "pinball_q10", COLORS["blue"], COLORS["blue_dark"]),
        (0, "pinball_q50", COLORS["orange"], COLORS["orange_dark"]),
        (width, "pinball_q90", COLORS["olive"], COLORS["olive_dark"]),
    ]:
        ax.bar(x + offset, plot_df[col], width, label=col.replace("pinball_", ""), color=color, edgecolor=edge)
    ax.set_xticks(x, plot_df["label"], rotation=35, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Pinball loss (log scale)")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    chart_header(
        fig,
        ax,
        "Probabilistic forecast quality by quantile",
        "Lower pinball loss is better; log scale keeps Electricity and financial tasks readable despite very different target scales.",
    )
    save(fig, "fig_probabilistic_pinball")


def plot_coverage(df: pd.DataFrame) -> None:
    matrix = pd.crosstab(df["dataset"], df["model"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(matrix.to_numpy(), cmap="YlOrBr")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iloc[i, j])
            ax.text(j, i, value, ha="center", va="center", color=TOKENS["ink"], fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Experiment rows")
    chart_header(
        fig,
        ax,
        "Dataset and model coverage in collected results",
        "Cell values count aggregated result rows after grouping duplicate per-series metric files.",
    )
    save(fig, "fig_dataset_model_coverage")


def plot_model_mode_inventory(df: pd.DataFrame) -> None:
    counts = df.assign(label=df.apply(lambda r: f"{r['model']} | {r['mode']}", axis=1))
    counts = counts.groupby("label").size().sort_values()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(counts.index, counts.values, color=COLORS["blue"], edgecolor=COLORS["blue_dark"])
    ax.set_xlabel("Result rows")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    chart_header(
        fig,
        ax,
        "Model and mode inventory in collected benchmark rows",
        "Counts include every row in summary_main.csv, including per-series baseline rows before report-level aggregation.",
    )
    save(fig, "fig_model_mode_inventory")


def plot_dataset_inventory_scale(report_text: str) -> None:
    table = parse_markdown_table_after_heading(report_text, "## 2. Dataset Inventory")
    plot_df = table.copy()
    plot_df["rows_num"] = plot_df["Rows"].map(clean_number)
    plot_df["features_num"] = plot_df["Features"].map(clean_number).fillna(0)
    plot_df = plot_df.sort_values("rows_num")

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    colors = [COLORS["orange"] if "Financial" in t else COLORS["blue"] if "Benchmark" in t else COLORS["olive"] for t in plot_df["Type"]]
    ax.barh(plot_df["Dataset"], plot_df["rows_num"], color=colors, edgecolor=COLORS["neutral_dark"])
    for i, row in enumerate(plot_df.itertuples()):
        ax.text(row.rows_num, i, f"  {row.Features} features", va="center", color=TOKENS["muted"], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Rows / tickers (log scale)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    chart_header(
        fig,
        ax,
        "Dataset inventory spans small weekly domains, finance panels, and large numeric benchmarks",
        "Bar length shows reported rows or ticker count; labels preserve the report's feature counts.",
    )
    save(fig, "fig_dataset_inventory_scale")


def timemmd_standard_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        df["dataset"].isin(TIMEMMD_STANDARD)
        & df["model"].isin(["Aurora", "Chronos-2", "Chronos-2-ECHO"])
        & (df["mode"] != "text_only")
    ].copy()
    out["h_num"] = out["setting"].map(setting_h_num)
    out = out[((out["dataset"] != "energy") & (out["h_num"] != 60) & (out["h_num"] != 120)) | ((out["dataset"] == "energy") & (out["h_num"] == 1056))]
    out["horizon"] = out["setting"].map(horizon_num)
    out["model_label"] = out.apply(lambda r: model_short(r["model"], r["mode"]), axis=1)
    return out


def plot_timemmd_standard_mae_heatmap(df: pd.DataFrame) -> None:
    std = timemmd_standard_df(df)
    best = std.loc[std.groupby(["dataset", "setting"])["mae"].idxmin()].copy()
    best = best.sort_values(["dataset", "horizon"])
    best["rank"] = best.groupby("dataset").cumcount() + 1
    best["cell"] = best.apply(lambda r: f"F{int(r['horizon'])}\n{model_short(r['model'], r['mode'])}\n{r['mae']:.3f}", axis=1)
    pivot = best.pivot(index="dataset", columns="rank", values="mae").sort_index()
    labels = best.pivot(index="dataset", columns="rank", values="cell").reindex(index=pivot.index, columns=pivot.columns)

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    image = ax.imshow(pivot.to_numpy(), cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), ["Shortest", "2nd", "3rd", "Longest"])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, labels.iloc[i, j], ha="center", va="center", fontsize=8, color=TOKENS["ink"])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Best MAE")
    chart_header(
        fig,
        ax,
        "Time-MMD Standard best model and MAE by domain and horizon",
        "Each cell shows the lowest-MAE model among Chronos-2, Aurora, and ECHO zero-shot.",
    )
    save(fig, "fig_timemmd_standard_mae_heatmap")


def plot_timemmd_standard_win_counts(df: pd.DataFrame) -> None:
    std = timemmd_standard_df(df)
    best = std.loc[std.groupby(["dataset", "setting"])["mae"].idxmin()].copy()
    counts = best.apply(lambda r: model_short(r["model"], r["mode"]), axis=1).value_counts().reindex(["ECHO(zs)", "Aurora", "Chronos-2"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(counts.index, counts.values, color=[COLORS["orange"], COLORS["gold"], COLORS["blue"]], edgecolor=COLORS["neutral_dark"])
    ax.set_ylabel("Wins")
    ax.set_ylim(0, max(36, counts.max()) * 1.12)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.6, str(int(v)), ha="center", va="bottom", color=TOKENS["ink"], weight="bold")
    chart_header(
        fig,
        ax,
        "ECHO zero-shot wins nearly every Time-MMD Standard task",
        "Wins count the best MAE across 9 domains x 4 horizons.",
    )
    save(fig, "fig_timemmd_standard_win_counts")


def plot_timemmd_horizon_average_mae(df: pd.DataFrame) -> None:
    std = timemmd_standard_df(df).copy()
    std["rank"] = std.groupby("dataset")["horizon"].rank(method="dense").astype(int)
    avg = std.groupby(["rank", "model_label"], as_index=False)["mae"].mean()
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for model, part in avg.groupby("model_label"):
        part = part.sort_values("rank")
        ax.plot(part["rank"], part["mae"], marker="o", linewidth=2, color=model_color("Chronos-2-ECHO" if model == "ECHO(zs)" else model), label=model)
    ax.set_xticks([1, 2, 3, 4], ["Shortest", "2nd", "3rd", "Longest"])
    ax.set_ylabel("Average MAE")
    ax.grid(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    chart_header(
        fig,
        ax,
        "Average Time-MMD Standard MAE rises with horizon for capable forecasters",
        "Averaged across the 9 standard domains after mapping each domain's four horizons to rank order.",
    )
    save(fig, "fig_timemmd_horizon_average_mae")


def plot_chronos_collapse_timemmd(df: pd.DataFrame) -> None:
    std = timemmd_standard_df(df).copy()
    std["rank"] = std.groupby("dataset")["horizon"].rank(method="dense").astype(int)
    avg = std.groupby(["rank", "model_label"], as_index=False)["mae"].mean()
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for model, part in avg.groupby("model_label"):
        part = part.sort_values("rank")
        ax.plot(part["rank"], part["mae"], marker="o", linewidth=2.2, color=model_color("Chronos-2-ECHO" if model == "ECHO(zs)" else model), label=model)
    chronos = avg[avg["model_label"] == "Chronos-2"].sort_values("rank")
    if not chronos.empty:
        y = chronos["mae"].mean()
        ax.axhline(y, color=COLORS["blue_dark"], linestyle=":", linewidth=1.2)
        ax.text(4.05, y, "flat error band", va="center", color=COLORS["blue_dark"], fontsize=8)
    ax.set_xticks([1, 2, 3, 4], ["Shortest", "2nd", "3rd", "Longest"])
    ax.set_ylabel("Average MAE")
    ax.grid(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    chart_header(
        fig,
        ax,
        "Chronos-2 stays nearly flat while Aurora and ECHO scale with horizon",
        "Flat MAE across forecast lengths is the collapse pattern discussed in the report.",
    )
    save(fig, "fig_chronos_collapse_timemmd")


def plot_cross_domain_h60_h120_mae(df: pd.DataFrame) -> None:
    plot_df = df[
        df["dataset"].isin(TIMEMMD_CROSS_DOMAIN)
        & df["setting"].isin(["H60_F1", "H120_F5"])
        & df["model"].isin(["Aurora", "Chronos-2", "Chronos-2-ECHO"])
    ].copy()
    plot_df["label"] = plot_df.apply(lambda r: model_short(r["model"], r["mode"]), axis=1)
    plot_df = plot_df[plot_df["label"].isin(["Aurora", "Chronos-2", "ECHO(zs)", "ECHO(text)"])]
    avg = plot_df.groupby(["setting", "label"], as_index=False)["mae"].mean()
    order = ["Chronos-2", "Aurora", "ECHO(zs)", "ECHO(text)"]
    x = np.arange(2)
    width = 0.2
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for i, model in enumerate(order):
        part = avg.set_index(["setting", "label"]).reindex([(s, model) for s in ["H60_F1", "H120_F5"]]).reset_index()
        ax.bar(x + (i - 1.5) * width, part["mae"], width, label=model, color=model_color("Chronos-2-ECHO" if model.startswith("ECHO") else model), edgecolor=COLORS["neutral_dark"])
    ax.set_xticks(x, ["H60_F1", "H120_F5"])
    ax.set_ylabel("Mean MAE across domains")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, frameon=False)
    chart_header(
        fig,
        ax,
        "Cross-domain Time-MMD short horizons favor ECHO zero-shot",
        "Means are computed across the eight non-energy Time-MMD domains with available H60/H120 rows.",
    )
    save(fig, "fig_cross_domain_h60_h120_mae")


def plot_energy_cross_domain_mae_runtime(df: pd.DataFrame) -> None:
    plot_df = df[(df["dataset"] == "energy") & df["setting"].isin(["H60_F1", "H120_F5"])].copy()
    plot_df["label"] = plot_df.apply(lambda r: f"{r['setting']} | {model_short(r['model'], r['mode'])}", axis=1)
    plot_df = plot_df.sort_values(["setting", "mae"])
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.scatter(plot_df["runtime_sec"], plot_df["mae"], s=95, c=[model_color(m) for m in plot_df["model"]], edgecolor=COLORS["neutral_dark"])
    for row in plot_df.itertuples():
        ax.text(row.runtime_sec, row.mae, f" {row.label}", va="center", fontsize=8, color=TOKENS["ink"])
    ax.set_xscale("log")
    ax.set_xlabel("Runtime seconds (log scale)")
    ax.set_ylabel("MAE")
    ax.grid(True)
    chart_header(
        fig,
        ax,
        "Energy cross-domain tradeoff: ECHO is both accurate and modest in runtime",
        "Each point is one H60/H120 energy run; lower and left is better.",
    )
    save(fig, "fig_energy_cross_domain_mae_runtime")


def plot_energy_h1056_s_mode(df: pd.DataFrame) -> None:
    plot_df = timemmd_standard_df(df)
    plot_df = plot_df[plot_df["dataset"] == "energy"].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for model, part in plot_df.groupby("model_label"):
        part = part.sort_values("horizon")
        ax.plot(part["horizon"], part["mae"], marker="o", linewidth=2, color=model_color("Chronos-2-ECHO" if model == "ECHO(zs)" else model), label=model)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("MAE")
    ax.grid(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    chart_header(
        fig,
        ax,
        "Energy H1056 S-mode confirms ECHO's long-horizon lead",
        "Pure univariate OT results from the Time-MMD Standard protocol.",
    )
    save(fig, "fig_energy_h1056_s_mode")


def plot_electricity_mae_runtime(df: pd.DataFrame) -> None:
    plot_df = df[df["dataset"] == "electricity"].copy()
    plot_df["label"] = plot_df.apply(lambda r: f"{r['setting']} | {model_short(r['model'], r['mode'])}", axis=1)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.scatter(plot_df["runtime_sec"], plot_df["mae"], s=95, c=[model_color(m) for m in plot_df["model"]], edgecolor=COLORS["neutral_dark"])
    for row in plot_df.itertuples():
        ax.text(row.runtime_sec, row.mae, f" {row.label}", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime seconds (log scale)")
    ax.set_ylabel("MAE (log scale)")
    ax.grid(True)
    chart_header(
        fig,
        ax,
        "Electricity separates trained numeric baselines from zero-shot foundation models",
        "Log axes keep 0.2-scale baselines and 100+ MAE zero-shot runs visible together.",
    )
    save(fig, "fig_electricity_mae_runtime")


def plot_fnspid_foundation_vs_baseline(df: pd.DataFrame) -> None:
    plot_df = df[df["dataset"] == "fnspid"].copy()
    plot_df["label"] = plot_df.apply(lambda r: f"{r['setting']} | {model_short(r['model'], r['mode'])}", axis=1)
    avg = plot_df.groupby(["setting", "model", "mode"], as_index=False)["mae"].mean()
    avg["label"] = avg.apply(lambda r: f"{r['setting']} | {model_short(r['model'], r['mode'])}", axis=1)
    avg = avg.sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.barh(avg["label"], avg["mae"], color=[model_color(m) for m in avg["model"]], edgecolor=COLORS["neutral_dark"])
    ax.set_xlabel("MAE")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    chart_header(
        fig,
        ax,
        "FNSPID: Aurora's multimodal setting dominates MAE",
        "Classic baselines are averaged across per-series rows before plotting.",
    )
    save(fig, "fig_fnspid_foundation_vs_baseline")


def plot_oiletf_intraday_direction(df: pd.DataFrame) -> None:
    plot_df = df[df["dataset"].isin(["oiletf", "oiletf_intraday"])].dropna(subset=["directional_accuracy", "f1_up_down"], how="all").copy()
    plot_df["label"] = plot_df.apply(lambda r: f"{r['dataset']} | {r['setting']} | {model_short(r['model'], r['mode'])}", axis=1)
    plot_df = plot_df.sort_values(["dataset", "setting", "directional_accuracy"])
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.scatter(plot_df["directional_accuracy"], y, s=48, color=COLORS["orange"], edgecolor=COLORS["orange_dark"], label="DirAcc")
    ax.scatter(plot_df["f1_up_down"], y, s=48, color=COLORS["blue"], edgecolor=COLORS["blue_dark"], label="F1")
    ax.set_yticks(y, plot_df["label"])
    ax.set_xlim(0, 1.02)
    ax.axvline(0.5, color=TOKENS["ink"], linestyle=":", linewidth=1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Score")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    chart_header(
        fig,
        ax,
        "OilETF and intraday direction metrics stay near random accuracy",
        "F1 can move independently from Directional Accuracy, especially for ECHO zero-shot.",
    )
    save(fig, "fig_oiletf_intraday_direction")


def plot_text_vs_zero_shot_delta(df: pd.DataFrame) -> None:
    echo = df[df["model"] == "Chronos-2-ECHO"].copy()
    pivot = echo.pivot_table(index=["dataset", "setting"], columns="mode", values="mae", aggfunc="mean")
    pivot = pivot.dropna(subset=["text_only", "zero_shot"]).copy()
    pivot["delta_pct"] = (pivot["text_only"] - pivot["zero_shot"]) / pivot["zero_shot"] * 100
    pivot = pivot.sort_values("delta_pct")
    labels = [f"{idx[0]} | {idx[1]}" for idx in pivot.index]
    fig, ax = plt.subplots(figsize=(10.5, max(5.8, len(pivot) * 0.36)))
    colors = [COLORS["olive"] if v <= 0 else COLORS["orange"] for v in pivot["delta_pct"]]
    ax.barh(labels, pivot["delta_pct"], color=colors, edgecolor=COLORS["neutral_dark"])
    ax.axvline(0, color=TOKENS["ink"], linewidth=1)
    ax.set_xlabel("text_only MAE delta vs zero_shot (%)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    chart_header(
        fig,
        ax,
        "ECHO text_only usually trails zero_shot, with a few small wins",
        "Negative values mean text_only has lower MAE; positive values mean zero_shot is better.",
    )
    save(fig, "fig_text_vs_zero_shot_delta")


def status_score(value: str) -> int:
    text = str(value)
    if "✅" in text:
        return 2
    if "❌" in text or "NaN" in text or "missing" in text or "no metrics" in text:
        return 0
    if "—" in text:
        return 1
    return 1


def plot_execution_status_matrix(report_text: str) -> None:
    table = parse_markdown_table_after_heading(report_text, "## 12. Execution Status Summary")
    datasets = table["Dataset"].tolist()
    cols = [c for c in table.columns if c != "Dataset"]
    matrix = np.array([[status_score(row[c]) for c in cols] for _, row in table.iterrows()])
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(datasets)), datasets)
    for i, row in enumerate(table.itertuples(index=False)):
        for j, col in enumerate(cols):
            text = str(getattr(row, col.replace("-", "_"), table.iloc[i][col]))
            mark = "OK" if "✅" in text else "MISS" if ("❌" in text or "NaN" in text) else "-"
            ax.text(j, i, mark, ha="center", va="center", fontsize=8, color=TOKENS["ink"])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["missing/NaN", "n/a", "done"])
    chart_header(
        fig,
        ax,
        "Execution status matrix highlights remaining gaps",
        "Green means completed, red means missing or invalid, neutral means not applicable.",
    )
    save(fig, "fig_execution_status_matrix")


def plot_missing_experiments_status(report_text: str) -> None:
    table = parse_markdown_table_after_heading(report_text, "**Status (2026-06-26 update):**")
    labels = table.apply(lambda r: f"{r['Dataset']} | {r['Setting']}", axis=1)
    scores = []
    for _, row in table.iterrows():
        text = " ".join(map(str, row.values))
        if "empty" in text or "never ran" in text:
            scores.append(0)
        elif "crashed" in text or "NaN" in text:
            scores.append(1)
        else:
            scores.append(2)
    colors = [COLORS["pink"], COLORS["orange"], COLORS["olive"]]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(labels, scores, color=[colors[s] for s in scores], edgecolor=COLORS["neutral_dark"])
    ax.set_xlim(0, 2.4)
    ax.set_xticks([0, 1, 2], ["rerun", "eval gap", "complete"])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    chart_header(
        fig,
        ax,
        "Training and text_only status: four complete, four still need work",
        "Status is parsed from the report's few-shot/training update table.",
    )
    save(fig, "fig_missing_experiments_status")


def plot_energy_s_mode_delta(report_text: str) -> None:
    table = parse_markdown_table_after_heading(report_text, "### 15.4 Energy H1056")
    plot_df = table.copy()
    plot_df["horizon"] = plot_df["Horizon"].map(clean_number)
    for col in ["Aurora (old)", "Aurora (S)", "ECHO(zs) (old)", "ECHO(zs) (S)"]:
        plot_df[col] = plot_df[col].map(clean_number)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(plot_df["horizon"], plot_df["Aurora (old)"], marker="o", linestyle=":", color=COLORS["gold_dark"], label="Aurora old")
    ax.plot(plot_df["horizon"], plot_df["Aurora (S)"], marker="o", color=COLORS["gold"], label="Aurora S")
    ax.plot(plot_df["horizon"], plot_df["ECHO(zs) (old)"], marker="o", linestyle=":", color=COLORS["orange_dark"], label="ECHO old")
    ax.plot(plot_df["horizon"], plot_df["ECHO(zs) (S)"], marker="o", color=COLORS["orange"], label="ECHO S")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("MAE")
    ax.grid(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, frameon=False)
    chart_header(
        fig,
        ax,
        "Energy H1056 S-mode migration leaves rankings unchanged",
        "Old two-feature and pure S-mode MAE are nearly identical for ECHO and close for Aurora.",
    )
    save(fig, "fig_energy_s_mode_delta")


def prediction_path(dataset: str) -> Path:
    candidates = [
        RESULTS_DIR / dataset / "Chronos-2-ECHO" / "H120_F5_text_only" / "predictions.parquet",
        RESULTS_DIR / dataset / "Chronos-2-ECHO" / "H60_F1_text_only" / "predictions.parquet",
        RESULTS_DIR / dataset / "Chronos-2-ECHO" / "H120_F5_text_image" / "predictions.parquet",
        RESULTS_DIR / dataset / "Chronos-2-ECHO" / "H60_F1_text_image" / "predictions.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No prediction parquet found for {dataset}")


def plot_prediction_example(dataset: str, name: str, n: int = 120) -> None:
    path = prediction_path(dataset)
    df = pd.read_parquet(path, columns=["window_id", "horizon_idx", "y_true", "y_pred", "q10", "q90"])
    df = df.sort_values(["window_id", "horizon_idx"]).head(n).reset_index(drop=True)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    if {"q10", "q90"}.issubset(df.columns) and df[["q10", "q90"]].notna().any().any():
        ax.fill_between(x, df["q10"].astype(float), df["q90"].astype(float), color=COLORS["blue"], alpha=0.22, label="q10-q90")
    ax.plot(x, df["y_true"], color=TOKENS["ink"], linewidth=1.2, label="True")
    ax.plot(x, df["y_pred"], color=COLORS["orange_dark"], linewidth=1.2, label="Pred")
    ax.axhline(0, color=COLORS["neutral_dark"], linewidth=0.8, linestyle=":")
    ax.set_xlabel("Ordered forecast rows")
    ax.set_ylabel("Target / prediction")
    ax.grid(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    chart_header(
        fig,
        ax,
        f"Prediction example for {dataset} with Chronos-2-ECHO",
        f"First {len(df)} ordered forecast rows from {path.relative_to(PROJECT_ROOT).as_posix()}; Electricity parquet is intentionally not sampled.",
    )
    save(fig, name)


def main() -> None:
    setup_theme()
    report_text = read_report_text()
    main_df = aggregate(read_summary("summary_main.csv"))
    ablation = aggregate(read_summary("summary_ablation.csv"))
    prob = aggregate(read_summary("summary_probabilistic.csv"))

    for fn, args in [
        (plot_main_mae, (main_df,)),
        (plot_finance_direction, (main_df,)),
        (plot_tradeoff, (main_df,)),
        (plot_echo_ablation, (ablation, "oiletf", "fig_echo_ablation_oiletf")),
        (plot_echo_ablation, (ablation, "fnspid", "fig_echo_ablation_fnspid")),
        (plot_runtime, (main_df,)),
        (plot_vram, (main_df,)),
        (plot_prediction_example, ("oiletf", "fig_oiletf_prediction_example")),
        (plot_prediction_example, ("fnspid", "fig_fnspid_prediction_example")),
        (plot_pinball, (prob,)),
        (plot_coverage, (main_df,)),
        (plot_model_mode_inventory, (main_df,)),
        (plot_dataset_inventory_scale, (report_text,)),
        (plot_timemmd_standard_mae_heatmap, (main_df,)),
        (plot_timemmd_standard_win_counts, (main_df,)),
        (plot_timemmd_horizon_average_mae, (main_df,)),
        (plot_chronos_collapse_timemmd, (main_df,)),
        (plot_cross_domain_h60_h120_mae, (main_df,)),
        (plot_energy_cross_domain_mae_runtime, (main_df,)),
        (plot_energy_h1056_s_mode, (main_df,)),
        (plot_electricity_mae_runtime, (main_df,)),
        (plot_fnspid_foundation_vs_baseline, (main_df,)),
        (plot_oiletf_intraday_direction, (main_df,)),
        (plot_text_vs_zero_shot_delta, (main_df,)),
        (plot_execution_status_matrix, (report_text,)),
        (plot_missing_experiments_status, (report_text,)),
        (plot_energy_s_mode_delta, (report_text,)),
    ]:
        safe_plot(fn, *args)

    expected = sorted(path.name for path in FIG_DIR.glob("fig_*.png"))
    print(f"Generated {len(expected)} PNG figures in {FIG_DIR}")
    for name in expected:
        print(f"- {name}")


if __name__ == "__main__":
    main()
