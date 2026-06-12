from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from simtrading.prediction_schema import QUANTILE_COLUMNS


def format_metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})


def make_prediction_figure(predictions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    ordered = predictions.sort_values("end_date")
    fig.add_trace(go.Scatter(x=ordered["end_date"], y=ordered["y_true"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=ordered["end_date"], y=ordered["y_pred"], mode="lines", name="Predicted"))
    if QUANTILE_COLUMNS.issubset(ordered.columns):
        fig.add_trace(
            go.Scatter(
                x=ordered["end_date"],
                y=ordered["q90"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ordered["end_date"],
                y=ordered["q10"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name="q10-q90",
                hoverinfo="skip",
            )
        )
        fig.add_trace(go.Scatter(x=ordered["end_date"], y=ordered["q50"], mode="lines", name="q50"))
    fig.update_layout(title="Prediction vs Actual", xaxis_title="Date", yaxis_title="Return")
    return fig


def make_signal_figure(trades: pd.DataFrame) -> go.Figure:
    ordered = trades.sort_values("execution_date")
    fig = px.bar(
        ordered,
        x="execution_date",
        y="raw_weight",
        color="symbol",
        title="Position Weights",
        barmode="group",
    )
    fig.update_layout(xaxis_title="Execution Date", yaxis_title="Weight")
    return fig


def make_equity_figure(equity_curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve["execution_date"], y=equity_curve["equity"], mode="lines", name="Equity"))
    fig.update_layout(title="Equity Curve", xaxis_title="Execution Date", yaxis_title="Equity")
    return fig


def make_drawdown_figure(equity_curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_curve["execution_date"],
            y=equity_curve["drawdown"],
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
        )
    )
    fig.update_layout(title="Drawdown", xaxis_title="Execution Date", yaxis_title="Drawdown")
    return fig


def make_monthly_return_heatmap(equity_curve: pd.DataFrame) -> go.Figure:
    monthly = equity_curve.copy()
    monthly["month"] = monthly["execution_date"].dt.to_period("M").astype(str)
    monthly_returns = monthly.groupby("month", as_index=False)["portfolio_return"].sum()
    fig = px.imshow([monthly_returns["portfolio_return"].tolist()], x=monthly_returns["month"], aspect="auto")
    fig.update_layout(title="Monthly Return Heatmap", xaxis_title="Month", yaxis_title="")
    return fig
