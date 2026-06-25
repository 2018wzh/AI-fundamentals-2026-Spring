from __future__ import annotations

import streamlit as st

from simtrading.backtest import run_backtest
from simtrading.data_loader import DatasetBundle
from simtrading.plots import make_drawdown_figure, make_equity_figure, make_monthly_return_heatmap
from simtrading.strategy import StrategyConfig


def render_portfolio_page(dataset_bundle: DatasetBundle, strategy_config: StrategyConfig) -> None:
    result = run_backtest(dataset_bundle, strategy_config)
    st.subheader("Portfolio")
    st.plotly_chart(make_equity_figure(result.equity_curve), use_container_width=True)
    st.plotly_chart(make_drawdown_figure(result.equity_curve), use_container_width=True)
    st.plotly_chart(make_monthly_return_heatmap(result.equity_curve), use_container_width=True)
    st.dataframe(result.equity_curve, use_container_width=True)
