from __future__ import annotations

import streamlit as st

from simtrading.backtest import run_backtest
from simtrading.data_loader import DatasetBundle
from simtrading.plots import make_signal_figure
from simtrading.strategy import StrategyConfig


def render_trading_page(dataset_bundle: DatasetBundle, strategy_config: StrategyConfig) -> None:
    result = run_backtest(dataset_bundle, strategy_config)
    st.subheader("Trading")
    st.plotly_chart(make_signal_figure(result.trades), use_container_width=True)
    st.dataframe(
        result.trades[
            [
                "symbol",
                "end_date",
                "execution_date",
                "next_execution_date",
                "score",
                "raw_weight",
                "gross_return",
                "transaction_cost",
                "net_return",
            ]
        ],
        use_container_width=True,
    )
