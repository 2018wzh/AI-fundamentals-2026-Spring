from __future__ import annotations

import streamlit as st

from simtrading.backtest import run_backtest
from simtrading.config import AppConfig
from simtrading.data_loader import DatasetBundle
from simtrading.plots import format_metrics_table
from simtrading.strategy import StrategyConfig


def render_overview_page(
    app_config: AppConfig,
    dataset_name: str,
    setting_key: str,
    model_name: str,
    dataset_bundle: DatasetBundle,
    strategy_config: StrategyConfig,
) -> None:
    result = run_backtest(dataset_bundle, strategy_config)
    st.subheader("Overview")
    st.write(
        {
            "dataset": app_config.datasets[dataset_name].display_name,
            "setting": setting_key,
            "model": model_name,
            "strategy": strategy_config.name.value,
            "prediction_rows": len(dataset_bundle.predictions),
            "market_rows": len(dataset_bundle.market_data),
        }
    )
    st.dataframe(format_metrics_table(result.metrics), use_container_width=True)
    st.dataframe(result.trades.tail(10), use_container_width=True)
