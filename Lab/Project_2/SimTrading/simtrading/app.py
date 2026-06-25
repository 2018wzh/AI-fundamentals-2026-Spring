from __future__ import annotations

from pathlib import Path

import streamlit as st

from simtrading.config import DEFAULT_CONFIG_PATH, load_app_config
from simtrading.data_loader import load_dataset
from simtrading.pages.dataset_compare import render_dataset_compare_page
from simtrading.pages.overview import render_overview_page
from simtrading.pages.portfolio import render_portfolio_page
from simtrading.pages.prediction import render_prediction_page
from simtrading.pages.trading import render_trading_page
from simtrading.strategy import StrategyConfig, StrategyName, get_strategy_availability


def _render_sidebar(config_path: Path):
    st.sidebar.title("SimTrading")
    st.sidebar.caption(f"Config: {config_path}")

    app_config = load_app_config(config_path)

    dataset_name = st.sidebar.selectbox(
        "Dataset",
        options=list(app_config.datasets.keys()),
        format_func=lambda key: app_config.datasets[key].display_name,
    )
    dataset_cfg = app_config.datasets[dataset_name]
    setting_key = st.sidebar.selectbox("Setting", options=list(dataset_cfg.sample_paths.keys()))
    model_name = st.sidebar.selectbox("Model", options=list(dataset_cfg.prediction_result_paths.keys()))
    dataset_bundle = load_dataset(app_config, dataset_name, setting_key, model_name)
    strategy_availability = get_strategy_availability(dataset_bundle.predictions)
    allowed_strategies = [name.value for name, meta in strategy_availability.items() if meta[0]]
    blocked_strategies = {name.value: meta[1] for name, meta in strategy_availability.items() if not meta[0]}
    if not allowed_strategies:
        st.error("No strategy is available for the selected prediction result file.")
        for strategy_name, reason in blocked_strategies.items():
            st.write(f"- {strategy_name}: {reason}")
        st.stop()
    strategy_name = st.sidebar.selectbox(
        "Strategy",
        options=allowed_strategies,
    )
    if blocked_strategies:
        st.sidebar.warning("Unavailable strategies:")
        for blocked_name, reason in blocked_strategies.items():
            st.sidebar.write(f"- {blocked_name}: {reason}")
    threshold = st.sidebar.number_input("Signal threshold", value=0.0, step=0.001, format="%.4f")
    fee_bps = st.sidebar.number_input("Fee (bps)", value=2.0, step=0.5, format="%.2f")
    slippage_bps = st.sidebar.number_input("Slippage (bps)", value=2.0, step=0.5, format="%.2f")
    top_k = st.sidebar.number_input("Top-K holdings", min_value=1, value=1, step=1)
    max_position = st.sidebar.number_input("Max position", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

    strategy_config = StrategyConfig(
        name=StrategyName(strategy_name),
        threshold=float(threshold),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
        top_k=int(top_k),
        max_position=float(max_position),
        score_column="y_pred",
    )
    return app_config, dataset_name, setting_key, model_name, dataset_bundle, strategy_config


def main() -> None:
    st.set_page_config(page_title="SimTrading", layout="wide")
    st.title("SimTrading")
    st.caption("Strict simulation trading dashboard for external model predictions.")

    config_path = DEFAULT_CONFIG_PATH
    if not config_path.exists():
        st.error(
            "Missing required config file `config/datasets.yaml`. Copy `config/datasets.example.yaml` "
            "to `config/datasets.yaml` and replace every placeholder with real local file paths."
        )
        st.stop()

    app_config, dataset_name, setting_key, model_name, dataset_bundle, strategy_config = _render_sidebar(config_path)

    tab_overview, tab_prediction, tab_trading, tab_portfolio, tab_compare = st.tabs(
        ["Overview", "Prediction", "Trading", "Portfolio", "Dataset Compare"]
    )

    with tab_overview:
        render_overview_page(
            app_config=app_config,
            dataset_name=dataset_name,
            setting_key=setting_key,
            model_name=model_name,
            dataset_bundle=dataset_bundle,
            strategy_config=strategy_config,
        )
    with tab_prediction:
        render_prediction_page(dataset_bundle=dataset_bundle, model_name=model_name)
    with tab_trading:
        render_trading_page(dataset_bundle=dataset_bundle, strategy_config=strategy_config)
    with tab_portfolio:
        render_portfolio_page(dataset_bundle=dataset_bundle, strategy_config=strategy_config)
    with tab_compare:
        render_dataset_compare_page(app_config=app_config, selected_dataset=dataset_name)


if __name__ == "__main__":
    main()
