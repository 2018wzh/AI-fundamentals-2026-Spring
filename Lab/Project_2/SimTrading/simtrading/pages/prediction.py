from __future__ import annotations

import streamlit as st

from simtrading.data_loader import DatasetBundle
from simtrading.prediction_schema import QUANTILE_COLUMNS
from simtrading.plots import make_prediction_figure


def render_prediction_page(dataset_bundle: DatasetBundle, model_name: str) -> None:
    st.subheader("Prediction")
    st.caption(f"Model: {model_name}")
    st.plotly_chart(make_prediction_figure(dataset_bundle.predictions), width="stretch")
    if not QUANTILE_COLUMNS.issubset(dataset_bundle.predictions.columns):
        st.warning("This prediction file does not provide q10/q50/q90, so quantile-only strategies and plots are unavailable.")
    st.dataframe(dataset_bundle.predictions.tail(20), width="stretch")
