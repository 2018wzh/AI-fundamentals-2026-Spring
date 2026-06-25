from __future__ import annotations

import streamlit as st

from simtrading.config import AppConfig


def render_dataset_compare_page(app_config: AppConfig, selected_dataset: str) -> None:
    st.subheader("Dataset Compare")
    rows = []
    for dataset_key, dataset_cfg in app_config.datasets.items():
        rows.append(
            {
                "dataset": dataset_key,
                "display_name": dataset_cfg.display_name,
                "panel_path": str(dataset_cfg.panel_path),
                "sample_settings": ", ".join(sorted(dataset_cfg.sample_paths.keys())),
                "models": ", ".join(sorted(dataset_cfg.prediction_result_paths.keys())),
                "selected": dataset_key == selected_dataset,
            }
        )
    st.dataframe(rows, use_container_width=True)
