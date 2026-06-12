# SimTrading

`SimTrading` is a strict Streamlit-based simulation trading dashboard for visualizing model predictions, trading signals, positions, equity curves, and drawdowns for `FNSPID` and `OilETF-TimeMMD`.

The runtime uses an explicit real-data contract. It does not guess whether a source uses `date` or `end_date`, whether a symbol is stored in a column or is a single constant series, or whether a prediction belongs to `H60_F1` or `H120_F5`.

## Quick Start

```powershell
cd D:\Workspace\AI-fundamentals-2026-Spring\Lab\Project_2\SimTrading
D:\Workspace\AI-fundamentals-2026-Spring\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

If you want the full user guide, read [docs/SIMTRADING_GUIDE.md](./docs/SIMTRADING_GUIDE.md).

## What This App Does

- loads real market data, sample metadata, and prediction files from `config/datasets.yaml`
- normalizes each dataset to a canonical internal contract
- builds trading signals from external model predictions
- runs a backtest and shows trade-level and portfolio-level results
- blocks invalid data instead of silently falling back

## Project Layout

```text
SimTrading/
├── app.py
├── config/
│   ├── datasets.example.yaml
│   └── datasets.yaml
├── docs/
│   └── SIMTRADING_GUIDE.md
├── predictions/
├── scripts/
├── simtrading/
│   ├── backtest.py
│   ├── config.py
│   ├── data_loader.py
│   ├── plots.py
│   ├── prediction_schema.py
│   ├── strategy.py
│   └── pages/
└── tests/
```

## Install

```powershell
cd D:\Workspace\AI-fundamentals-2026-Spring\Lab\Project_2\SimTrading
D:\Workspace\AI-fundamentals-2026-Spring\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Configuration

Copy `config/datasets.example.yaml` to `config/datasets.yaml` and replace placeholder paths with real local files.

Each dataset entry must declare:

- `dataset_label`, matching the `dataset` column inside prediction files
- `panel.path`, `panel.date_column`, `panel.symbol_mode`, and price columns
- `samples.format`, `samples.symbol_mode`, and sample paths for `H60_F1` and `H120_F5`
- `predictions.require_setting: true`
- `predictions.signal_horizons`
- `predictions.result_paths`

Supported sample formats:

- `metadata_table`: a table with `sample_id`, `symbol`, `end_date`, `H`, `F`, and `split`
- `series_manifest`: a CSV with `series_id` and `csv_path`; SimTrading builds the canonical sample index from the listed series files

## Prediction Schema

Every prediction result file must be a `.csv` or `.parquet` and must contain:

- `dataset`
- `model`
- `setting`
- `symbol`
- `end_date`
- `window`
- `horizon`
- `y_true`
- `y_pred`
- `split`

Quantile-based functionality additionally requires:

- `q10`
- `q50`
- `q90`

`setting` is mandatory. `window` and `horizon` are the model output indices (`window_id` and `horizon_idx`), not the historical lookback or forecast length from `H60_F1` / `H120_F5`.

## Pages

- `Overview`: summary metrics and recent trade rows
- `Prediction`: actual vs predicted curve and prediction table
- `Trading`: signal and position table
- `Portfolio`: equity curve, drawdown, monthly heatmap
- `Dataset Compare`: dataset and model availability matrix

## Return Metrics

- `total_return` is the cumulative return of the equity curve
- `annualized_return` is the annualized mean-return estimate
- `annualized_volatility` is the annualized standard deviation
- `max_drawdown` is the worst drawdown on the equity curve
- `sharpe` is the annualized Sharpe ratio
- `net_return` in the Trading table is the per-trade after-cost return

## Backtest Rules

- A prediction with `end_date=t` is first tradable on the next market date
- Rebalancing occurs at the next execution timestamp available in the market panel
- Transaction fee and slippage are both required in strategy settings
- `Top-K Rotation` trades only dates with at least `top_k` available symbols
- `Confidence-Adjusted Long/Cash` requires `q10`, `q50`, and `q90`

## Validation

```powershell
cd D:\Workspace\AI-fundamentals-2026-Spring\Lab\Project_2\SimTrading
D:\Workspace\AI-fundamentals-2026-Spring\.venv\Scripts\python.exe scripts\validate_simtrading_data.py
```

The validator loads every configured dataset, setting, and supported model, runs the core backtest path, and reports blocking format or backtest issues.

## Tests

```powershell
cd D:\Workspace\AI-fundamentals-2026-Spring\Lab\Project_2\SimTrading
D:\Workspace\AI-fundamentals-2026-Spring\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Common Issues

- Missing config file: copy `config/datasets.example.yaml` to `config/datasets.yaml`
- Missing paths: verify every configured file exists locally
- Strategy unavailable: check the model file and the selected setting
- Quantile strategy unavailable: check that `q10`, `q50`, `q90` exist and that intervals are valid
- Top-K unavailable: use `FNSPID`; `OilETF-TimeMMD` is single-symbol data

## Notes

- This project is intentionally strict and will fail fast on schema mismatches
- The active browser session and local Streamlit run in this workspace already point to a validated configuration
