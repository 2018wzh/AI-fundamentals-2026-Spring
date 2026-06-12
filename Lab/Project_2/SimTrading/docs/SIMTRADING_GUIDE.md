# SimTrading Detailed Guide

## 1. Overview

`SimTrading` is a Streamlit dashboard for evaluating external prediction files on two real datasets:

- `FNSPID`
- `OilETF-TimeMMD`

The app does not train models. It loads precomputed prediction files, aligns them with the market panel and sample metadata, then runs a strict backtest.

The core design principle is simple: if the data contract is wrong, the app should fail clearly instead of guessing.

## 2. Starting The App

Recommended launch command:

```powershell
cd D:\Workspace\AI-fundamentals-2026-Spring\Lab\Project_2\SimTrading
D:\Workspace\AI-fundamentals-2026-Spring\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`.

If you want to re-check data without opening the browser:

```powershell
cd D:\Workspace\AI-fundamentals-2026-Spring\Lab\Project_2\SimTrading
D:\Workspace\AI-fundamentals-2026-Spring\.venv\Scripts\python.exe scripts\validate_simtrading_data.py
```

## 3. Configuration Model

All runtime inputs come from `config/datasets.yaml`.

Each dataset entry declares:

- `dataset_label`: the exact label stored inside prediction files
- `panel`: market panel path plus explicit date and symbol handling
- `samples`: sample file format and sample paths for `H60_F1` and `H120_F5`
- `predictions`: result paths and the signal horizon to use for each setting
- `benchmark_symbols`: labels shown in the UI

### 3.1 Panel contract

The market panel defines:

- `path`
- `date_column`
- `symbol_mode`
- `price_columns`

Two symbol modes are supported:

- `column`: use a symbol column from the panel
- `constant`: assign one canonical symbol to all rows

### 3.2 Sample contract

Two sample formats are supported:

- `metadata_table`
- `series_manifest`

`metadata_table` requires:

- `sample_id`
- `symbol`
- `end_date`
- `H`
- `F`
- `split`

`series_manifest` requires:

- `series_id`
- `csv_path`

The manifest files are used for `FNSPID`, where the sample index is built from the per-series echo CSVs.

### 3.3 Prediction contract

Prediction files must contain:

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

Quantile-based functionality also needs:

- `q10`
- `q50`
- `q90`

The `setting` column is required. It disambiguates `H60_F1` from `H120_F5`.

## 4. How The App Chooses Data

The left sidebar selects:

- dataset
- setting
- model
- strategy parameters

After a dataset and setting are selected, the model list is filtered to only models whose prediction files actually contain rows for that dataset and setting.

The app then:

1. loads the market panel
2. loads the selected sample file
3. loads the selected prediction file
4. keeps only the configured signal horizon for the selected setting
5. runs the backtest

## 5. How To Read The Screens

### 5.1 Overview

Use this as the first stop.

It shows:

- dataset name
- setting
- model
- strategy
- number of prediction rows
- number of market rows
- summary metrics
- recent trade rows

The key return metric is `total_return`.

### 5.2 Prediction

Use this page to compare the predicted curve and the actual curve.

If quantile columns exist, the chart also includes the uncertainty band.

### 5.3 Trading

Use this page to inspect trade-level details.

Important columns:

- `execution_date`
- `next_execution_date`
- `score`
- `raw_weight`
- `gross_return`
- `transaction_cost`
- `net_return`

`net_return` is the per-trade after-cost return.

### 5.4 Portfolio

Use this page to see portfolio-level behavior:

- equity curve
- drawdown
- monthly return heatmap

The last value of the equity curve corresponds to cumulative performance.

### 5.5 Dataset Compare

This page summarizes the declared dataset contract:

- dataset label
- panel path
- sample format
- sample settings
- configured models

It is a quick way to confirm that the data registry matches what you expect.

## 6. How To Compare Results

### 6.1 Compare models

Keep dataset and setting fixed.

Change only the model in the sidebar.

### 6.2 Compare settings

Keep dataset and model fixed.

Change only the setting.

### 6.3 Compare strategies

Keep dataset, setting, and model fixed.

Change only the strategy.

This is the easiest way to see how a model behaves under different trading rules.

## 7. Return Metrics

The main return metrics come from the backtest equity curve.

- `total_return`: final equity minus 1
- `annualized_return`: annualized mean-return estimate
- `annualized_volatility`: annualized standard deviation of returns
- `max_drawdown`: worst drawdown
- `sharpe`: annualized Sharpe ratio
- `win_rate`: share of positive portfolio-return periods
- `avg_gross_exposure`: average gross exposure across portfolio dates

If you only want a single number for “profitability”, use `total_return`.

## 8. Strategy Behavior

### 8.1 Long/Cash Threshold

This strategy uses the configured score column, usually `y_pred`.

If the score exceeds the threshold, the model takes a position.

### 8.2 Confidence-Adjusted Long/Cash

This strategy requires:

- `q10`
- `q50`
- `q90`

It is disabled when quantile intervals are invalid or missing.

### 8.3 Top-K Rotation

This strategy is only meaningful when a date has enough symbols to form a cross-sectional ranking.

`FNSPID` can use it.

`OilETF-TimeMMD` is single-symbol data, so Top-K is intentionally unavailable.

## 9. Validation And Tests

Validation command:

```powershell
D:\Workspace\AI-fundamentals-2026-Spring\.venv\Scripts\python.exe scripts\validate_simtrading_data.py
```

Unit tests:

```powershell
D:\Workspace\AI-fundamentals-2026-Spring\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The validator checks:

- config loading
- prediction schema
- sample alignment
- market alignment
- backtest execution
- strategy availability

## 10. Troubleshooting

### Missing config

Copy `config/datasets.example.yaml` to `config/datasets.yaml`.

### Missing paths

Ensure every declared file exists on disk.

### Prediction missing `setting`

Rebuild the prediction file or re-run the conversion stage so the `setting` column is preserved.

### Quantile strategy unavailable

Check that the prediction file contains `q10`, `q50`, and `q90`, and that the interval width is positive.

### Top-K unavailable

This is expected for `OilETF-TimeMMD`.

## 11. Data Pipeline Summary

The current data flow is:

1. external experiment results are converted into SimTrading prediction files
2. configuration points to the canonical panel and sample files
3. the app loads one dataset, one setting, and one model at a time
4. predictions are filtered by `dataset + model + setting`
5. the configured signal horizon is used for trading
6. the backtest runs and renders the views

## 12. Practical Workflow

For a clean comparison session:

1. start the app
2. select a dataset
3. select a setting
4. inspect `Overview`
5. compare models in `Trading` and `Portfolio`
6. switch to the other setting
7. repeat
8. switch to the other dataset
9. repeat

This workflow gives you a stable apples-to-apples comparison.
