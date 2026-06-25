# SimTrading

`SimTrading` is a strict Streamlit-based simulation trading dashboard for visualizing model predictions, trading signals, positions, equity curves, and drawdowns for `FNSPID` and `OilETF-TimeMMD`.

The project does not train models and does not ship with any built-in source data, sample predictions, or copied assets. You must provide local dataset paths and prediction result paths through a user-owned configuration file.

## Project layout

```text
SimTrading/
├── app.py
├── config/
│   └── datasets.example.yaml
├── requirements.txt
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
python -m pip install -r requirements.txt
```

## Configure local data paths

1. Copy `config/datasets.example.yaml` to `config/datasets.yaml`.
2. Replace every placeholder with real local file paths.
3. Ensure all referenced files exist before launching the app.

The app refuses to start when:

- `config/datasets.yaml` is missing
- any configured path does not exist
- any required market data column is missing
- any prediction result file violates the required schema

## Run

```powershell
cd D:\Workspace\AI-fundamentals-2026-Spring\Lab\Project_2\SimTrading
streamlit run app.py
```

## Prediction result schema

Every prediction result file must be a `.csv` or `.parquet` and must contain:

- `dataset`
- `model`
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

There is no automatic downgrade:

- missing quantiles block quantile strategies and quantile plots
- invalid dates stop the file from loading
- missing symbols or price alignment errors stop the backtest

## Dataset configuration contract

Each dataset entry must declare:

- market panel path
- sample metadata paths for `H60_F1` and `H120_F5`
- explicit date and price columns
- explicit execution and mark-to-market price columns
- prediction result file paths

`config/datasets.yaml` is the only supported source registry. The code does not hardcode repo-external directories.

## Backtest rules

- A prediction with `end_date=t` is first tradable on the next market date
- Rebalancing occurs at the next execution timestamp available in the market panel
- Transaction fee and slippage are both required in strategy settings
- `Top-K Rotation` requires multiple symbols on the same execution date
- `Confidence-Adjusted Long/Cash` requires `q10`, `q50`, and `q90`

## Tests

```powershell
cd D:\Workspace\AI-fundamentals-2026-Spring\Lab\Project_2\SimTrading
python -m unittest discover -s tests -v
```
