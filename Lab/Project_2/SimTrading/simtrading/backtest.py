from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from simtrading.data_loader import DatasetBundle
from simtrading.strategy import StrategyConfig, build_signal_frame


class BacktestError(ValueError):
    """Raised when a backtest cannot be executed."""


@dataclass(frozen=True)
class BacktestResult:
    signals: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]
    diagnostics: dict[str, int]


def _resolve_execution_dates(
    signals: pd.DataFrame,
    market_data: pd.DataFrame,
    date_column: str,
    symbol_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    market_lookup = market_data[[symbol_column, date_column]].drop_duplicates().sort_values([symbol_column, date_column])
    signal_keys = signals[["symbol", "end_date"]].drop_duplicates().sort_values(["symbol", "end_date"])
    signal_keys["end_date"] = pd.to_datetime(signal_keys["end_date"]).astype("datetime64[ns]")
    pieces = []
    for symbol, symbol_signals in signal_keys.groupby("symbol", sort=False):
        calendar = market_lookup.loc[market_lookup[symbol_column] == symbol, [date_column]].rename(
            columns={date_column: "execution_date"}
        )
        calendar["execution_date"] = pd.to_datetime(calendar["execution_date"]).astype("datetime64[ns]")
        if calendar.empty:
            matched = symbol_signals.copy()
            matched["execution_date"] = pd.NaT
        else:
            matched = pd.merge_asof(
                symbol_signals.sort_values("end_date"),
                calendar.sort_values("execution_date"),
                left_on="end_date",
                right_on="execution_date",
                direction="forward",
                allow_exact_matches=False,
            )
        pieces.append(matched)

    execution_frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    dropped_keys = execution_frame["execution_date"].isna()
    dropped_no_execution = int(
        signals.merge(
            execution_frame.loc[dropped_keys, ["symbol", "end_date"]],
            on=["symbol", "end_date"],
            how="inner",
        ).shape[0]
    )
    execution_frame = execution_frame.loc[~dropped_keys].copy()
    if execution_frame.empty:
        raise BacktestError("No execution dates could be resolved.")
    merged = signals.merge(execution_frame, on=["symbol", "end_date"], how="inner", validate="many_to_one")
    return merged, {"dropped_no_execution": dropped_no_execution}


def _attach_market_prices(dataset_bundle: DatasetBundle, signals: pd.DataFrame) -> pd.DataFrame:
    cfg = dataset_bundle.dataset_config
    market = dataset_bundle.market_data.rename(columns={cfg.date_column: "market_date", cfg.symbol_column: "market_symbol"})
    execution_price_col = cfg.price_columns.execution_price
    mark_price_col = cfg.price_columns.mark_price

    merged = signals.merge(
        market[["market_symbol", "market_date", execution_price_col, mark_price_col]],
        left_on=["symbol", "execution_date"],
        right_on=["market_symbol", "market_date"],
        how="left",
        validate="many_to_one",
    )
    if merged[execution_price_col].isna().any() or merged[mark_price_col].isna().any():
        raise BacktestError("Unable to align execution dates with required market price columns.")
    merged = merged.sort_values(["symbol", "execution_date"]).reset_index(drop=True)
    execution_price_lookup = merged[["symbol", "execution_date", execution_price_col]].rename(
        columns={"execution_date": "next_execution_date", execution_price_col: "next_execution_price"}
    )
    executable_counts = merged.groupby("symbol")["execution_date"].size().sort_values(ascending=False)
    insufficient_symbols = executable_counts[executable_counts < 2]
    if not insufficient_symbols.empty:
        # Symbols with only one executable point cannot form a return interval and are dropped.
        merged = merged.loc[~merged["symbol"].isin(insufficient_symbols.index)].copy()
    merged["next_execution_date"] = merged.groupby("symbol")["execution_date"].shift(-1)
    merged = merged.loc[merged["next_execution_date"].notna()].copy()
    if merged.empty:
        raise BacktestError("Backtest requires at least two executable trading points for at least one symbol.")

    merged = merged.merge(execution_price_lookup, on=["symbol", "next_execution_date"], how="left", validate="many_to_one")
    if merged["next_execution_price"].isna().any():
        raise BacktestError("Unable to align the next execution date with the execution price column.")
    return merged


def _compute_trade_frame(dataset_bundle: DatasetBundle, signals: pd.DataFrame, strategy_config: StrategyConfig) -> pd.DataFrame:
    cfg = dataset_bundle.dataset_config
    execution_price_col = cfg.price_columns.execution_price
    trades, _ = _resolve_execution_dates(signals, dataset_bundle.market_data, cfg.date_column, cfg.symbol_column)
    trades = _attach_market_prices(dataset_bundle, trades)

    trades = trades.copy()
    trades["gross_return"] = trades["next_execution_price"] / trades[execution_price_col] - 1.0
    turnover = trades.groupby("symbol")["raw_weight"].diff().abs().fillna(trades["raw_weight"].abs())
    trades["transaction_cost"] = turnover * ((strategy_config.fee_bps + strategy_config.slippage_bps) / 10000.0)
    trades["net_return"] = trades["raw_weight"] * trades["gross_return"] - trades["transaction_cost"]
    trades["holding_days"] = (trades["next_execution_date"] - trades["execution_date"]).dt.days
    return trades


def _portfolio_curves(trades: pd.DataFrame) -> pd.DataFrame:
    portfolio = trades.groupby("execution_date", as_index=False).agg(
        portfolio_return=("net_return", "sum"),
        gross_exposure=("raw_weight", "sum"),
        trade_count=("symbol", "count"),
    )
    portfolio = portfolio.sort_values("execution_date").reset_index(drop=True)
    portfolio["equity"] = (1.0 + portfolio["portfolio_return"]).cumprod()
    portfolio["peak_equity"] = portfolio["equity"].cummax()
    portfolio["drawdown"] = portfolio["equity"] / portfolio["peak_equity"] - 1.0
    return portfolio


def _compute_metrics(equity_curve: pd.DataFrame) -> dict[str, float]:
    returns = equity_curve["portfolio_return"]
    if returns.empty:
        raise BacktestError("Equity curve is empty.")
    mean_return = float(returns.mean())
    volatility = float(returns.std(ddof=0))
    sharpe = mean_return / volatility if volatility > 0 else 0.0
    annualizer = 252.0
    return {
        "total_return": float(equity_curve["equity"].iloc[-1] - 1.0),
        "annualized_return": float((1.0 + mean_return) ** annualizer - 1.0),
        "annualized_volatility": float(volatility * (annualizer ** 0.5)),
        "max_drawdown": float(equity_curve["drawdown"].min()),
        "sharpe": float(sharpe * (annualizer ** 0.5)),
        "win_rate": float((returns > 0).mean()),
        "avg_gross_exposure": float(equity_curve["gross_exposure"].mean()),
    }


def run_backtest(dataset_bundle: DatasetBundle, strategy_config: StrategyConfig) -> BacktestResult:
    signals = build_signal_frame(dataset_bundle.predictions, strategy_config)
    trades_with_execution, diagnostics = _resolve_execution_dates(
        signals,
        dataset_bundle.market_data,
        dataset_bundle.dataset_config.date_column,
        dataset_bundle.dataset_config.symbol_column,
    )
    trades = _attach_market_prices(dataset_bundle, trades_with_execution)

    execution_price_col = dataset_bundle.dataset_config.price_columns.execution_price
    trades = trades.copy()
    trades["gross_return"] = trades["next_execution_price"] / trades[execution_price_col] - 1.0
    turnover = trades.groupby("symbol")["raw_weight"].diff().abs().fillna(trades["raw_weight"].abs())
    trades["transaction_cost"] = turnover * ((strategy_config.fee_bps + strategy_config.slippage_bps) / 10000.0)
    trades["net_return"] = trades["raw_weight"] * trades["gross_return"] - trades["transaction_cost"]
    trades["holding_days"] = (trades["next_execution_date"] - trades["execution_date"]).dt.days
    diagnostics["trade_rows"] = int(len(trades))
    equity_curve = _portfolio_curves(trades)
    metrics = _compute_metrics(equity_curve)
    return BacktestResult(signals=signals, trades=trades, equity_curve=equity_curve, metrics=metrics, diagnostics=diagnostics)
