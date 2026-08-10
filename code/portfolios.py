from __future__ import annotations

import numpy as np
import pandas as pd

from inference import GateResult
from strategies import annualized_sharpe


def _combine_sleeves(
    candidate_returns: pd.DataFrame,
    candidate_turnover: pd.DataFrame,
    target_weights: pd.DataFrame,
    risk_free: pd.Series,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series]:
    target_weights = target_weights.reindex_like(candidate_returns).fillna(0.0)
    candidate_turnover = candidate_turnover.reindex_like(candidate_returns).fillna(
        0.0
    )
    rf = risk_free.reindex(candidate_returns.index).fillna(0.0).to_numpy()
    returns_values = candidate_returns.fillna(0.0).to_numpy()
    turnover_values = candidate_turnover.to_numpy()
    targets = target_weights.to_numpy()
    current = np.zeros(candidate_returns.shape[1])
    portfolio_returns = np.zeros(len(candidate_returns))
    total_turnover = np.zeros(len(candidate_returns))
    cost_rate = cost_bps * 1e-4

    for t in range(len(candidate_returns)):
        target = targets[t]
        meta_turnover = 0.5 * np.abs(target - current).sum()
        underlying_turnover = float(target @ turnover_values[t])
        portfolio_returns[t] = (
            float(target @ returns_values[t]) - meta_turnover * cost_rate
        )
        total_turnover[t] = underlying_turnover + meta_turnover

        total_return = rf[t] + portfolio_returns[t]
        sleeve_total_returns = rf[t] + returns_values[t]
        denominator = max(1.0 + total_return, 1e-8)
        current = target * (1.0 + sleeve_total_returns) / denominator

    return (
        pd.Series(portfolio_returns, index=candidate_returns.index),
        pd.Series(total_turnover, index=candidate_returns.index),
    )


def performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    risk_aversion: float = 3.0,
) -> dict[str, float]:
    values = returns.dropna()
    if values.empty:
        return {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "certainty_equivalent": np.nan,
            "max_drawdown": np.nan,
            "cvar_5_daily": np.nan,
            "annual_turnover": np.nan,
        }
    mean = float(values.mean())
    variance = float(values.var(ddof=1))
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    cutoff = float(values.quantile(0.05))
    cvar = float(values[values <= cutoff].mean())
    annual_turnover = (
        float(turnover.reindex(values.index).fillna(0.0).mean() * 252.0)
        if turnover is not None
        else np.nan
    )
    return {
        "annual_return": mean * 252.0,
        "annual_volatility": float(values.std(ddof=1) * np.sqrt(252.0)),
        "sharpe": annualized_sharpe(values),
        "certainty_equivalent": mean * 252.0
        - 0.5 * risk_aversion * variance * 252.0,
        "max_drawdown": float(drawdown.min()),
        "cvar_5_daily": cvar,
        "annual_turnover": annual_turnover,
    }


def safe_alpha_portfolio(
    candidate_returns: pd.DataFrame,
    candidate_turnover: pd.DataFrame,
    result: GateResult,
    risk_free: pd.Series,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    dates = candidate_returns.index
    identifiers = result.promotion_dates.index.tolist()
    promotion_dates = result.promotion_dates
    weights = pd.DataFrame(0.0, index=dates, columns=identifiers)
    for date in dates:
        active = promotion_dates[
            (promotion_dates.notna()) & (promotion_dates < date)
        ].index
        if len(active):
            weights.loc[date, active] = 1.0 / len(active)
    aligned_returns = candidate_returns.reindex(index=dates, columns=identifiers)
    aligned_turnover = candidate_turnover.reindex(index=dates, columns=identifiers)
    portfolio_returns, total_turnover = _combine_sleeves(
        aligned_returns,
        aligned_turnover,
        weights,
        risk_free,
        cost_bps,
    )
    return portfolio_returns, total_turnover, weights


def rolling_topk_portfolio(
    candidate_returns: pd.DataFrame,
    candidate_turnover: pd.DataFrame,
    start: str,
    end: str,
    risk_free: pd.Series,
    cost_bps: float,
    lookback_days: int = 1260,
    top_k: int = 5,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    returns = candidate_returns.loc[start:end]
    turnover = candidate_turnover.reindex_like(returns).fillna(0.0)
    periods = returns.index.to_period("Q")
    rebalance = np.ones(len(returns), dtype=bool)
    rebalance[1:] = periods[1:] != periods[:-1]
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    current = np.zeros(returns.shape[1])
    for t, date in enumerate(returns.index):
        if rebalance[t]:
            history = candidate_returns.loc[:date].iloc[:-1].tail(lookback_days)
            sharpes = history.mean() / history.std(ddof=1) * np.sqrt(252.0)
            eligible = sharpes.replace([np.inf, -np.inf], np.nan).dropna()
            chosen = eligible.nlargest(min(top_k, len(eligible))).index
            current = np.zeros(returns.shape[1])
            current[returns.columns.get_indexer(chosen)] = 1.0 / len(chosen)
        weights.iloc[t] = current
    portfolio_returns, total_turnover = _combine_sleeves(
        returns,
        turnover,
        weights,
        risk_free,
        cost_bps,
    )
    return portfolio_returns, total_turnover, weights


def fixed_split_portfolio(
    candidate_returns: pd.DataFrame,
    candidate_turnover: pd.DataFrame,
    risk_free: pd.Series,
    cost_bps: float,
    train_start: str = "2002-01-01",
    train_end: str = "2010-12-31",
    validation_start: str = "2011-01-01",
    validation_end: str = "2014-12-31",
    test_start: str = "2015-01-01",
    test_end: str = "2026-05-29",
    top_k: int = 5,
) -> tuple[pd.Series, pd.Series, tuple[str, ...]]:
    train = candidate_returns.loc[train_start:train_end]
    train_sharpe = train.mean() / train.std(ddof=1) * np.sqrt(252.0)
    shortlist = train_sharpe.nlargest(min(3 * top_k, len(train_sharpe))).index
    validation = candidate_returns.loc[validation_start:validation_end, shortlist]
    validation_sharpe = validation.mean() / validation.std(ddof=1) * np.sqrt(252.0)
    chosen = tuple(validation_sharpe.nlargest(top_k).index.tolist())
    sleeve_returns = candidate_returns.loc[test_start:test_end, list(chosen)]
    sleeve_turnover = candidate_turnover.loc[test_start:test_end, list(chosen)]
    weights = pd.DataFrame(
        1.0 / len(chosen), index=sleeve_returns.index, columns=sleeve_returns.columns
    )
    test_returns, test_turnover = _combine_sleeves(
        sleeve_returns,
        sleeve_turnover,
        weights,
        risk_free,
        cost_bps,
    )
    return test_returns, test_turnover, chosen


def full_sample_topk(
    candidate_returns: pd.DataFrame,
    candidate_turnover: pd.DataFrame,
    risk_free: pd.Series,
    cost_bps: float,
    start: str = "2007-01-03",
    end: str = "2026-05-29",
    top_k: int = 5,
) -> tuple[pd.Series, pd.Series, tuple[str, ...]]:
    sample = candidate_returns.loc[start:end]
    sharpes = sample.mean() / sample.std(ddof=1) * np.sqrt(252.0)
    chosen = tuple(sharpes.nlargest(top_k).index.tolist())
    sleeve_returns = sample[list(chosen)]
    sleeve_turnover = candidate_turnover.loc[start:end, list(chosen)]
    weights = pd.DataFrame(
        1.0 / len(chosen), index=sleeve_returns.index, columns=sleeve_returns.columns
    )
    returns, turnover = _combine_sleeves(
        sleeve_returns,
        sleeve_turnover,
        weights,
        risk_free,
        cost_bps,
    )
    return returns, turnover, chosen
