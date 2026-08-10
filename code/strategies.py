from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    family: str
    direction: int
    lookback: int
    skip: int
    breadth: int
    rebalance: str
    cost_bps: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rebalance_mask(index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    if frequency == "daily":
        return np.ones(len(index), dtype=bool)
    periods = index.to_period("W" if frequency == "weekly" else "M")
    mask = np.ones(len(index), dtype=bool)
    mask[1:] = periods[1:] != periods[:-1]
    return mask


def _rank_weights(
    signal: np.ndarray,
    breadth: int,
    direction: int = 1,
) -> np.ndarray:
    count = len(signal)
    if not np.isfinite(signal).all() or breadth <= 0 or 2 * breadth > count:
        return np.full(count, np.nan)
    order = np.argsort(direction * signal)
    weights = np.zeros(count)
    weights[order[:breadth]] = -0.5 / breadth
    weights[order[-breadth:]] = 0.5 / breadth
    return weights


def _signal_portfolio(
    returns: pd.DataFrame,
    risk_free: pd.Series,
    signal: pd.DataFrame,
    breadth: int,
    direction: int,
    rebalance: str,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series]:
    mask = _rebalance_mask(returns.index, rebalance)
    weights = np.zeros(returns.shape[1])
    strategy_returns = np.zeros(len(returns))
    turnover = np.zeros(len(returns))
    lagged_signal = signal.shift(1).to_numpy()
    values = returns.to_numpy()
    rf = risk_free.reindex(returns.index).to_numpy(dtype=float)
    for t in range(len(returns)):
        if mask[t]:
            proposed = _rank_weights(lagged_signal[t], breadth, direction)
            if np.isfinite(proposed).all():
                turnover[t] = 0.5 * np.abs(proposed - weights).sum()
                weights = proposed
        cost = turnover[t] * cost_bps * 1e-4
        strategy_returns[t] = weights @ (values[t] - rf[t]) - cost
        total_return = rf[t] + strategy_returns[t]
        denominator = max(1.0 + total_return, 1e-8)
        weights = weights * (1.0 + values[t]) / denominator
    return (
        pd.Series(strategy_returns, index=returns.index),
        pd.Series(turnover, index=returns.index),
    )


def _time_series_portfolio(
    returns: pd.DataFrame,
    risk_free: pd.Series,
    signal: pd.DataFrame,
    direction: int,
    rebalance: str,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series]:
    mask = _rebalance_mask(returns.index, rebalance)
    weights = np.zeros(returns.shape[1])
    strategy_returns = np.zeros(len(returns))
    turnover = np.zeros(len(returns))
    lagged_signal = signal.shift(1).to_numpy()
    values = returns.to_numpy()
    rf = risk_free.reindex(returns.index).to_numpy(dtype=float)
    for t in range(len(returns)):
        if mask[t] and np.isfinite(lagged_signal[t]).all():
            signs = np.sign(direction * lagged_signal[t])
            if np.abs(signs).sum() > 0:
                proposed = signs / np.abs(signs).sum()
                turnover[t] = 0.5 * np.abs(proposed - weights).sum()
                weights = proposed
        cost = turnover[t] * cost_bps * 1e-4
        strategy_returns[t] = weights @ (values[t] - rf[t]) - cost
        total_return = rf[t] + strategy_returns[t]
        denominator = max(1.0 + total_return, 1e-8)
        weights = weights * (1.0 + values[t]) / denominator
    return (
        pd.Series(strategy_returns, index=returns.index),
        pd.Series(turnover, index=returns.index),
    )


def _broad_market(
    returns: pd.DataFrame,
    risk_free: pd.Series,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series]:
    weights = np.zeros(returns.shape[1])
    target = np.repeat(1.0 / returns.shape[1], returns.shape[1])
    mask = _rebalance_mask(returns.index, "monthly")
    strategy_returns = np.zeros(len(returns))
    turnover = np.zeros(len(returns))
    values = returns.to_numpy()
    for t in range(len(returns)):
        if mask[t]:
            turnover[t] = 0.5 * np.abs(target - weights).sum()
            weights = target.copy()
        rf = float(risk_free.iloc[t])
        cost = turnover[t] * cost_bps * 1e-4
        strategy_returns[t] = float(weights @ (values[t] - rf)) - cost
        total_return = rf + strategy_returns[t]
        denominator = max(1.0 + total_return, 1e-8)
        weights = weights * (1.0 + values[t]) / denominator
    return (
        pd.Series(strategy_returns, index=returns.index),
        pd.Series(turnover, index=returns.index),
    )


def build_candidates(
    industries: pd.DataFrame,
    risk_free: pd.Series,
    cost_bps: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[StrategySpec]]:
    industries = industries.dropna()
    risk_free = risk_free.reindex(industries.index).fillna(0.0)
    candidate_returns: dict[str, pd.Series] = {}
    candidate_turnover: dict[str, pd.Series] = {}
    specs: list[StrategySpec] = []

    market_id = "market_ew30"
    market_returns, market_turnover = _broad_market(
        industries, risk_free, cost_bps=cost_bps
    )
    candidate_returns[market_id] = market_returns
    candidate_turnover[market_id] = market_turnover
    specs.append(
        StrategySpec(
            strategy_id=market_id,
            family="broad_market",
            direction=1,
            lookback=0,
            skip=0,
            breadth=30,
            rebalance="monthly",
            cost_bps=cost_bps,
            description="Equal-weighted 30-industry excess-return sleeve.",
        )
    )

    cumulative_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for lookback in [21, 63, 126, 252]:
        for skip in [0, 21]:
            if lookback <= skip:
                continue
            raw = (1.0 + industries).rolling(lookback).apply(np.prod, raw=True) - 1.0
            cumulative_cache[(lookback, skip)] = raw.shift(skip)

    for lookback, skip in [(21, 0), (63, 0), (126, 0), (252, 21)]:
        for breadth in [3, 5, 10]:
            for direction, family in [(1, "cross_momentum"), (-1, "cross_reversal")]:
                for rebalance in ["weekly", "monthly"]:
                    strategy_id = (
                        f"{family}_lb{lookback}_skip{skip}_b{breadth}_{rebalance}"
                    )
                    realized, turnover = _signal_portfolio(
                        industries,
                        risk_free,
                        cumulative_cache[(lookback, skip)],
                        breadth=breadth,
                        direction=direction,
                        rebalance=rebalance,
                        cost_bps=cost_bps,
                    )
                    candidate_returns[strategy_id] = realized
                    candidate_turnover[strategy_id] = turnover
                    specs.append(
                        StrategySpec(
                            strategy_id=strategy_id,
                            family=family,
                            direction=direction,
                            lookback=lookback,
                            skip=skip,
                            breadth=breadth,
                            rebalance=rebalance,
                            cost_bps=cost_bps,
                            description=(
                                f"Industry {family.replace('_', ' ')}, "
                                f"{lookback}-day signal, {skip}-day skip."
                            ),
                        )
                    )

    for lookback in [21, 63, 126, 252]:
        volatility = industries.rolling(lookback).std()
        for breadth in [3, 5, 10]:
            for direction, family in [(-1, "low_volatility"), (1, "high_volatility")]:
                strategy_id = f"{family}_lb{lookback}_b{breadth}_monthly"
                realized, turnover = _signal_portfolio(
                    industries,
                    risk_free,
                    volatility,
                    breadth=breadth,
                    direction=direction,
                    rebalance="monthly",
                    cost_bps=cost_bps,
                )
                candidate_returns[strategy_id] = realized
                candidate_turnover[strategy_id] = turnover
                specs.append(
                    StrategySpec(
                        strategy_id=strategy_id,
                        family=family,
                        direction=direction,
                        lookback=lookback,
                        skip=0,
                        breadth=breadth,
                        rebalance="monthly",
                        cost_bps=cost_bps,
                        description=f"Cross-industry {family.replace('_', ' ')} spread.",
                    )
                )

    for lookback in [21, 63, 126, 252]:
        signal = cumulative_cache[(lookback, 0)]
        for direction, family in [(1, "time_series_momentum"), (-1, "time_series_reversal")]:
            for rebalance in ["weekly", "monthly"]:
                strategy_id = f"{family}_lb{lookback}_{rebalance}"
                realized, turnover = _time_series_portfolio(
                    industries,
                    risk_free,
                    signal,
                    direction=direction,
                    rebalance=rebalance,
                    cost_bps=cost_bps,
                )
                candidate_returns[strategy_id] = realized
                candidate_turnover[strategy_id] = turnover
                specs.append(
                    StrategySpec(
                        strategy_id=strategy_id,
                        family=family,
                        direction=direction,
                        lookback=lookback,
                        skip=0,
                        breadth=30,
                        rebalance=rebalance,
                        cost_bps=cost_bps,
                        description=f"Industry-level {family.replace('_', ' ')} overlay.",
                    )
                )

    returns_frame = pd.DataFrame(candidate_returns, index=industries.index)
    turnover_frame = pd.DataFrame(candidate_turnover, index=industries.index)
    return returns_frame, turnover_frame, specs


def annualized_sharpe(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2 or np.std(array, ddof=1) <= 0:
        return np.nan
    return float(np.mean(array) / np.std(array, ddof=1) * np.sqrt(252.0))


def adaptive_proposals(
    candidate_returns: pd.DataFrame,
    start: str = "2007-01-03",
    end: str = "2025-12-31",
    lookback_days: int = 1260,
    proposals_per_quarter: int = 1,
    diversity_penalty: float = 0.20,
) -> pd.DataFrame:
    index = candidate_returns.index
    eligible = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    quarters = eligible.to_period("Q")
    proposal_dates = pd.DatetimeIndex(
        [eligible[np.flatnonzero(quarters == quarter)[0]] for quarter in quarters.unique()]
    )
    proposed: list[str] = []
    records: list[dict[str, Any]] = []

    if "market_ew30" in candidate_returns:
        proposed.append("market_ew30")
        records.append(
            {
                "proposal_index": 1,
                "strategy_id": "market_ew30",
                "proposal_date": proposal_dates[0],
                "trailing_sharpe": annualized_sharpe(
                    candidate_returns.loc[: proposal_dates[0], "market_ew30"]
                    .iloc[:-1]
                    .tail(lookback_days)
                ),
                "diversity_penalty": 0.0,
                "agent_score": np.nan,
                "role": "economic_positive_control",
            }
        )

    for date in proposal_dates:
        history = candidate_returns.loc[:date].iloc[:-1].tail(lookback_days)
        if len(history) < min(504, lookback_days // 2):
            continue
        for _ in range(proposals_per_quarter):
            best: tuple[float, str, float, float] | None = None
            for strategy_id in candidate_returns.columns:
                if strategy_id in proposed:
                    continue
                values = history[strategy_id].dropna()
                sharpe = annualized_sharpe(values)
                if not np.isfinite(sharpe):
                    continue
                penalty = 0.0
                if proposed:
                    correlations = history[[strategy_id] + proposed].corr().iloc[0, 1:]
                    if len(correlations):
                        penalty = float(correlations.abs().max())
                score = sharpe - diversity_penalty * penalty
                if best is None or score > best[0]:
                    best = (score, strategy_id, sharpe, penalty)
            if best is None:
                break
            score, strategy_id, sharpe, penalty = best
            proposed.append(strategy_id)
            records.append(
                {
                    "proposal_index": len(records) + 1,
                    "strategy_id": strategy_id,
                    "proposal_date": date,
                    "trailing_sharpe": sharpe,
                    "diversity_penalty": penalty,
                    "agent_score": score,
                    "role": "adaptive_symbolic_proposal",
                }
            )
    return pd.DataFrame.from_records(records)
