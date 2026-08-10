from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GateResult:
    e_values: pd.DataFrame
    rejection_count: pd.Series
    rejection_sets: dict[pd.Timestamp, tuple[str, ...]]
    promotion_dates: pd.Series
    lambdas: pd.Series
    gammas: pd.Series


def bounded_scores(
    returns: pd.DataFrame,
    scale_window: int = 63,
    scale_multiple: float = 4.0,
    scale_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    reference = returns if scale_reference is None else scale_reference.reindex_like(returns)
    rolling = reference.rolling(scale_window, min_periods=20).std().shift(1)
    expanding = reference.expanding(min_periods=20).std().shift(1)
    scale = rolling.fillna(expanding).clip(lower=1e-6)
    return (returns / (scale_multiple * scale)).clip(lower=-1.0, upper=1.0)


def spending_weights(count: int) -> np.ndarray:
    indices = np.arange(1, count + 1, dtype=float)
    return 1.0 / (indices * (indices + 1.0))


def weighted_ebh(
    e_values: np.ndarray,
    gamma: np.ndarray,
    level: float,
    previously_rejected: np.ndarray | None = None,
    admission_floor: float | None = None,
) -> np.ndarray:
    """Weighted step-up with an optional floor for new admissions.

    Previously rejected hypotheses retain their frozen evidence and need only
    satisfy weighted self-consistency.  A never-rejected hypothesis must also
    exceed ``admission_floor``.  SAFE-ALPHA's authoritative public and
    simulation calls set this floor to ``1 / level`` so weak-evidence
    passengers cannot enter behind a strong discovery.
    """
    e_values = np.asarray(e_values, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if len(e_values) != len(gamma):
        raise ValueError("e_values and gamma must align")
    if not (0.0 < level <= 1.0):
        raise ValueError("level must lie in (0, 1]")
    old = (
        np.zeros(len(e_values), dtype=bool)
        if previously_rejected is None
        else np.asarray(previously_rejected, dtype=bool)
    )
    if len(old) != len(e_values):
        raise ValueError("previously_rejected must align with e_values")
    floor = 0.0 if admission_floor is None else float(admission_floor)
    if floor < 0.0 or not np.isfinite(floor):
        raise ValueError("admission_floor must be finite and nonnegative")
    finite = np.isfinite(e_values) & (e_values >= 0) & (gamma > 0)
    if not finite.any():
        return np.zeros(len(e_values), dtype=bool)
    admissible = finite & (old | (e_values >= floor))
    scaled = np.zeros(len(e_values), dtype=float)
    scaled[admissible] = level * gamma[admissible] * e_values[admissible]
    ordered = np.sort(scaled)[::-1]
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    feasible = np.flatnonzero(ranks * ordered >= 1.0)
    if not len(feasible):
        if old.any():
            raise AssertionError("frozen rejection set lost self-consistency")
        return np.zeros(len(e_values), dtype=bool)
    maximum = int(feasible[-1] + 1)
    rejected = admissible & (scaled >= 1.0 / float(maximum))
    if np.any(old & ~rejected):
        raise AssertionError("weighted gate must retain frozen discoveries")
    return rejected


def _preproposal_lambda(
    scores: pd.Series,
    proposal_date: pd.Timestamp,
    lookback: int = 1260,
    lower: float = 0.01,
    upper: float = 0.50,
) -> float:
    history = scores.loc[:proposal_date].iloc[:-1].dropna().tail(lookback).to_numpy()
    if len(history) < 20:
        return lower
    second_moment = float(np.mean(history * history))
    if second_moment <= 0:
        return lower
    estimate = float(np.mean(history) / second_moment)
    return float(np.clip(estimate, lower, upper))


def run_gate(
    returns: pd.DataFrame,
    proposals: pd.DataFrame,
    level: float = 0.10,
    inspect_every: int = 5,
    scale_window: int = 63,
    scale_multiple: float = 4.0,
    unsafe_same_bar: bool = False,
    scale_reference: pd.DataFrame | None = None,
    fixed_lambdas: pd.Series | None = None,
    fixed_gammas: pd.Series | np.ndarray | None = None,
) -> GateResult:
    proposal_ids = proposals["strategy_id"].tolist()
    proposal_dates = pd.to_datetime(proposals["proposal_date"]).tolist()
    scores = bounded_scores(
        returns[proposal_ids],
        scale_window=scale_window,
        scale_multiple=scale_multiple,
        scale_reference=(
            None if scale_reference is None else scale_reference[proposal_ids]
        ),
    )
    count = len(proposal_ids)
    if fixed_gammas is None:
        gammas = spending_weights(count)
    elif isinstance(fixed_gammas, pd.Series):
        gammas = fixed_gammas.reindex(proposal_ids).to_numpy(dtype=float)
    else:
        gammas = np.asarray(fixed_gammas, dtype=float)
    if len(gammas) != count or not np.all(np.isfinite(gammas) & (gammas > 0.0)):
        raise ValueError("fixed_gammas must provide one positive weight per proposal")
    if float(gammas.sum()) > 1.0 + 1e-12:
        raise ValueError("proposal weights must sum to at most one")
    if fixed_lambdas is None:
        lambdas = np.array(
            [
                _preproposal_lambda(scores[strategy_id], proposal_date)
                for strategy_id, proposal_date in zip(proposal_ids, proposal_dates)
            ]
        )
    else:
        lambdas = fixed_lambdas.reindex(proposal_ids).to_numpy(dtype=float)
        if not np.isfinite(lambdas).all():
            raise ValueError("fixed_lambdas does not cover every proposed strategy")
    e_values = np.ones(count, dtype=float)
    promoted = np.zeros(count, dtype=bool)
    promotion_dates = np.full(count, np.datetime64("NaT"), dtype="datetime64[ns]")
    paths = np.ones((len(scores), count), dtype=float)
    proposal_locations = np.array(
        [scores.index.searchsorted(date, side="right") for date in proposal_dates]
    )
    rejection_count = np.zeros(len(scores), dtype=int)
    rejection_sets: dict[pd.Timestamp, tuple[str, ...]] = {}

    score_values = scores.to_numpy()
    for t in range(len(scores)):
        active = proposal_locations <= t
        observed = np.nan_to_num(score_values[t], nan=0.0)
        if unsafe_same_bar:
            bets = lambdas * (observed >= 0.0)
        else:
            bets = lambdas
        factors = 1.0 + bets * observed
        updating = active & ~promoted
        e_values[updating] *= factors[updating]
        if t % inspect_every == 0 or t == len(scores) - 1:
            eligible = active | promoted
            gate_input = np.zeros_like(e_values)
            gate_input[eligible] = e_values[eligible]
            gate_set = weighted_ebh(
                gate_input,
                gammas,
                level,
                previously_rejected=promoted,
                admission_floor=1.0 / level,
            ) & eligible
            new_promotions = gate_set & ~promoted
            if new_promotions.any():
                promoted |= new_promotions
                promotion_dates[new_promotions] = scores.index[t].to_datetime64()
            rejection_count[t] = int(promoted.sum())
            if promoted.any():
                rejection_sets[scores.index[t]] = tuple(
                    np.asarray(proposal_ids)[promoted].tolist()
                )
        elif t > 0:
            rejection_count[t] = rejection_count[t - 1]
        paths[t] = e_values

    return GateResult(
        e_values=pd.DataFrame(paths, index=scores.index, columns=proposal_ids),
        rejection_count=pd.Series(rejection_count, index=scores.index),
        rejection_sets=rejection_sets,
        promotion_dates=pd.Series(
            pd.to_datetime(promotion_dates), index=proposal_ids, dtype="datetime64[ns]"
        ),
        lambdas=pd.Series(lambdas, index=proposal_ids),
        gammas=pd.Series(gammas, index=proposal_ids),
    )


def terminal_rejections(result: GateResult, level: float = 0.10) -> tuple[str, ...]:
    del level
    promoted = result.promotion_dates.notna().to_numpy()
    return tuple(result.e_values.columns[promoted].tolist())


def first_rejection_date(result: GateResult) -> pd.Timestamp | None:
    positive = result.rejection_count[result.rejection_count > 0]
    return None if positive.empty else pd.Timestamp(positive.index[0])
