from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from inference import spending_weights, weighted_ebh


@dataclass
class StreamOutcome:
    safe_count: int
    leakage_count: int
    sequential_count: int
    holdout_count: int
    safe_first_day: int
    leakage_first_day: int
    sequential_first_day: int
    holdout_first_day: int


@dataclass
class EvidenceComparison:
    """Paired SAFE and terminal decisions from one common raw e-process.

    ``raw_terminal`` is updated after proposal through the common terminal
    date without regard to either decision rule.  SAFE reads that same path
    at the predeclared inspections, but its deployed copy is frozen after a
    promotion.  The terminal rule is applied once to ``raw_terminal`` with
    exactly the same proposal weights.
    """

    raw_terminal: np.ndarray
    safe_terminal: np.ndarray
    safe_promoted: np.ndarray
    safe_promotion_day: np.ndarray
    terminal_rejected: np.ndarray


def _run_common_evidence(
    values: np.ndarray,
    arrivals: np.ndarray,
    lambdas: np.ndarray,
    gammas: np.ndarray,
    levels: np.ndarray,
    inspect_every: int = 5,
) -> list[EvidenceComparison]:
    """Run multiple levels on a single simulated score/evidence path.

    All levels and both decision timings receive the same score matrix,
    proposal order, proposal-time bets, proposal weights, and multiplicative
    factors.  A separate SAFE copy is necessary because evidence is frozen
    only after that level's irreversible promotion.  The matched terminal
    weighted e-BH rule instead reads the untouched raw endpoints once.
    """
    values = np.asarray(values, dtype=float)
    arrivals = np.asarray(arrivals, dtype=int)
    lambdas = np.asarray(lambdas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    proposal_count = len(arrivals)
    if values.ndim != 2 or values.shape[1] != proposal_count:
        raise ValueError("values and arrivals must describe the same proposals")
    if not (
        len(lambdas) == proposal_count
        and len(gammas) == proposal_count
        and np.all(gammas > 0.0)
    ):
        raise ValueError("lambdas and positive gammas must align with proposals")
    if np.any((levels <= 0.0) | (levels > 1.0)):
        raise ValueError("levels must lie in (0, 1]")

    raw_e = np.ones(proposal_count, dtype=float)
    safe_e = np.ones((len(levels), proposal_count), dtype=float)
    promoted = np.zeros((len(levels), proposal_count), dtype=bool)
    promotion_day = np.full((len(levels), proposal_count), -1, dtype=int)
    first_day = max(int(arrivals.min()), 0) if proposal_count else 0
    for t in range(first_day, len(values)):
        active = arrivals < t
        factors = 1.0 + lambdas * values[t]
        raw_e[active] *= factors[active]
        for level_index in range(len(levels)):
            updating = active & ~promoted[level_index]
            safe_e[level_index, updating] *= factors[updating]
        if t % inspect_every == 0 or t == len(values) - 1:
            for level_index, level in enumerate(levels):
                eligible = active | promoted[level_index]
                gate_input = np.zeros(proposal_count, dtype=float)
                gate_input[eligible] = safe_e[level_index, eligible]
                gate = weighted_ebh(
                    gate_input,
                    gammas,
                    float(level),
                    previously_rejected=promoted[level_index],
                    admission_floor=1.0 / float(level),
                ) & eligible
                new = gate & ~promoted[level_index]
                promotion_day[level_index, new] = t
                promoted[level_index] |= new

    return [
        EvidenceComparison(
            raw_terminal=raw_e.copy(),
            safe_terminal=safe_e[level_index].copy(),
            safe_promoted=promoted[level_index].copy(),
            safe_promotion_day=promotion_day[level_index].copy(),
            terminal_rejected=weighted_ebh(
                raw_e,
                gammas,
                float(level),
                admission_floor=1.0 / float(level),
            ),
        )
        for level_index, level in enumerate(levels)
    ]


def _adaptive_order(
    scores: np.ndarray,
    proposal_times: np.ndarray,
    maximum: int,
    forced_first: int = 0,
    lookback: int = 1260,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = [forced_first]
    arrivals: list[int] = [int(proposal_times[0])]
    for proposal_time in proposal_times[1:]:
        if len(selected) >= maximum:
            break
        history = scores[max(0, proposal_time - lookback) : proposal_time]
        means = np.nanmean(history, axis=0)
        standard_deviations = np.nanstd(history, axis=0, ddof=1)
        metric = means / np.maximum(standard_deviations, 1e-8) * np.sqrt(252.0)
        metric[np.asarray(selected, dtype=int)] = -np.inf
        choice = int(np.nanargmax(metric))
        selected.append(choice)
        arrivals.append(int(proposal_time))
    return np.asarray(selected, dtype=int), np.asarray(arrivals, dtype=int)


def _initial_lambdas(
    scores: np.ndarray,
    selected: np.ndarray,
    arrivals: np.ndarray,
    lookback: int = 1260,
    lower: float = 0.01,
    upper: float = 0.50,
) -> np.ndarray:
    values = np.zeros(len(selected))
    for j, (candidate, arrival) in enumerate(zip(selected, arrivals)):
        history = scores[max(0, arrival - lookback) : arrival, candidate]
        history = history[np.isfinite(history)]
        second = float(np.mean(history * history)) if len(history) else 0.0
        estimate = float(np.mean(history) / second) if second > 0 else lower
        values[j] = np.clip(estimate, lower, upper)
    return values


def _run_null_stream(
    base_scores: np.ndarray,
    start_location: int,
    proposal_gap: int,
    maximum_proposals: int,
    level: float,
    inspect_every: int,
    seed: int,
) -> StreamOutcome:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(base_scores))
    scores = np.nan_to_num(base_scores, nan=0.0) * signs[:, None]
    proposal_times = np.arange(
        start_location, len(scores) - 63, proposal_gap, dtype=int
    )
    selected, arrivals = _adaptive_order(
        scores, proposal_times, maximum=maximum_proposals
    )
    values = scores[:, selected]
    lambdas = _initial_lambdas(scores, selected, arrivals)
    gammas = spending_weights(len(selected))

    safe_e = np.ones(len(selected))
    leakage_e = np.ones(len(selected))
    safe_promoted = np.zeros(len(selected), dtype=bool)
    leakage_promoted = np.zeros(len(selected), dtype=bool)
    sequential_promoted = np.zeros(len(selected), dtype=bool)
    holdout_promoted = np.zeros(len(selected), dtype=bool)
    holdout_evaluated = np.zeros(len(selected), dtype=bool)
    sums = np.zeros(len(selected))
    sums_of_squares = np.zeros(len(selected))
    observations = np.zeros(len(selected), dtype=int)
    first = {"safe": -1, "leakage": -1, "sequential": -1, "holdout": -1}
    for t in range(start_location, len(scores)):
        active = arrivals < t
        observation = values[t]
        update_safe = active & ~safe_promoted
        update_leakage = active & ~leakage_promoted
        safe_e[update_safe] *= 1.0 + lambdas[update_safe] * observation[update_safe]
        leakage_bets = lambdas * (observation >= 0.0)
        leakage_e[update_leakage] *= (
            1.0 + leakage_bets[update_leakage] * observation[update_leakage]
        )
        sequential_active = active & ~sequential_promoted
        sums[sequential_active] += observation[sequential_active]
        sums_of_squares[sequential_active] += observation[sequential_active] ** 2
        observations[sequential_active] += 1

        if t % inspect_every == 0 or t == len(scores) - 1:
            safe_input = np.zeros_like(safe_e)
            leakage_input = np.zeros_like(leakage_e)
            eligible = active | safe_promoted | leakage_promoted
            safe_input[eligible] = safe_e[eligible]
            leakage_input[eligible] = leakage_e[eligible]
            safe_gate = weighted_ebh(
                safe_input,
                gammas,
                level,
                previously_rejected=safe_promoted,
                admission_floor=1.0 / level,
            ) & eligible
            leakage_gate = weighted_ebh(
                leakage_input,
                gammas,
                level,
                previously_rejected=leakage_promoted,
                admission_floor=1.0 / level,
            ) & eligible
            safe_promoted |= safe_gate
            leakage_promoted |= leakage_gate
            enough = observations >= 63
            variance = np.maximum(
                (
                    sums_of_squares
                    - np.divide(
                        sums * sums,
                        np.maximum(observations, 1),
                        out=np.zeros_like(sums),
                        where=observations > 0,
                    )
                )
                / np.maximum(observations - 1, 1),
                1e-10,
            )
            statistic = np.divide(
                sums,
                np.sqrt(variance * np.maximum(observations, 1)),
                out=np.zeros_like(sums),
                where=observations > 0,
            )
            critical_values = student_t.ppf(
                0.95, np.maximum(observations - 1, 1)
            )
            sequential_promoted |= enough & (statistic > critical_values)

            holdout_due = active & ~holdout_evaluated & (t >= arrivals + 252)
            for j in np.flatnonzero(holdout_due):
                sample = values[arrivals[j] + 1 : arrivals[j] + 253, j]
                standard_error = np.std(sample, ddof=1) / np.sqrt(len(sample))
                if (
                    standard_error > 0
                    and np.mean(sample) / standard_error
                    > student_t.ppf(0.95, len(sample) - 1)
                ):
                    holdout_promoted[j] = True
                holdout_evaluated[j] = True

            for key, promoted in [
                ("safe", safe_promoted),
                ("leakage", leakage_promoted),
                ("sequential", sequential_promoted),
                ("holdout", holdout_promoted),
            ]:
                if first[key] < 0 and promoted.any():
                    first[key] = t - start_location

    return StreamOutcome(
        safe_count=int(safe_promoted.sum()),
        leakage_count=int(leakage_promoted.sum()),
        sequential_count=int(sequential_promoted.sum()),
        holdout_count=int(holdout_promoted.sum()),
        safe_first_day=first["safe"],
        leakage_first_day=first["leakage"],
        sequential_first_day=first["sequential"],
        holdout_first_day=first["holdout"],
    )


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def null_calibration(
    scores: pd.DataFrame,
    start_date: str = "2007-01-03",
    repetitions: int = 500,
    proposal_gap: int = 63,
    maximum_proposals: int = 77,
    level: float = 0.10,
    inspect_every: int = 5,
    seed: int = 271828,
) -> pd.DataFrame:
    start_location = int(scores.index.searchsorted(pd.Timestamp(start_date)))
    outcomes = [
        _run_null_stream(
            scores.to_numpy(),
            start_location=start_location,
            proposal_gap=proposal_gap,
            maximum_proposals=maximum_proposals,
            level=level,
            inspect_every=inspect_every,
            seed=seed + repetition,
        )
        for repetition in range(repetitions)
    ]
    rows = []
    for method, count_field, day_field in [
        ("SAFE-ALPHA", "safe_count", "safe_first_day"),
        ("Same-bar leakage", "leakage_count", "leakage_first_day"),
        ("Repeated 5% t-test", "sequential_count", "sequential_first_day"),
        (
            "Unadjusted one-year holdout",
            "holdout_count",
            "holdout_first_day",
        ),
    ]:
        counts = np.asarray([getattr(outcome, count_field) for outcome in outcomes])
        first_days = np.asarray([getattr(outcome, day_field) for outcome in outcomes])
        successes = int(np.sum(counts > 0))
        lower, upper = _wilson(successes, repetitions)
        rows.append(
            {
                "method": method,
                "false_discovery_probability": successes / repetitions,
                "ci_low": lower,
                "ci_high": upper,
                "mean_false_discoveries": float(np.mean(counts)),
                "median_first_day_if_any": (
                    float(np.median(first_days[first_days >= 0]))
                    if np.any(first_days >= 0)
                    else np.nan
                ),
                "repetitions": repetitions,
            }
        )
    return pd.DataFrame(rows)


def _ordinary_bh(p_values: np.ndarray, level: float) -> np.ndarray:
    order = np.argsort(p_values)
    sorted_values = p_values[order]
    eligible = np.flatnonzero(
        sorted_values <= level * np.arange(1, len(p_values) + 1) / len(p_values)
    )
    rejected = np.zeros(len(p_values), dtype=bool)
    if len(eligible):
        rejected[order[: eligible[-1] + 1]] = True
    return rejected


def _planted_repetition(
    annual_sharpe: float,
    correlation: float,
    seed: int,
    candidates: int = 80,
    alternatives: int = 16,
    pre_days: int = 504,
    post_days: int = 1260,
    proposal_gap: int = 15,
    maximum_proposals: int = 50,
    level: float = 0.10,
    inspect_every: int = 5,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    truth = np.zeros(candidates, dtype=bool)
    truth[rng.choice(candidates, alternatives, replace=False)] = True
    total_days = pre_days + post_days
    common = rng.normal(size=(total_days, 1))
    idiosyncratic = rng.normal(size=(total_days, candidates))
    sigma = 0.25
    drift = annual_sharpe * sigma / np.sqrt(252.0)
    scores = sigma * (
        np.sqrt(correlation) * common
        + np.sqrt(max(1.0 - correlation, 0.0)) * idiosyncratic
    )
    scores += truth[None, :] * drift
    scores = np.clip(scores, -1.0, 1.0)

    proposal_times = np.arange(
        pre_days, total_days - 63, proposal_gap, dtype=int
    )
    selected, arrivals = _adaptive_order(
        scores,
        proposal_times,
        maximum=maximum_proposals,
        forced_first=int(np.argmax(scores[:pre_days].mean(axis=0))),
        lookback=pre_days,
    )
    values = scores[:, selected]
    selected_truth = truth[selected]
    lambdas = _initial_lambdas(
        scores,
        selected,
        arrivals,
        lookback=pre_days,
    )
    gammas = spending_weights(len(selected))
    evidence = _run_common_evidence(
        values,
        arrivals,
        lambdas,
        gammas,
        levels=np.array([level]),
        inspect_every=inspect_every,
    )[0]
    promoted = evidence.safe_promoted
    promotion_day = evidence.safe_promotion_day
    terminal_ebh = evidence.terminal_rejected

    sample_means = np.zeros(len(selected))
    sample_standard_errors = np.full(len(selected), np.inf)
    for j, arrival in enumerate(arrivals):
        sample = values[arrival + 1 :, j]
        sample_means[j] = np.mean(sample)
        sample_standard_errors[j] = np.std(sample, ddof=1) / np.sqrt(len(sample))
    z_statistics = sample_means / np.maximum(sample_standard_errors, 1e-12)
    degrees_of_freedom = np.asarray(
        [len(values[arrival + 1 :, j]) - 1 for j, arrival in enumerate(arrivals)]
    )
    p_values = student_t.sf(
        z_statistics, np.maximum(degrees_of_freedom, 1)
    )
    bh = _ordinary_bh(p_values, level)
    uncorrected = p_values <= 0.05
    top_five = np.zeros(len(selected), dtype=bool)
    top_five[np.argsort(sample_means)[-5:]] = True

    def summarize(discoveries: np.ndarray, prefix: str) -> dict[str, float]:
        count = int(discoveries.sum())
        false = int(np.sum(discoveries & ~selected_truth))
        true = int(np.sum(discoveries & selected_truth))
        available = max(int(selected_truth.sum()), 1)
        return {
            f"{prefix}_fdp": false / max(count, 1),
            f"{prefix}_power": true / available,
            f"{prefix}_end_to_end_power": true / max(int(truth.sum()), 1),
            f"{prefix}_discoveries": count,
        }

    output: dict[str, float] = {}
    output["proposal_recall"] = float(selected_truth.sum() / max(truth.sum(), 1))
    output.update(summarize(promoted, "safe"))
    output.update(summarize(terminal_ebh, "terminal_ebh"))
    output.update(summarize(bh, "terminal_bh"))
    output.update(summarize(uncorrected, "uncorrected"))
    output.update(summarize(top_five, "top5"))
    alternative_delays = promotion_day[promoted & selected_truth] - arrivals[
        promoted & selected_truth
    ]
    output["safe_delay"] = (
        float(np.mean(alternative_delays)) if len(alternative_delays) else np.nan
    )
    terminal_alternative_delays = (
        total_days - 1 - arrivals[terminal_ebh & selected_truth]
    )
    output["terminal_ebh_delay"] = (
        float(np.mean(terminal_alternative_delays))
        if len(terminal_alternative_delays)
        else np.nan
    )
    return output


def planted_power(
    annual_sharpes: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
    correlations: tuple[float, ...] = (0.0, 0.5),
    repetitions: int = 250,
    seed: int = 314159,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    counter = 0
    for correlation in correlations:
        for annual_sharpe in annual_sharpes:
            outcomes = []
            for _ in range(repetitions):
                outcomes.append(
                    _planted_repetition(
                        annual_sharpe=annual_sharpe,
                        correlation=correlation,
                        seed=seed + counter,
                    )
                )
                counter += 1
            frame = pd.DataFrame(outcomes)
            for method, prefix in [
                ("SAFE-ALPHA", "safe"),
                ("Matched terminal weighted e-BH", "terminal_ebh"),
                ("Terminal BH", "terminal_bh"),
                ("Uncorrected 5%", "uncorrected"),
                ("Top-5 backtest", "top5"),
            ]:
                rows.append(
                    {
                        "method": method,
                        "annual_sharpe": annual_sharpe,
                        "correlation": correlation,
                        "mean_fdp": float(frame[f"{prefix}_fdp"].mean()),
                        "mean_power": float(frame[f"{prefix}_power"].mean()),
                        "mean_end_to_end_power": float(
                            frame[f"{prefix}_end_to_end_power"].mean()
                        ),
                        "mean_proposal_recall": float(
                            frame["proposal_recall"].mean()
                        ),
                        "mean_discoveries": float(
                            frame[f"{prefix}_discoveries"].mean()
                        ),
                        "mean_delay_days": (
                            float(frame[f"{prefix}_delay"].mean())
                            if prefix in {"safe", "terminal_ebh"}
                            else np.nan
                        ),
                        "repetitions": repetitions,
                    }
                )
    return pd.DataFrame(rows)


DENSE_ANNUAL_SHARPES = (
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
    2.75,
    3.00,
)
ARCHIVED_ANNUAL_SHARPES = (0.50, 1.00, 1.50, 2.00, 3.00)
INSERTED_ANNUAL_SHARPES = (0.75, 1.25, 1.75, 2.25, 2.50, 2.75)
POWER_CORRELATIONS = (0.0, 0.5)
Q_SENSITIVITY_LEVELS = (0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)


def matched_seed_blocks(repetitions: int = 250) -> pd.DataFrame:
    """Stable hybrid seed map for the dense paired power grid.

    Published anchor cells retain their exact sequential 314159 stream.
    Newly inserted cells occupy deterministic, nonoverlapping 20260730-based
    blocks.  Inserting the dense cells therefore cannot perturb an anchor.
    """
    rows: list[dict[str, object]] = []
    for correlation_index, correlation in enumerate(POWER_CORRELATIONS):
        for annual_sharpe in DENSE_ANNUAL_SHARPES:
            if annual_sharpe in ARCHIVED_ANNUAL_SHARPES:
                effect_index = ARCHIVED_ANNUAL_SHARPES.index(annual_sharpe)
                block_index = (
                    correlation_index * len(ARCHIVED_ANNUAL_SHARPES)
                    + effect_index
                )
                seed_first = 314159 + repetitions * block_index
                source = "archived_314159_anchor"
            else:
                effect_index = INSERTED_ANNUAL_SHARPES.index(annual_sharpe)
                block_index = (
                    correlation_index * len(INSERTED_ANNUAL_SHARPES)
                    + effect_index
                )
                seed_first = 20260730 + repetitions * block_index
                source = "inserted_20260730_block"
            rows.append(
                {
                    "annual_sharpe": annual_sharpe,
                    "correlation": correlation,
                    "seed_source": source,
                    "seed_first": seed_first,
                    "seed_last": seed_first + repetitions - 1,
                    "repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def _planted_level_repetition(
    annual_sharpe: float,
    correlation: float,
    seed: int,
    levels: tuple[float, ...] = Q_SENSITIVITY_LEVELS,
    candidates: int = 80,
    alternatives: int = 16,
    pre_days: int = 504,
    post_days: int = 1260,
    proposal_gap: int = 15,
    maximum_proposals: int = 50,
    inspect_every: int = 5,
) -> list[dict[str, float]]:
    """Evaluate several q levels without changing a planted random path."""
    rng = np.random.default_rng(seed)
    truth = np.zeros(candidates, dtype=bool)
    truth[rng.choice(candidates, alternatives, replace=False)] = True
    total_days = pre_days + post_days
    common = rng.normal(size=(total_days, 1))
    idiosyncratic = rng.normal(size=(total_days, candidates))
    sigma = 0.25
    drift = annual_sharpe * sigma / np.sqrt(252.0)
    scores = sigma * (
        np.sqrt(correlation) * common
        + np.sqrt(max(1.0 - correlation, 0.0)) * idiosyncratic
    )
    scores += truth[None, :] * drift
    scores = np.clip(scores, -1.0, 1.0)
    proposal_times = np.arange(
        pre_days, total_days - 63, proposal_gap, dtype=int
    )
    selected, arrivals = _adaptive_order(
        scores,
        proposal_times,
        maximum=maximum_proposals,
        forced_first=int(np.argmax(scores[:pre_days].mean(axis=0))),
        lookback=pre_days,
    )
    selected_truth = truth[selected]
    lambdas = _initial_lambdas(scores, selected, arrivals, lookback=pre_days)
    gammas = spending_weights(len(selected))
    comparisons = _run_common_evidence(
        scores[:, selected],
        arrivals,
        lambdas,
        gammas,
        levels=np.asarray(levels, dtype=float),
        inspect_every=inspect_every,
    )
    rows: list[dict[str, float]] = []
    for level, comparison in zip(levels, comparisons):
        for method, rejected, decision_day in [
            (
                "SAFE-ALPHA",
                comparison.safe_promoted,
                comparison.safe_promotion_day,
            ),
            (
                "Matched terminal weighted e-BH",
                comparison.terminal_rejected,
                np.full(len(selected), total_days - 1, dtype=int),
            ),
        ]:
            discoveries = int(rejected.sum())
            true_discoveries = int(np.sum(rejected & selected_truth))
            false_discoveries = int(np.sum(rejected & ~selected_truth))
            true_delays = decision_day[rejected & selected_truth] - arrivals[
                rejected & selected_truth
            ]
            rows.append(
                {
                    "q": float(level),
                    "method": method,
                    "fdp": false_discoveries / max(discoveries, 1),
                    "end_to_end_power": true_discoveries / max(alternatives, 1),
                    "discoveries": discoveries,
                    "delay_days": (
                        float(np.mean(true_delays)) if len(true_delays) else np.nan
                    ),
                }
            )
    return rows


def _null_level_repetition(
    base_scores: np.ndarray,
    start_location: int,
    proposal_gap: int,
    maximum_proposals: int,
    levels: tuple[float, ...],
    inspect_every: int,
    seed: int,
) -> list[dict[str, float]]:
    """Reuse one archived wild-sign null path at every requested q level."""
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(base_scores))
    scores = np.nan_to_num(base_scores, nan=0.0) * signs[:, None]
    proposal_times = np.arange(
        start_location, len(scores) - 63, proposal_gap, dtype=int
    )
    selected, arrivals = _adaptive_order(
        scores, proposal_times, maximum=maximum_proposals
    )
    lambdas = _initial_lambdas(scores, selected, arrivals)
    comparisons = _run_common_evidence(
        scores[:, selected],
        arrivals,
        lambdas,
        spending_weights(len(selected)),
        levels=np.asarray(levels, dtype=float),
        inspect_every=inspect_every,
    )
    return [
        {
            "q": float(level),
            "false_discoveries": int(comparison.safe_promoted.sum()),
            "any_false": bool(comparison.safe_promoted.any()),
        }
        for level, comparison in zip(levels, comparisons)
    ]


def _heavy_tailed_null_repetition(
    seed: int,
    candidates: int = 80,
    pre_days: int = 504,
    post_days: int = 1260,
    proposal_gap: int = 15,
    maximum_proposals: int = 50,
    correlation: float = 0.5,
    level: float = 0.10,
    inspect_every: int = 5,
) -> dict[str, int]:
    """Untuned conditionally symmetric heavy-tail/volatility null diagnostic.

    Innovations are independent Student-t_3 draws standardized to unit
    variance.  A common predictable variance follows

        h_0 = 1,
        h_t = 0.05 + 0.10 * u_{t-1}^2 + 0.85 * h_{t-1},

    where u is the standardized common innovation.  At date t, every stream
    is multiplied by 0.25*sqrt(h_t), so the scale is measurable before the
    current symmetric innovations arrive.  Cross-sectional correlation is
    induced by sqrt(rho) common plus sqrt(1-rho) idiosyncratic innovations,
    and final scores are clipped to [-1,1].
    """
    rng = np.random.default_rng(seed)
    total_days = pre_days + post_days
    standardizer = np.sqrt(3.0)
    common = rng.standard_t(3.0, size=(total_days, 1)) / standardizer
    idiosyncratic = (
        rng.standard_t(3.0, size=(total_days, candidates)) / standardizer
    )
    variance = np.ones(total_days, dtype=float)
    for t in range(1, total_days):
        variance[t] = (
            0.05
            + 0.10 * float(common[t - 1, 0] ** 2)
            + 0.85 * variance[t - 1]
        )
    innovations = (
        np.sqrt(correlation) * common
        + np.sqrt(max(1.0 - correlation, 0.0)) * idiosyncratic
    )
    scores = np.clip(0.25 * np.sqrt(variance)[:, None] * innovations, -1.0, 1.0)
    proposal_times = np.arange(
        pre_days, total_days - 63, proposal_gap, dtype=int
    )
    selected, arrivals = _adaptive_order(
        scores,
        proposal_times,
        maximum=maximum_proposals,
        forced_first=int(np.argmax(scores[:pre_days].mean(axis=0))),
        lookback=pre_days,
    )
    values = scores[:, selected]
    lambdas = _initial_lambdas(scores, selected, arrivals, lookback=pre_days)
    gammas = spending_weights(len(selected))
    safe = _run_common_evidence(
        values,
        arrivals,
        lambdas,
        gammas,
        levels=np.array([level]),
        inspect_every=inspect_every,
    )[0]

    leakage_e = np.ones(len(selected), dtype=float)
    leakage_promoted = np.zeros(len(selected), dtype=bool)
    sequential_promoted = np.zeros(len(selected), dtype=bool)
    sums = np.zeros(len(selected), dtype=float)
    sums_of_squares = np.zeros(len(selected), dtype=float)
    observations = np.zeros(len(selected), dtype=int)
    leakage_first = -1
    sequential_first = -1
    for t in range(pre_days, total_days):
        active = arrivals < t
        observation = values[t]
        leakage_bets = lambdas * (observation >= 0.0)
        leakage_update = active & ~leakage_promoted
        leakage_e[leakage_update] *= (
            1.0
            + leakage_bets[leakage_update] * observation[leakage_update]
        )
        sequential_update = active & ~sequential_promoted
        sums[sequential_update] += observation[sequential_update]
        sums_of_squares[sequential_update] += observation[sequential_update] ** 2
        observations[sequential_update] += 1
        if t % inspect_every == 0 or t == total_days - 1:
            leakage_input = np.zeros(len(selected), dtype=float)
            leakage_input[active | leakage_promoted] = leakage_e[
                active | leakage_promoted
            ]
            leakage_promoted |= weighted_ebh(
                leakage_input,
                gammas,
                level,
                previously_rejected=leakage_promoted,
                admission_floor=1.0 / level,
            )
            enough = observations >= 63
            variance_estimate = np.maximum(
                (
                    sums_of_squares
                    - np.divide(
                        sums * sums,
                        np.maximum(observations, 1),
                        out=np.zeros_like(sums),
                        where=observations > 0,
                    )
                )
                / np.maximum(observations - 1, 1),
                1e-10,
            )
            statistic = np.divide(
                sums,
                np.sqrt(variance_estimate * np.maximum(observations, 1)),
                out=np.zeros_like(sums),
                where=observations > 0,
            )
            critical = student_t.ppf(0.95, np.maximum(observations - 1, 1))
            sequential_promoted |= enough & (statistic > critical)
            if leakage_first < 0 and leakage_promoted.any():
                leakage_first = t - pre_days
            if sequential_first < 0 and sequential_promoted.any():
                sequential_first = t - pre_days

    safe_days = safe.safe_promotion_day[safe.safe_promoted]
    return {
        "safe_count": int(safe.safe_promoted.sum()),
        "safe_first_day": int(safe_days.min() - pre_days) if len(safe_days) else -1,
        "leakage_count": int(leakage_promoted.sum()),
        "leakage_first_day": leakage_first,
        "sequential_count": int(sequential_promoted.sum()),
        "sequential_first_day": sequential_first,
    }
