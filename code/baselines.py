from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import kurtosis, skew

from strategies import annualized_sharpe


def circular_block_indices(
    observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = int(np.ceil(observations / block_length))
    starts = rng.integers(0, observations, size=blocks)
    offsets = np.arange(block_length)
    indices = (starts[:, None] + offsets[None, :]) % observations
    return indices.ravel()[:observations]


def stationary_bootstrap_indices(
    observations: int,
    mean_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis--Romano stationary-bootstrap indices with geometric blocks."""
    if observations < 1 or mean_block_length <= 1.0:
        raise ValueError("observations must be positive and mean block > 1")
    probability = 1.0 / mean_block_length
    indices = np.empty(observations, dtype=int)
    position = 0
    while position < observations:
        start = int(rng.integers(0, observations))
        length = min(int(rng.geometric(probability)), observations - position)
        indices[position : position + length] = (
            start + np.arange(length, dtype=int)
        ) % observations
        position += length
    return indices


def hansen_spa(
    returns: pd.DataFrame,
    repetitions: int = 1999,
    mean_block_length: float = 20.0,
    seed: int = 20260730,
) -> dict[str, float]:
    """Studentized Hansen SPA against a zero-return benchmark.

    The stationary bootstrap preserves each resampled row across all models.
    Long-run standard deviations are estimated from the bootstrap distribution
    of sqrt(n) times the mean.  We report Hansen's lower, consistent, and upper
    p-values.  The consistent recentering retains a negative sample mean only
    when its studentized value is below ``-sqrt(2 log log n)``; the lower rule
    retains every negative mean, while the upper rule treats every model as a
    binding null.
    """
    values = returns.dropna().to_numpy(dtype=float)
    observations, models = values.shape
    if observations < 3 or models < 1:
        raise ValueError("SPA requires at least three rows and one model")
    sample_means = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty((repetitions, models), dtype=float)
    for repetition in range(repetitions):
        indices = stationary_bootstrap_indices(
            observations, mean_block_length, rng
        )
        bootstrap_means[repetition] = values[indices].mean(axis=0)
    centered = np.sqrt(observations) * (
        bootstrap_means - sample_means[None, :]
    )
    long_run_sd = np.sqrt(np.mean(centered**2, axis=0))
    long_run_sd = np.maximum(long_run_sd, 1e-12)
    observed_components = np.sqrt(observations) * sample_means / long_run_sd
    observed = float(max(0.0, np.max(observed_components)))
    cutoff = -long_run_sd * np.sqrt(
        2.0 * np.log(np.log(float(observations))) / observations
    )
    recentering = {
        "lower": np.minimum(sample_means, 0.0),
        "consistent": sample_means * (sample_means < cutoff),
        "upper": np.zeros(models, dtype=float),
    }
    p_values: dict[str, float] = {}
    for label, null_mean in recentering.items():
        statistics = np.maximum(
            0.0,
            np.max(
                (
                    centered
                    + np.sqrt(observations) * null_mean[None, :]
                )
                / long_run_sd[None, :],
                axis=1,
            ),
        )
        p_values[f"p_value_{label}"] = float(
            (1 + np.sum(statistics >= observed)) / (repetitions + 1)
        )
    return {
        "observed_max_studentized_statistic": observed,
        **p_values,
        "repetitions": repetitions,
        "mean_block_length": mean_block_length,
        "seed": seed,
        "observations": observations,
        "models": models,
    }


def white_reality_check(
    returns: pd.DataFrame,
    repetitions: int = 1000,
    block_length: int = 20,
    seed: int = 161803,
) -> dict[str, float]:
    values = returns.dropna().to_numpy()
    observations = len(values)
    means = values.mean(axis=0)
    observed = float(np.sqrt(observations) * np.max(means))
    centered = values - means[None, :]
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(repetitions)
    for repetition in range(repetitions):
        indices = circular_block_indices(observations, block_length, rng)
        bootstrap[repetition] = np.sqrt(observations) * np.max(
            centered[indices].mean(axis=0)
        )
    probability = float((1 + np.sum(bootstrap >= observed)) / (repetitions + 1))
    return {
        "observed_max_statistic": observed,
        "p_value": probability,
        "repetitions": repetitions,
        "block_length": block_length,
    }


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes: pd.Series,
) -> dict[str, float]:
    values = returns.dropna().to_numpy()
    observations = len(values)
    estimated_sharpe = annualized_sharpe(values)
    daily_sharpe = estimated_sharpe / np.sqrt(252.0)
    trials = max(int(trial_sharpes.notna().sum()), 2)
    daily_trial_sharpes = trial_sharpes.dropna().to_numpy() / np.sqrt(252.0)
    trial_variance = float(np.var(daily_trial_sharpes, ddof=1))
    euler = 0.5772156649015329
    expected_maximum = np.sqrt(max(trial_variance, 1e-12)) * (
        (1.0 - euler) * ndtri(1.0 - 1.0 / trials)
        + euler * ndtri(1.0 - 1.0 / (trials * np.e))
    )
    sample_skewness = float(skew(values, bias=False))
    sample_kurtosis = float(kurtosis(values, fisher=False, bias=False))
    denominator = np.sqrt(
        max(
            1.0
            - sample_skewness * daily_sharpe
            + 0.25 * (sample_kurtosis - 1.0) * daily_sharpe**2,
            1e-12,
        )
    )
    statistic = (
        (daily_sharpe - expected_maximum)
        * np.sqrt(max(observations - 1, 1))
        / denominator
    )
    return {
        "estimated_annual_sharpe": estimated_sharpe,
        "expected_max_annual_sharpe": expected_maximum * np.sqrt(252.0),
        "deflated_sharpe_probability": float(ndtr(statistic)),
        "trials": trials,
        "skewness": sample_skewness,
        "kurtosis": sample_kurtosis,
    }


def probability_backtest_overfitting(
    returns: pd.DataFrame,
    blocks: int = 10,
) -> dict[str, float]:
    values = returns.dropna()
    if blocks % 2:
        raise ValueError("CSCV requires an even number of blocks")
    partitions = np.array_split(np.arange(len(values)), blocks)
    logits: list[float] = []
    degradation: list[float] = []
    columns = values.columns
    for train_blocks in itertools.combinations(range(blocks), blocks // 2):
        train_mask = np.concatenate([partitions[i] for i in train_blocks])
        test_blocks = sorted(set(range(blocks)) - set(train_blocks))
        test_mask = np.concatenate([partitions[i] for i in test_blocks])
        train = values.iloc[train_mask]
        test = values.iloc[test_mask]
        train_sharpes = train.mean() / train.std(ddof=1)
        winner = train_sharpes.idxmax()
        test_sharpes = test.mean() / test.std(ddof=1)
        rank = float(test_sharpes.rank(method="average")[winner])
        relative_rank = (rank - 0.5) / len(columns)
        relative_rank = float(np.clip(relative_rank, 1e-8, 1.0 - 1e-8))
        logits.append(float(np.log(relative_rank / (1.0 - relative_rank))))
        degradation.append(float(train_sharpes[winner] - test_sharpes[winner]))
    logits_array = np.asarray(logits)
    return {
        "pbo": float(np.mean(logits_array <= 0.0)),
        "median_logit": float(np.median(logits_array)),
        "mean_sharpe_degradation_daily": float(np.mean(degradation)),
        "splits": len(logits),
        "blocks": blocks,
    }
