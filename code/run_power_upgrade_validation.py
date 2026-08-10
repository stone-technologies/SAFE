from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from power_upgrade import (
    BASELINE,
    CAMPAIGN_75_DAILY,
    design_spending_weights,
    planted_path,
    run_mixture_gate,
    summarize_design,
)
from simulation import matched_seed_blocks


DESIGNS = (BASELINE, CAMPAIGN_75_DAILY)
PRIMARY_EFFECT = 1.5
PRIMARY_CORRELATION = 0.5
RANK_BINS = np.array([0, 5, 10, 20, 35, 50])
RANK_LABELS = ("1-5", "6-10", "11-20", "21-35", "36-50")
CAMPAIGN_CAP = 50


def _one_repetition(
    task: tuple[float, float, int]
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    annual_sharpe, correlation, seed = task
    values, arrivals, lambdas, selected_truth, truth = planted_path(
        annual_sharpe=annual_sharpe,
        correlation=correlation,
        seed=seed,
    )
    summaries: list[dict[str, float]] = []
    rank_rows: list[dict[str, float]] = []
    for design in DESIGNS:
        outcome = run_mixture_gate(
            values=values,
            arrivals=arrivals,
            base_lambdas=lambdas,
            gammas=design_spending_weights(
                len(arrivals), design, campaign_cap=CAMPAIGN_CAP
            ),
            bet_multipliers=design.bet_multipliers,
            inspect_every=design.inspect_every,
        )
        summaries.append(
            {
                "design": design.name,
                "annual_sharpe": annual_sharpe,
                "correlation": correlation,
                "seed": seed,
                **summarize_design(
                    outcome,
                    selected_truth=selected_truth,
                    all_truth=truth,
                    arrivals=arrivals,
                ),
            }
        )
        if annual_sharpe == PRIMARY_EFFECT and correlation == PRIMARY_CORRELATION:
            ranks = np.arange(len(arrivals))
            for left, right, label in zip(
                RANK_BINS[:-1], RANK_BINS[1:], RANK_LABELS
            ):
                mask = (ranks >= left) & (ranks < right) & selected_truth
                rank_rows.append(
                    {
                        "design": design.name,
                        "annual_sharpe": annual_sharpe,
                        "correlation": correlation,
                        "seed": seed,
                        "rank_bin": label,
                        "true_proposals": int(mask.sum()),
                        "true_promotions": int((mask & outcome.promoted).sum()),
                    }
                )
    return summaries, rank_rows


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    half = 1.96 * float(array.std(ddof=1)) / np.sqrt(len(array))
    return mean, mean - half, mean + half


def run(root: Path, jobs: int = 4) -> None:
    results = root / "results"
    mapping = matched_seed_blocks(repetitions=250)
    tasks: list[tuple[float, float, int]] = []
    for row in mapping.itertuples(index=False):
        for seed in range(int(row.seed_first), int(row.seed_last) + 1):
            tasks.append((float(row.annual_sharpe), float(row.correlation), seed))

    summary_rows: list[dict[str, float]] = []
    rank_rows: list[dict[str, float]] = []
    if jobs == 1:
        iterator = map(_one_repetition, tasks)
        for summaries, ranks in iterator:
            summary_rows.extend(summaries)
            rank_rows.extend(ranks)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for summaries, ranks in executor.map(
                _one_repetition, tasks, chunksize=4
            ):
                summary_rows.extend(summaries)
                rank_rows.extend(ranks)

    seeds = pd.DataFrame(summary_rows)
    seeds.to_csv(results / "power_upgrade_validation_seed_results.csv", index=False)
    metrics = [
        "fdp",
        "end_to_end_power",
        "conditional_power",
        "proposal_recall",
        "discoveries",
        "delay",
        "terminal_fdp",
        "terminal_end_to_end_power",
    ]
    summary = (
        seeds.groupby(["design", "annual_sharpe", "correlation"], as_index=False)[
            metrics
        ]
        .mean()
        .sort_values(["correlation", "annual_sharpe", "design"])
    )
    summary.to_csv(results / "power_upgrade_validation_summary.csv", index=False)

    primary = seeds[
        (seeds["annual_sharpe"] == PRIMARY_EFFECT)
        & (seeds["correlation"] == PRIMARY_CORRELATION)
    ].pivot(index="seed", columns="design", values="end_to_end_power")
    difference = primary[CAMPAIGN_75_DAILY.name] - primary[BASELINE.name]
    mean, low, high = _mean_ci(difference)
    paired = pd.DataFrame(
        [
            {
                "annual_sharpe": PRIMARY_EFFECT,
                "correlation": PRIMARY_CORRELATION,
                "repetitions": len(difference),
                "baseline_power": float(primary[BASELINE.name].mean()),
                "campaign_power": float(primary[CAMPAIGN_75_DAILY.name].mean()),
                "paired_difference": mean,
                "paired_ci_low": low,
                "paired_ci_high": high,
                "campaign_wins": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
                "campaign_losses": int((difference < 0).sum()),
            }
        ]
    )
    paired.to_csv(results / "power_upgrade_primary_paired.csv", index=False)

    rank_seed = pd.DataFrame(rank_rows)
    rank_seed.to_csv(results / "power_upgrade_rank_seed_results.csv", index=False)
    rank_summary = (
        rank_seed.groupby(["design", "rank_bin"], as_index=False)[
            ["true_proposals", "true_promotions"]
        ]
        .sum()
    )
    rank_summary["power"] = (
        rank_summary["true_promotions"] / rank_summary["true_proposals"]
    )
    rank_summary["rank_bin"] = pd.Categorical(
        rank_summary["rank_bin"], categories=RANK_LABELS, ordered=True
    )
    rank_summary.sort_values(["rank_bin", "design"]).to_csv(
        results / "power_upgrade_rank_summary.csv", index=False
    )


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1], jobs=4)
