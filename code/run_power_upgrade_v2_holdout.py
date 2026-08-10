from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from power_upgrade import (
    BASELINE,
    CAMPAIGN_75_DAILY,
    CAMPAIGN_75_GEOMETRIC_DAILY,
    design_spending_weights,
    planted_path,
    run_mixture_gate,
    summarize_design,
)


DESIGNS = (BASELINE, CAMPAIGN_75_DAILY, CAMPAIGN_75_GEOMETRIC_DAILY)
CAMPAIGN_CAP = 50
ANNUAL_SHARPES = (1.0, 1.5, 2.0)
CORRELATIONS = (0.0, 0.5)
REPETITIONS = 250
SEED_START = 52_000_000
PRIMARY_EFFECT = 1.5
PRIMARY_CORRELATION = 0.5
RANK_EDGES = np.array([0, 5, 10, 20, 35, 50])
RANK_LABELS = ("1-5", "6-10", "11-20", "21-35", "36-50")


def _one_repetition(
    task: tuple[float, float, int]
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    annual_sharpe, correlation, seed = task
    values, arrivals, lambdas, selected_truth, truth = planted_path(
        annual_sharpe=annual_sharpe,
        correlation=correlation,
        seed=seed,
    )
    rows: list[dict[str, float]] = []
    rank_rows: list[dict[str, float]] = []
    ranks = np.arange(len(arrivals))
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
        rows.append(
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
            for left, right, label in zip(
                RANK_EDGES[:-1], RANK_EDGES[1:], RANK_LABELS
            ):
                mask = (ranks >= left) & (ranks < right) & selected_truth
                rank_rows.append(
                    {
                        "design": design.name,
                        "seed": seed,
                        "rank_bin": label,
                        "true_proposals": int(mask.sum()),
                        "true_promotions": int((mask & outcome.promoted).sum()),
                    }
                )
    return rows, rank_rows


def _paired_interval(values: pd.Series) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    half_width = 1.96 * float(array.std(ddof=1)) / np.sqrt(len(array))
    return mean, mean - half_width, mean + half_width


def run(root: Path, jobs: int = 4) -> None:
    tasks: list[tuple[float, float, int]] = []
    block = 0
    for correlation in CORRELATIONS:
        for annual_sharpe in ANNUAL_SHARPES:
            first = SEED_START + block * REPETITIONS
            tasks.extend(
                (annual_sharpe, correlation, first + repetition)
                for repetition in range(REPETITIONS)
            )
            block += 1

    rows: list[dict[str, float]] = []
    rank_rows: list[dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for summaries, ranks in executor.map(_one_repetition, tasks, chunksize=4):
            rows.extend(summaries)
            rank_rows.extend(ranks)

    results = root / "results"
    seeds = pd.DataFrame(rows)
    seeds.to_csv(results / "power_upgrade_v2_holdout_seed_results.csv", index=False)
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
    summary.to_csv(results / "power_upgrade_v2_holdout_summary.csv", index=False)

    primary = seeds[
        (seeds["annual_sharpe"] == PRIMARY_EFFECT)
        & (seeds["correlation"] == PRIMARY_CORRELATION)
    ].pivot(index="seed", columns="design", values="end_to_end_power")
    comparisons = []
    final_name = CAMPAIGN_75_GEOMETRIC_DAILY.name
    for comparator in (BASELINE.name, CAMPAIGN_75_DAILY.name):
        difference = primary[final_name] - primary[comparator]
        mean, low, high = _paired_interval(difference)
        comparisons.append(
            {
                "design": final_name,
                "comparator": comparator,
                "repetitions": len(difference),
                "design_power": float(primary[final_name].mean()),
                "comparator_power": float(primary[comparator].mean()),
                "paired_difference": mean,
                "paired_ci_low": low,
                "paired_ci_high": high,
                "wins": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
                "losses": int((difference < 0).sum()),
            }
        )
    pd.DataFrame(comparisons).to_csv(
        results / "power_upgrade_v2_primary_paired.csv", index=False
    )

    rank_seed = pd.DataFrame(rank_rows)
    rank_seed.to_csv(results / "power_upgrade_v2_rank_seed_results.csv", index=False)
    rank_summary = rank_seed.groupby(["design", "rank_bin"], as_index=False)[
        ["true_proposals", "true_promotions"]
    ].sum()
    rank_summary["power"] = (
        rank_summary["true_promotions"] / rank_summary["true_proposals"]
    )
    rank_summary["rank_bin"] = pd.Categorical(
        rank_summary["rank_bin"], categories=RANK_LABELS, ordered=True
    )
    rank_summary.sort_values(["rank_bin", "design"]).to_csv(
        results / "power_upgrade_v2_rank_summary.csv", index=False
    )


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
