from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from power_upgrade import (
    DEVELOPMENT_DESIGNS,
    design_spending_weights,
    planted_path,
    run_mixture_gate,
    summarize_design,
)


CAMPAIGN_CAP = 50


def _one_repetition(task: tuple[float, float, int]) -> list[dict[str, float]]:
    annual_sharpe, correlation, seed = task
    values, arrivals, lambdas, selected_truth, truth = planted_path(
        annual_sharpe=annual_sharpe,
        correlation=correlation,
        seed=seed,
    )
    rows: list[dict[str, float]] = []
    for design in DEVELOPMENT_DESIGNS:
        gammas = design_spending_weights(
            len(arrivals), design, campaign_cap=CAMPAIGN_CAP
        )
        outcome = run_mixture_gate(
            values=values,
            arrivals=arrivals,
            base_lambdas=lambdas,
            gammas=gammas,
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
    return rows


def run(
    output: Path,
    seed_start: int,
    repetitions: int,
    annual_sharpes: tuple[float, ...],
    correlations: tuple[float, ...],
    jobs: int,
) -> None:
    tasks: list[tuple[float, float, int]] = []
    counter = 0
    for correlation in correlations:
        for annual_sharpe in annual_sharpes:
            for repetition in range(repetitions):
                tasks.append(
                    (annual_sharpe, correlation, seed_start + counter + repetition)
                )
            counter += repetitions

    rows: list[dict[str, float]] = []
    if jobs == 1:
        for task in tasks:
            rows.extend(_one_repetition(task))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for result in executor.map(_one_repetition, tasks, chunksize=4):
                rows.extend(result)

    seeds = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(output, index=False)
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
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--annual-sharpes", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--correlations", type=float, nargs="+", default=[0.5])
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args()
    run(
        output=arguments.output,
        seed_start=arguments.seed_start,
        repetitions=arguments.repetitions,
        annual_sharpes=tuple(arguments.annual_sharpes),
        correlations=tuple(arguments.correlations),
        jobs=arguments.jobs,
    )
