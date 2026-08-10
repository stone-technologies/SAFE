from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from conservative_baseline import run_joint_and_alpha_spending
from power_upgrade import CAMPAIGN_75_GEOMETRIC_DAILY, design_spending_weights
from simulation import _adaptive_order, _initial_lambdas, _wilson


REPETITIONS = 500
SEED_START = 20_260_730
METHODS = (
    ("Final persistent weighted e-BH", "joint"),
    ("Proposal e-alpha-spending (FWER)", "alpha_spending"),
)


def _one(seed: int) -> list[dict[str, int | str]]:
    rng = np.random.default_rng(seed)
    candidates, pre_days, post_days = 80, 504, 1260
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
    innovations = np.sqrt(0.5) * common + np.sqrt(0.5) * idiosyncratic
    scores = np.clip(0.25 * np.sqrt(variance)[:, None] * innovations, -1.0, 1.0)
    proposal_times = np.arange(pre_days, total_days - 63, 15, dtype=int)
    selected, arrivals = _adaptive_order(
        scores,
        proposal_times,
        maximum=50,
        forced_first=int(np.argmax(scores[:pre_days].mean(axis=0))),
        lookback=pre_days,
    )
    lambdas = _initial_lambdas(scores, selected, arrivals, lookback=pre_days)
    design = CAMPAIGN_75_GEOMETRIC_DAILY
    comparison = run_joint_and_alpha_spending(
        values=scores[:, selected],
        arrivals=arrivals,
        base_lambdas=lambdas,
        gammas=design_spending_weights(len(arrivals), design, campaign_cap=50),
        bet_multipliers=design.bet_multipliers,
        level=0.10,
        inspect_every=design.inspect_every,
    )
    rows: list[dict[str, int | str]] = []
    for method, attribute in METHODS:
        outcome = getattr(comparison, attribute)
        days = outcome.promotion_day[outcome.promoted]
        rows.append(
            {
                "method": method,
                "seed": seed,
                "false_discoveries": int(outcome.promoted.sum()),
                "any_false": int(outcome.promoted.any()),
                "first_day": int(days.min() - pre_days) if len(days) else -1,
            }
        )
    return rows


def run(root: Path, jobs: int = 4) -> None:
    seeds = range(SEED_START, SEED_START + REPETITIONS)
    rows: list[dict[str, int | str]] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for result in executor.map(_one, seeds, chunksize=4):
            rows.extend(result)
    frame = pd.DataFrame(rows)
    results = root / "results"
    frame.to_csv(results / "power_upgrade_final_heavy_null_seed_results.csv", index=False)
    summary_rows = []
    for method, group in frame.groupby("method"):
        successes = int(group["any_false"].sum())
        low, high = _wilson(successes, len(group))
        summary_rows.append(
            {
                "method": method,
                "false_discovery_probability": successes / len(group),
                "ci_low": low,
                "ci_high": high,
                "mean_false_discoveries": float(group["false_discoveries"].mean()),
                "repetitions": len(group),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        results / "power_upgrade_final_heavy_null_summary.csv", index=False
    )


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
