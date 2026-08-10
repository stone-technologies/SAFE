from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from conservative_baseline import ProcedureOutcome, run_joint_and_alpha_spending
from data import load_panel
from inference import bounded_scores
from power_upgrade import (
    ANCHORED_75_DAILY,
    BASELINE,
    CAMPAIGN_75_DAILY,
    CAMPAIGN_75_GEOMETRIC_DAILY,
    UpgradeDesign,
    design_spending_weights,
    planted_path,
)
from simulation import (
    _adaptive_order,
    _initial_lambdas,
    _wilson,
    matched_seed_blocks,
)
from strategies import build_candidates


DESIGNS = {
    design.name: design
    for design in (
        BASELINE,
        CAMPAIGN_75_DAILY,
        CAMPAIGN_75_GEOMETRIC_DAILY,
        ANCHORED_75_DAILY,
    )
}
PRIMARY_EFFECT = 1.5
PRIMARY_CORRELATION = 0.5
METHOD_JOINT = "Persistent weighted e-BH"
METHOD_FWER = "Proposal e-alpha-spending (FWER)"


def _summarize(
    outcome: ProcedureOutcome,
    selected_truth: np.ndarray,
    all_truth: np.ndarray,
    arrivals: np.ndarray,
) -> dict[str, float]:
    discoveries = outcome.promoted
    total = int(discoveries.sum())
    true = int(np.sum(discoveries & selected_truth))
    false = int(np.sum(discoveries & ~selected_truth))
    delays = outcome.promotion_day[discoveries & selected_truth] - arrivals[
        discoveries & selected_truth
    ]
    return {
        "fdp": false / max(total, 1),
        "end_to_end_power": true / max(int(all_truth.sum()), 1),
        "conditional_power": true / max(int(selected_truth.sum()), 1),
        "proposal_recall": int(selected_truth.sum())
        / max(int(all_truth.sum()), 1),
        "discoveries": total,
        "true_discoveries": true,
        "false_discoveries": false,
        "delay": float(np.mean(delays)) if len(delays) else np.nan,
    }


def _one_planted(
    task: tuple[float, float, int, UpgradeDesign]
) -> list[dict[str, float | int | str]]:
    annual_sharpe, correlation, seed, design = task
    values, arrivals, lambdas, selected_truth, truth = planted_path(
        annual_sharpe=annual_sharpe,
        correlation=correlation,
        seed=seed,
    )
    comparison = run_joint_and_alpha_spending(
        values=values,
        arrivals=arrivals,
        base_lambdas=lambdas,
        gammas=design_spending_weights(
            len(arrivals), design, campaign_cap=50
        ),
        bet_multipliers=design.bet_multipliers,
        level=0.10,
        inspect_every=design.inspect_every,
    )
    rows: list[dict[str, float | int | str]] = []
    for method, outcome in [
        (METHOD_JOINT, comparison.joint),
        (METHOD_FWER, comparison.alpha_spending),
    ]:
        rows.append(
            {
                "design": design.name,
                "method": method,
                "annual_sharpe": annual_sharpe,
                "correlation": correlation,
                "seed": seed,
                **_summarize(outcome, selected_truth, truth, arrivals),
            }
        )
    return rows


_NULL_BASE_SCORES: np.ndarray | None = None
_NULL_START = 0
_NULL_DESIGN = ANCHORED_75_DAILY


def _initialize_null(
    base_scores: np.ndarray,
    start_location: int,
    design: UpgradeDesign,
) -> None:
    global _NULL_BASE_SCORES, _NULL_START, _NULL_DESIGN
    _NULL_BASE_SCORES = base_scores
    _NULL_START = start_location
    _NULL_DESIGN = design


def _one_null(seed: int) -> list[dict[str, int | str]]:
    if _NULL_BASE_SCORES is None:
        raise RuntimeError("null worker was not initialized")
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(_NULL_BASE_SCORES))
    scores = np.nan_to_num(_NULL_BASE_SCORES, nan=0.0) * signs[:, None]
    proposal_times = np.arange(
        _NULL_START, len(scores) - 63, 63, dtype=int
    )
    selected, arrivals = _adaptive_order(
        scores, proposal_times, maximum=77
    )
    values = scores[:, selected]
    lambdas = _initial_lambdas(scores, selected, arrivals)
    comparison = run_joint_and_alpha_spending(
        values=values,
        arrivals=arrivals,
        base_lambdas=lambdas,
        gammas=design_spending_weights(
            len(arrivals), _NULL_DESIGN, campaign_cap=77
        ),
        bet_multipliers=_NULL_DESIGN.bet_multipliers,
        level=0.10,
        inspect_every=_NULL_DESIGN.inspect_every,
    )
    rows: list[dict[str, int | str]] = []
    for method, outcome in [
        (METHOD_JOINT, comparison.joint),
        (METHOD_FWER, comparison.alpha_spending),
    ]:
        promoted_days = outcome.promotion_day[outcome.promoted]
        rows.append(
            {
                "design": _NULL_DESIGN.name,
                "method": method,
                "seed": seed,
                "false_discoveries": int(outcome.promoted.sum()),
                "any_false": int(outcome.promoted.any()),
                "first_day": (
                    int(promoted_days.min() - _NULL_START)
                    if len(promoted_days)
                    else -1
                ),
            }
        )
    return rows


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    half = 1.96 * float(array.std(ddof=1)) / np.sqrt(len(array))
    return mean, mean - half, mean + half


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(
    root: Path,
    design: UpgradeDesign,
    jobs: int = 4,
    fast: bool = False,
    skip_null: bool = False,
) -> None:
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    repetitions = 25 if fast else 250
    mapping = matched_seed_blocks(repetitions=repetitions)
    if fast:
        mapping = mapping[
            (mapping["annual_sharpe"] == PRIMARY_EFFECT)
            & (mapping["correlation"] == PRIMARY_CORRELATION)
        ]
    tasks: list[tuple[float, float, int, UpgradeDesign]] = []
    for row in mapping.itertuples(index=False):
        for seed in range(int(row.seed_first), int(row.seed_last) + 1):
            tasks.append(
                (float(row.annual_sharpe), float(row.correlation), seed, design)
            )

    planted_rows: list[dict[str, float | int | str]] = []
    if jobs == 1:
        for rows in map(_one_planted, tasks):
            planted_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for rows in executor.map(_one_planted, tasks, chunksize=4):
                planted_rows.extend(rows)
    planted = pd.DataFrame(planted_rows)
    prefix = f"conservative_baseline_{design.name}"
    planted.to_csv(results / f"{prefix}_power_seed_results.csv", index=False)
    metrics = [
        "fdp",
        "end_to_end_power",
        "conditional_power",
        "proposal_recall",
        "discoveries",
        "true_discoveries",
        "false_discoveries",
        "delay",
    ]
    summary = (
        planted.groupby(
            ["design", "method", "annual_sharpe", "correlation"],
            as_index=False,
        )[metrics]
        .mean()
        .sort_values(["correlation", "annual_sharpe", "method"])
    )
    summary.to_csv(results / f"{prefix}_power_summary.csv", index=False)

    primary = planted[
        (planted["annual_sharpe"] == PRIMARY_EFFECT)
        & (planted["correlation"] == PRIMARY_CORRELATION)
    ].pivot(index="seed", columns="method", values="end_to_end_power")
    difference = primary[METHOD_JOINT] - primary[METHOD_FWER]
    difference_mean, difference_low, difference_high = _mean_ci(difference)
    paired = pd.DataFrame(
        [
            {
                "design": design.name,
                "annual_sharpe": PRIMARY_EFFECT,
                "correlation": PRIMARY_CORRELATION,
                "repetitions": len(difference),
                "joint_power": float(primary[METHOD_JOINT].mean()),
                "alpha_spending_power": float(primary[METHOD_FWER].mean()),
                "paired_difference": difference_mean,
                "paired_ci_low": difference_low,
                "paired_ci_high": difference_high,
                "joint_wins": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
                "joint_losses": int((difference < 0).sum()),
            }
        ]
    )
    paired.to_csv(results / f"{prefix}_primary_paired.csv", index=False)

    null_repetitions = 25 if fast else 500
    if not skip_null:
        industries, _, risk_free = load_panel(root / "data" / "raw")
        candidate_returns, _, _ = build_candidates(
            industries, risk_free, cost_bps=2.0
        )
        score_frame = bounded_scores(candidate_returns).loc[
            "2002-01-01":"2026-05-29"
        ]
        start = int(score_frame.index.searchsorted(pd.Timestamp("2007-01-03")))
        seeds = list(range(271828, 271828 + null_repetitions))
        null_rows: list[dict[str, int | str]] = []
        if jobs == 1:
            _initialize_null(score_frame.to_numpy(), start, design)
            for rows in map(_one_null, seeds):
                null_rows.extend(rows)
        else:
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_initialize_null,
                initargs=(score_frame.to_numpy(), start, design),
            ) as executor:
                for rows in executor.map(_one_null, seeds, chunksize=2):
                    null_rows.extend(rows)
        null = pd.DataFrame(null_rows)
        null.to_csv(results / f"{prefix}_null_seed_results.csv", index=False)
        null_summary_rows = []
        for method, frame in null.groupby("method"):
            successes = int(frame["any_false"].sum())
            low, high = _wilson(successes, len(frame))
            days = frame.loc[frame["first_day"] >= 0, "first_day"]
            null_summary_rows.append(
                {
                    "design": design.name,
                    "method": method,
                    "false_discovery_probability": successes / len(frame),
                    "ci_low": low,
                    "ci_high": high,
                    "mean_false_discoveries": float(
                        frame["false_discoveries"].mean()
                    ),
                    "median_first_day_if_any": (
                        float(days.median()) if len(days) else np.nan
                    ),
                    "repetitions": len(frame),
                }
            )
        pd.DataFrame(null_summary_rows).to_csv(
            results / f"{prefix}_null_summary.csv", index=False
        )

    _write_json(
        results / f"{prefix}_manifest.json",
        {
            "design": design.name,
            "level": 0.10,
            "admission_floor": 10.0,
            "inspection_cadence": design.inspect_every,
            "proposal_weights": (
                "identical design_spending_weights for both procedures"
            ),
            "betting_rule": (
                "identical proposal-time fixed bets and mixture components"
            ),
            "joint_method": METHOD_JOINT,
            "conservative_method": METHOD_FWER,
            "conservative_boundary": "E_j(t) >= 1 / (q gamma_j)",
            "conservative_guarantee": (
                "FWER <= q by Ville plus the proposal-weight union bound; "
                "arbitrary dependence and adaptive arrivals allowed under the "
                "global conditional e-process assumptions"
            ),
            "planted_repetitions_per_cell": repetitions,
            "null_repetitions": 0 if skip_null else null_repetitions,
            "fast": fast,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--design", choices=sorted(DESIGNS), default=ANCHORED_75_DAILY.name
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--skip-null", action="store_true")
    args = parser.parse_args()
    run(
        args.root.resolve(),
        DESIGNS[args.design],
        jobs=args.jobs,
        fast=args.fast,
        skip_null=args.skip_null,
    )
