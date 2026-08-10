from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import (
    deflated_sharpe_ratio,
    probability_backtest_overfitting,
    white_reality_check,
)
from data import download, load_panel, sha256, write_processed
from inference import (
    bounded_scores,
    first_rejection_date,
    run_gate,
    terminal_rejections,
)
from portfolios import (
    fixed_split_portfolio,
    full_sample_topk,
    performance_metrics,
    rolling_topk_portfolio,
    safe_alpha_portfolio,
)
from simulation import null_calibration, planted_power
from strategies import adaptive_proposals, annualized_sharpe, build_candidates


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def run(
    root: Path,
    fast: bool = False,
    skip_simulations: bool = False,
) -> None:
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    provenance = download(raw_dir)
    industries, factors, risk_free = load_panel(raw_dir)
    write_processed(processed_dir, industries, factors, risk_free)

    metric_rows: list[dict[str, object]] = []
    matched_metric_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    main_returns: pd.DataFrame | None = None
    main_turnover: pd.DataFrame | None = None
    main_proposals: pd.DataFrame | None = None
    main_gate = None
    reference_returns, reference_turnover, reference_specs = build_candidates(
        industries, risk_free, cost_bps=2.0
    )
    reference_proposals = adaptive_proposals(
        reference_returns,
        start="2007-01-03",
        end="2025-12-31",
        lookback_days=1260,
        proposals_per_quarter=1,
        diversity_penalty=0.20,
    )
    reference_testing_returns = reference_returns.loc[
        "2002-01-01":"2026-05-29"
    ]
    reference_gate = run_gate(
        reference_testing_returns,
        reference_proposals,
        level=0.10,
        inspect_every=5,
        scale_window=63,
        scale_multiple=4.0,
    )

    for cost_bps in [2.0, 5.0, 10.0]:
        if cost_bps == 2.0:
            candidate_returns = reference_returns
            candidate_turnover = reference_turnover
            specs = reference_specs
        else:
            candidate_returns, candidate_turnover, specs = build_candidates(
                industries, risk_free, cost_bps=cost_bps
            )
        proposals = reference_proposals.copy()
        testing_returns = candidate_returns.loc["2002-01-01":"2026-05-29"]
        if cost_bps == 2.0:
            gate = reference_gate
        else:
            gate = run_gate(
                testing_returns,
                proposals,
                level=0.10,
                inspect_every=5,
                scale_window=63,
                scale_multiple=4.0,
                scale_reference=reference_testing_returns,
                fixed_lambdas=reference_gate.lambdas,
            )
        unsafe_gate = run_gate(
            testing_returns,
            proposals,
            level=0.10,
            inspect_every=5,
            scale_window=63,
            scale_multiple=4.0,
            unsafe_same_bar=True,
            scale_reference=reference_testing_returns,
            fixed_lambdas=reference_gate.lambdas,
        )
        promoted = terminal_rejections(gate)
        unsafe_promoted = terminal_rejections(unsafe_gate)
        first_date = first_rejection_date(gate)

        safe_returns, safe_turnover, _ = safe_alpha_portfolio(
            candidate_returns.loc["2007-01-03":"2026-05-29"],
            candidate_turnover.loc["2007-01-03":"2026-05-29"],
            gate,
            risk_free,
            cost_bps,
        )
        if first_date is None:
            active_start = safe_returns.index[-1]
        else:
            deployable = safe_returns.index[safe_returns.index > first_date]
            active_start = deployable[0] if len(deployable) else safe_returns.index[-1]
        safe_metrics = performance_metrics(
            safe_returns.loc[active_start:], safe_turnover.loc[active_start:]
        )
        metric_rows.append(
            {
                "method": "SAFE-ALPHA",
                "cost_bps": cost_bps,
                "sample": (
                    f"{pd.Timestamp(active_start).date()}--"
                    f"{safe_returns.index[-1].date()}"
                ),
                "selected": len(promoted),
                **safe_metrics,
            }
        )
        matched_metric_rows.append(
            {
                "method": "SAFE-ALPHA",
                "cost_bps": cost_bps,
                "sample": (
                    f"{pd.Timestamp(active_start).date()}--"
                    f"{safe_returns.index[-1].date()}"
                ),
                "selected": len(promoted),
                **safe_metrics,
            }
        )

        rolling_returns, rolling_turnover, _ = rolling_topk_portfolio(
            candidate_returns,
            candidate_turnover,
            start="2007-01-03",
            end="2026-05-29",
            risk_free=risk_free,
            cost_bps=cost_bps,
            lookback_days=1260,
            top_k=5,
        )
        metric_rows.append(
            {
                "method": "Ungated adaptive top-5",
                "cost_bps": cost_bps,
                "sample": "2007-01-03--2026-05-29",
                "selected": 5,
                **performance_metrics(rolling_returns, rolling_turnover),
            }
        )
        matched_metric_rows.append(
            {
                "method": "Ungated adaptive top-5",
                "cost_bps": cost_bps,
                "sample": (
                    f"{pd.Timestamp(active_start).date()}--"
                    f"{rolling_returns.index[-1].date()}"
                ),
                "selected": 5,
                **performance_metrics(
                    rolling_returns.loc[active_start:],
                    rolling_turnover.loc[active_start:],
                ),
            }
        )

        split_returns, split_turnover, split_chosen = fixed_split_portfolio(
            candidate_returns,
            candidate_turnover,
            risk_free,
            cost_bps,
        )
        metric_rows.append(
            {
                "method": "Fixed train/validation/test",
                "cost_bps": cost_bps,
                "sample": "2015-01-02--2026-05-29",
                "selected": len(split_chosen),
                **performance_metrics(split_returns, split_turnover),
            }
        )
        matched_metric_rows.append(
            {
                "method": "Fixed train/validation/test",
                "cost_bps": cost_bps,
                "sample": (
                    f"{pd.Timestamp(active_start).date()}--"
                    f"{split_returns.index[-1].date()}"
                ),
                "selected": len(split_chosen),
                **performance_metrics(
                    split_returns.loc[active_start:],
                    split_turnover.loc[active_start:],
                ),
            }
        )

        biased_returns, biased_turnover, biased_chosen = full_sample_topk(
            candidate_returns,
            candidate_turnover,
            risk_free,
            cost_bps,
        )
        metric_rows.append(
            {
                "method": "Full-sample top-5",
                "cost_bps": cost_bps,
                "sample": "2007-01-03--2026-05-29 (in sample)",
                "selected": len(biased_chosen),
                **performance_metrics(biased_returns, biased_turnover),
            }
        )
        matched_metric_rows.append(
            {
                "method": "Full-sample top-5",
                "cost_bps": cost_bps,
                "sample": (
                    f"{pd.Timestamp(active_start).date()}--"
                    f"{biased_returns.index[-1].date()} (selected in sample)"
                ),
                "selected": len(biased_chosen),
                **performance_metrics(
                    biased_returns.loc[active_start:],
                    biased_turnover.loc[active_start:],
                ),
            }
        )

        sensitivity_rows.append(
            {
                "cost_bps": cost_bps,
                "new_admission_floor": 10.0,
                "proposals": len(proposals),
                "safe_promotions": len(promoted),
                "first_safe_promotion": first_date,
                "unsafe_same_bar_promotions": len(unsafe_promoted),
                "safe_terminal_set": "|".join(promoted),
                "unsafe_terminal_set": "|".join(unsafe_promoted),
            }
        )

        if cost_bps == 2.0:
            main_returns = candidate_returns
            main_turnover = candidate_turnover
            main_proposals = proposals
            main_gate = gate
            pd.DataFrame([spec.to_dict() for spec in specs]).to_csv(
                results_dir / "strategy_catalog.csv", index=False
            )
            proposals.to_csv(results_dir / "proposal_ledger.csv", index=False)
            gate.e_values.to_csv(
                results_dir / "e_value_paths.csv",
                float_format="%.10g",
            )
            pd.DataFrame(
                {
                    "strategy_id": gate.promotion_dates.index,
                    "promotion_date": gate.promotion_dates.values,
                    "frozen_e_value": [
                        (
                            gate.e_values.loc[date, strategy_id]
                            if pd.notna(date)
                            else np.nan
                        )
                        for strategy_id, date in gate.promotion_dates.items()
                    ],
                    "gamma": gate.gammas.values,
                    "lambda": gate.lambdas.values,
                    "new_admission_floor": np.repeat(10.0, len(gate.gammas)),
                }
            ).to_csv(results_dir / "certification_ledger.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results_dir / "economic_metrics.csv", index=False)
    pd.DataFrame(matched_metric_rows).to_csv(
        results_dir / "economic_metrics_matched.csv", index=False
    )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(results_dir / "cost_sensitivity.csv", index=False)

    if main_returns is None or main_proposals is None or main_gate is None:
        raise RuntimeError("Main experiment was not initialized")

    sample = main_returns.loc["2007-01-03":"2026-05-29"].dropna()
    sharpes = sample.apply(annualized_sharpe)
    winner = sharpes.idxmax()
    classical = {
        "white_reality_check": white_reality_check(
            sample,
            repetitions=200 if fast else 1000,
            block_length=20,
        ),
        "deflated_sharpe_ratio": deflated_sharpe_ratio(
            sample[winner], sharpes
        ),
        "probability_backtest_overfitting": probability_backtest_overfitting(
            sample, blocks=10
        ),
        "full_sample_winner": winner,
        "full_sample_winner_sharpe": float(sharpes[winner]),
    }
    symbolic_sample = sample.drop(columns=["market_ew30"])
    classical["without_positive_control"] = {
        "white_reality_check": white_reality_check(
            symbolic_sample,
            repetitions=200 if fast else 1000,
            block_length=20,
        ),
        "winner": symbolic_sample.apply(annualized_sharpe).idxmax(),
    }
    proposed_ids = [
        strategy_id
        for strategy_id in main_proposals["strategy_id"]
        if strategy_id != "market_ew30"
    ]
    proposed_sample = sample[proposed_ids]
    classical["adaptive_proposals_only"] = {
        "white_reality_check": white_reality_check(
            proposed_sample,
            repetitions=200 if fast else 1000,
            block_length=20,
        ),
        "candidates": len(proposed_ids),
    }
    _write_json(results_dir / "classical_diagnostics.json", classical)

    if not skip_simulations:
        scores = bounded_scores(main_returns).loc["2002-01-01":"2026-05-29"]
        null_results = null_calibration(
            scores,
            repetitions=25 if fast else 500,
            proposal_gap=63,
            maximum_proposals=len(main_proposals),
            level=0.10,
            inspect_every=5,
        )
        null_results.to_csv(results_dir / "null_calibration.csv", index=False)

        power_results = planted_power(
            annual_sharpes=(0.5, 1.0, 1.5, 2.0, 3.0),
            correlations=(0.0, 0.5),
            repetitions=15 if fast else 250,
        )
        power_results.to_csv(results_dir / "planted_power.csv", index=False)

    manifest = {
        "data": provenance,
        "sample_start": str(industries.index.min().date()),
        "sample_end": str(industries.index.max().date()),
        "reported_window_start": "2007-01-03",
        "reported_window_end": "2026-05-29",
        "instruments": int(industries.shape[1]),
        "candidate_strategies": int(main_returns.shape[1]),
        "proposals": int(len(main_proposals)),
        "fdr_level": 0.10,
        "new_admission_floor": "h=1/q=10 for every new certification",
        "inspection_frequency_trading_days": 5,
        "score_scale_window": 63,
        "score_scale_multiple": 4.0,
        "spending_weights": "gamma_j=1/[j(j+1)]",
        "null_repetitions": 25 if fast else 500,
        "power_repetitions_per_cell": 15 if fast else 250,
        "python": sys.version,
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "scipy", "matplotlib"]
        },
        "code_sha256": {
            path.name: sha256(path)
            for path in sorted((root / "code").glob("*.py"))
        },
        "output_sha256": {
            path.name: sha256(path)
            for path in sorted(results_dir.glob("*"))
            if path.is_file() and path.name != "run_manifest.json"
        },
        "processed_data_sha256": {
            path.name: sha256(path)
            for path in sorted(processed_dir.glob("*.csv"))
        },
        "platform": platform.platform(),
        "fast_mode": fast,
        "simulations_reused": skip_simulations,
    }
    _write_json(results_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--skip-simulations", action="store_true")
    arguments = parser.parse_args()
    run(
        arguments.root.resolve(),
        fast=arguments.fast,
        skip_simulations=arguments.skip_simulations,
    )
