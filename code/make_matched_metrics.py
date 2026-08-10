from __future__ import annotations

from pathlib import Path

import pandas as pd

from data import load_panel
from inference import first_rejection_date, run_gate, terminal_rejections
from portfolios import (
    fixed_split_portfolio,
    full_sample_topk,
    performance_metrics,
    rolling_topk_portfolio,
    safe_alpha_portfolio,
)
from strategies import adaptive_proposals, build_candidates


def _row(
    method: str,
    cost_bps: float,
    selected: int,
    returns: pd.Series,
    turnover: pd.Series,
    active_start: pd.Timestamp,
    note: str = "",
) -> dict[str, object]:
    sample_note = f" {note}" if note else ""
    return {
        "method": method,
        "cost_bps": cost_bps,
        "sample": (
            f"{active_start.date()}--{returns.index[-1].date()}{sample_note}"
        ),
        "selected": selected,
        **performance_metrics(
            returns.loc[active_start:],
            turnover.loc[active_start:],
        ),
    }


def run(root: Path) -> None:
    raw_dir = root / "data" / "raw"
    results_dir = root / "results"
    industries, _, risk_free = load_panel(raw_dir)

    reference_returns, _, _ = build_candidates(
        industries, risk_free, cost_bps=2.0
    )
    proposals = adaptive_proposals(
        reference_returns,
        start="2007-01-03",
        end="2025-12-31",
        lookback_days=1260,
        proposals_per_quarter=1,
        diversity_penalty=0.20,
    )
    reference_testing = reference_returns.loc["2002-01-01":"2026-05-29"]
    reference_gate = run_gate(
        reference_testing,
        proposals,
        level=0.10,
        inspect_every=5,
        scale_window=63,
        scale_multiple=4.0,
    )

    rows: list[dict[str, object]] = []
    for cost_bps in [2.0, 5.0, 10.0]:
        candidate_returns, candidate_turnover, _ = build_candidates(
            industries, risk_free, cost_bps=cost_bps
        )
        testing = candidate_returns.loc["2002-01-01":"2026-05-29"]
        gate = (
            reference_gate
            if cost_bps == 2.0
            else run_gate(
                testing,
                proposals,
                level=0.10,
                inspect_every=5,
                scale_window=63,
                scale_multiple=4.0,
                scale_reference=reference_testing,
                fixed_lambdas=reference_gate.lambdas,
            )
        )
        first_date = first_rejection_date(gate)
        if first_date is None:
            continue
        safe_returns, safe_turnover, _ = safe_alpha_portfolio(
            candidate_returns.loc["2007-01-03":"2026-05-29"],
            candidate_turnover.loc["2007-01-03":"2026-05-29"],
            gate,
            risk_free,
            cost_bps,
        )
        deployable = safe_returns.index[safe_returns.index > first_date]
        active_start = pd.Timestamp(
            deployable[0] if len(deployable) else safe_returns.index[-1]
        )
        rows.append(
            _row(
                "SAFE-ALPHA",
                cost_bps,
                len(terminal_rejections(gate)),
                safe_returns,
                safe_turnover,
                active_start,
            )
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
        rows.append(
            _row(
                "Ungated adaptive top-5",
                cost_bps,
                5,
                rolling_returns,
                rolling_turnover,
                active_start,
            )
        )

        split_returns, split_turnover, split_chosen = fixed_split_portfolio(
            candidate_returns,
            candidate_turnover,
            risk_free,
            cost_bps,
        )
        rows.append(
            _row(
                "Fixed train/validation/test",
                cost_bps,
                len(split_chosen),
                split_returns,
                split_turnover,
                active_start,
            )
        )

        biased_returns, biased_turnover, biased_chosen = full_sample_topk(
            candidate_returns,
            candidate_turnover,
            risk_free,
            cost_bps,
        )
        rows.append(
            _row(
                "Full-sample top-5",
                cost_bps,
                len(biased_chosen),
                biased_returns,
                biased_turnover,
                active_start,
                "(selected in sample)",
            )
        )

    pd.DataFrame(rows).to_csv(
        results_dir / "economic_metrics_matched.csv", index=False
    )


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
