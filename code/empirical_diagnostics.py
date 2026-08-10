from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import zeta
from scipy.stats import t as student_t

from baselines import hansen_spa
from data import load_panel
from inference import (
    GateResult,
    _preproposal_lambda,
    bounded_scores,
    run_gate,
    spending_weights,
)
from strategies import annualized_sharpe, build_candidates


FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _hac_regression(
    dependent: pd.Series,
    factors: pd.DataFrame,
    lags: int = 5,
) -> dict[str, object]:
    frame = pd.concat([dependent.rename("dependent"), factors], axis=1).dropna()
    y = frame.pop("dependent").to_numpy(dtype=float)
    names = frame.columns.tolist()
    x = np.column_stack([np.ones(len(frame)), frame.to_numpy(dtype=float)])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coefficients
    bread = np.linalg.inv(x.T @ x)
    scores = x * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / float(lags + 1)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    covariance = bread @ meat @ bread
    covariance *= len(y) / max(len(y) - x.shape[1], 1)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    alpha_t = coefficients[0] / max(standard_errors[0], 1e-15)
    alpha_p = 2.0 * student_t.sf(abs(alpha_t), max(len(y) - x.shape[1], 1))
    centered = y - y.mean()
    r_squared = 1.0 - float(residual @ residual) / max(float(centered @ centered), 1e-15)
    output: dict[str, object] = {
        "sample_start": str(frame.index.min().date()),
        "sample_end": str(frame.index.max().date()),
        "observations": len(frame),
        "hac_lags": lags,
        "alpha_daily": float(coefficients[0]),
        "alpha_annualized": float(252.0 * coefficients[0]),
        "alpha_hac_se_daily": float(standard_errors[0]),
        "alpha_hac_t": float(alpha_t),
        "alpha_hac_p_two_sided": float(alpha_p),
        "r_squared": r_squared,
    }
    for name, value, standard_error in zip(
        names, coefficients[1:], standard_errors[1:]
    ):
        key = name.lower().replace("-", "_")
        output[f"beta_{key}"] = float(value)
        output[f"beta_{key}_hac_se"] = float(standard_error)
    return output


def predictable_factor_residuals(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    lookback: int = 1260,
    minimum: int = 252,
) -> pd.DataFrame:
    """Subtract only beta[t-1]'f[t], retaining the fitted intercept.

    The rolling OLS coefficient for date t uses rows max(0,t-lookback),...,t-1.
    This implementation requires the aligned diagnostic panel to be complete;
    the archived 2002--2026 candidate/factor panel satisfies that condition.
    """
    aligned_factors = factors.reindex(returns.index)[FACTOR_COLUMNS]
    if returns.isna().any().any() or aligned_factors.isna().any().any():
        raise ValueError("predictable residual diagnostic requires a complete panel")
    y = returns.to_numpy(dtype=float)
    f = aligned_factors.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(f)), f])
    xx = np.einsum("ni,nj->nij", design, design)
    xy = np.einsum("ni,nj->nij", design, y)
    cumulative_xx = np.concatenate(
        [np.zeros((1, design.shape[1], design.shape[1])), np.cumsum(xx, axis=0)]
    )
    cumulative_xy = np.concatenate(
        [np.zeros((1, design.shape[1], y.shape[1])), np.cumsum(xy, axis=0)]
    )
    residual = np.full_like(y, np.nan)
    for t in range(minimum, len(returns)):
        start = max(0, t - lookback)
        if t - start < minimum:
            continue
        cross_xx = cumulative_xx[t] - cumulative_xx[start]
        cross_xy = cumulative_xy[t] - cumulative_xy[start]
        coefficients = np.linalg.lstsq(cross_xx, cross_xy, rcond=None)[0]
        residual[t] = y[t] - f[t] @ coefficients[1:]
    return pd.DataFrame(residual, index=returns.index, columns=returns.columns)


def _gate_ledger(gate: GateResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_id": gate.promotion_dates.index,
            "promotion_date": gate.promotion_dates.values,
            "frozen_e_value": [
                gate.e_values.loc[date, strategy]
                if pd.notna(date)
                else np.nan
                for strategy, date in gate.promotion_dates.items()
            ],
            "gamma": gate.gammas.values,
            "lambda": gate.lambdas.values,
            "terminal_stopped_e_value": gate.e_values.iloc[-1].values,
            "maximum_stopped_e_value": gate.e_values.max(axis=0).values,
        }
    )


def _factor_alpha_diagnostics(
    candidate_returns: pd.DataFrame,
    factors: pd.DataFrame,
    results_dir: Path,
) -> pd.DataFrame:
    promotion_date = pd.Timestamp("2018-01-19")
    deployment = candidate_returns.index[candidate_returns.index > promotion_date][0]
    dependent = candidate_returns.loc[deployment:"2026-05-29", "market_ew30"]
    rows = []
    for model, columns in [
        ("CAPM", ["Mkt-RF"]),
        ("FF5+Mom", FACTOR_COLUMNS),
    ]:
        rows.append(
            {
                "model": model,
                "dependent_variable": "2-bp net excess return of market_ew30",
                **_hac_regression(dependent, factors[columns], lags=5),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(
        results_dir / "factor_regression_diagnostics.csv",
        index=False,
        float_format="%.12g",
    )
    return output


def _winner_curse_diagnostics(
    candidate_returns: pd.DataFrame,
    proposals: pd.DataFrame,
    results_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    symbolic = proposals[proposals["role"] == "adaptive_symbolic_proposal"]
    for proposal in symbolic.itertuples(index=False):
        future = candidate_returns.loc[
            candidate_returns.index > pd.Timestamp(proposal.proposal_date),
            proposal.strategy_id,
        ].iloc[:252]
        complete = len(future) == 252
        post_sharpe = annualized_sharpe(future) if complete else np.nan
        rows.append(
            {
                "proposal_index": proposal.proposal_index,
                "strategy_id": proposal.strategy_id,
                "proposal_date": proposal.proposal_date,
                "preproposal_1260d_sharpe": proposal.trailing_sharpe,
                "postproposal_days": len(future),
                "complete_252d_holdout": complete,
                "postproposal_252d_sharpe": post_sharpe,
                "optimism_gap": (
                    proposal.trailing_sharpe - post_sharpe
                    if np.isfinite(post_sharpe)
                    else np.nan
                ),
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(
        results_dir / "winner_curse_by_proposal.csv",
        index=False,
        float_format="%.12g",
    )
    complete = detail[detail["complete_252d_holdout"]].copy()
    pre = complete["preproposal_1260d_sharpe"]
    post = complete["postproposal_252d_sharpe"]
    gaps = (pre - post).to_numpy(dtype=float)
    rng = np.random.default_rng(20260730)
    bootstrap_medians = np.median(
        gaps[rng.integers(0, len(gaps), size=(10_000, len(gaps)))], axis=1
    )
    summary: dict[str, object] = {
        "symbolic_proposals": len(detail),
        "complete_252_day_holdouts": len(complete),
        "score_definition": "2-bp net excess candidate return",
        "preproposal_window_days": 1260,
        "strict_postproposal_window_days": 252,
        "mean_preproposal_sharpe": float(pre.mean()),
        "median_preproposal_sharpe": float(pre.median()),
        "mean_postproposal_sharpe": float(post.mean()),
        "median_postproposal_sharpe": float(post.median()),
        "mean_optimism_gap": float((pre - post).mean()),
        "median_optimism_gap": float((pre - post).median()),
        "median_optimism_gap_bootstrap_ci_low": float(
            np.quantile(bootstrap_medians, 0.025)
        ),
        "median_optimism_gap_bootstrap_ci_high": float(
            np.quantile(bootstrap_medians, 0.975)
        ),
        "median_bootstrap_repetitions": 10_000,
        "median_bootstrap_seed": 20260730,
        "fraction_post_below_pre": float((post < pre).mean()),
        "fraction_post_nonpositive": float((post <= 0.0).mean()),
        "pre_post_pearson_correlation": float(pre.corr(post)),
    }
    _write_json(results_dir / "winner_curse_summary.json", summary)
    return detail, summary


def _predictable_residual_gate(
    candidate_returns: pd.DataFrame,
    factors: pd.DataFrame,
    proposals: pd.DataFrame,
    results_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    testing = candidate_returns.loc["2002-01-01":"2026-05-29"]
    residual = predictable_factor_residuals(
        testing, factors, lookback=1260, minimum=252
    )
    gate = run_gate(
        residual,
        proposals,
        level=0.10,
        inspect_every=5,
        scale_window=63,
        scale_multiple=4.0,
    )
    ledger = _gate_ledger(gate)
    ledger.to_csv(
        results_dir / "predictable_residual_gate_ledger.csv",
        index=False,
        float_format="%.12g",
    )
    promoted = ledger[ledger["promotion_date"].notna()]
    summary: dict[str, object] = {
        "diagnostic_label": "predictable FF5+Mom residual-gate robustness check",
        "factor_beta_lookback_days": 1260,
        "minimum_beta_observations": 252,
        "beta_timing": "beta for date t is estimated only through t-1",
        "intercept_treatment": "retained; subtract beta[t-1]' factor[t] only",
        "proposal_ledger": "identical archived 77 proposals and dates",
        "q": 0.10,
        "new_admission_floor": 10.0,
        "inspection_days": 5,
        "score_scale_window": 63,
        "score_scale_multiple": 4.0,
        "proposal_weights": "gamma_j=1/[j(j+1)]",
        "bet_policy": "proposal-time fixed lambda re-estimated from predictable residual-score history",
        "certifications": len(promoted),
        "certified_strategies": promoted["strategy_id"].tolist(),
        "certification_dates": [
            str(pd.Timestamp(date).date()) for date in promoted["promotion_date"]
        ],
    }
    _write_json(results_dir / "predictable_residual_gate_summary.json", summary)
    return ledger, summary


def _summarize_gate(
    factor: str,
    level: str,
    gate: GateResult,
) -> dict[str, object]:
    ledger = _gate_ledger(gate)
    promoted = ledger[ledger["promotion_date"].notna()]
    market = ledger[ledger["strategy_id"] == "market_ew30"].iloc[0]
    return {
        "varied_factor": factor,
        "level": level,
        "certifications": len(promoted),
        "certified_strategies": "|".join(promoted["strategy_id"]),
        "certification_dates": "|".join(
            f"{row.strategy_id}:{pd.Timestamp(row.promotion_date).date()}"
            for row in promoted.itertuples(index=False)
        ),
        "first_certification_date": (
            str(pd.to_datetime(promoted["promotion_date"]).min().date())
            if len(promoted)
            else ""
        ),
        "market_control_certified": bool(pd.notna(market.promotion_date)),
        "market_control_date": (
            str(pd.Timestamp(market.promotion_date).date())
            if pd.notna(market.promotion_date)
            else ""
        ),
        "market_control_frozen_e": market.frozen_e_value,
        "market_control_gamma": market.gamma,
    }


def _scale_motivation_diagnostics(
    candidate_returns: pd.DataFrame,
    proposals: pd.DataFrame,
    results_dir: Path,
) -> pd.DataFrame:
    """Quantify score clipping and proposal-time bet caps by scale choice.

    Clipping is measured over every finite score in the full 89-rule public
    diagnostic panel.  Bet-cap incidence is measured over the 77 archived
    proposal-time lambdas, using each proposal's own predictable history.
    """
    testing = candidate_returns.loc["2002-01-01":"2026-05-29"]
    rolling = testing.rolling(63, min_periods=20).std().shift(1)
    expanding = testing.expanding(min_periods=20).std().shift(1)
    predictable_scale = rolling.fillna(expanding).clip(lower=1e-6)
    proposal_ids = proposals["strategy_id"].tolist()
    proposal_dates = pd.to_datetime(proposals["proposal_date"]).tolist()
    rows: list[dict[str, object]] = []
    for multiple in [2.0, 4.0, 8.0]:
        unbounded = testing / (multiple * predictable_scale)
        values = unbounded.to_numpy(dtype=float)
        finite = np.isfinite(values)
        clipped = finite & (np.abs(values) > 1.0)
        proposal_scores = bounded_scores(
            testing[proposal_ids],
            scale_window=63,
            scale_multiple=multiple,
        )
        lambdas = np.array(
            [
                _preproposal_lambda(proposal_scores[strategy_id], proposal_date)
                for strategy_id, proposal_date in zip(
                    proposal_ids, proposal_dates
                )
            ],
            dtype=float,
        )
        at_cap = np.isclose(lambdas, 0.50, rtol=0.0, atol=1e-15)
        rows.append(
            {
                "score_scale_multiple": multiple,
                "score_panel_candidates": testing.shape[1],
                "finite_score_count": int(finite.sum()),
                "clipped_score_count": int(clipped.sum()),
                "finite_score_clipping_fraction": float(
                    clipped.sum() / finite.sum()
                ),
                "proposal_count": len(lambdas),
                "lambda_upper_cap": 0.50,
                "lambda_at_upper_cap_count": int(at_cap.sum()),
                "proposal_lambda_upper_cap_fraction": float(at_cap.mean()),
                "proposal_lambda_median": float(np.median(lambdas)),
                "proposal_lambda_minimum": float(np.min(lambdas)),
                "proposal_lambda_maximum": float(np.max(lambdas)),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(
        results_dir / "scale_motivation_diagnostics.csv",
        index=False,
        float_format="%.12g",
    )
    return output


def _public_design_sensitivity(
    candidate_returns: pd.DataFrame,
    proposals: pd.DataFrame,
    results_dir: Path,
) -> pd.DataFrame:
    testing = candidate_returns.loc["2002-01-01":"2026-05-29"]
    primary = run_gate(
        testing,
        proposals,
        level=0.10,
        inspect_every=5,
        scale_window=63,
        scale_multiple=4.0,
    )
    rows = [_summarize_gate("primary", "q=.10; cadence=5; scale=4; telescope; rank=1", primary)]

    for level in [0.05, 0.20]:
        gate = run_gate(
            testing,
            proposals,
            level=level,
            inspect_every=5,
            scale_window=63,
            scale_multiple=4.0,
            fixed_lambdas=primary.lambdas,
        )
        rows.append(_summarize_gate("q", f"{level:g}", gate))
    for cadence in [1, 21]:
        gate = run_gate(
            testing,
            proposals,
            level=0.10,
            inspect_every=cadence,
            scale_window=63,
            scale_multiple=4.0,
            fixed_lambdas=primary.lambdas,
        )
        rows.append(_summarize_gate("inspection_cadence_days", str(cadence), gate))
    for multiple in [2.0, 8.0]:
        gate = run_gate(
            testing,
            proposals,
            level=0.10,
            inspect_every=5,
            scale_window=63,
            scale_multiple=multiple,
        )
        rows.append(_summarize_gate("score_scale_multiple", f"{multiple:g}", gate))
    indices = np.arange(1, len(proposals) + 1, dtype=float)
    for exponent in [1.5, 2.0]:
        gammas = indices ** (-exponent) / float(zeta(exponent, 1.0))
        gate = run_gate(
            testing,
            proposals,
            level=0.10,
            inspect_every=5,
            scale_window=63,
            scale_multiple=4.0,
            fixed_lambdas=primary.lambdas,
            fixed_gammas=gammas,
        )
        rows.append(
            _summarize_gate(
                "infinite_budget_schedule",
                f"gamma_j=j^-{exponent:g}/zeta({exponent:g})",
                gate,
            )
        )
    for rank in [5, 10, 20]:
        control = proposals[proposals["strategy_id"] == "market_ew30"]
        symbolic = proposals[proposals["strategy_id"] != "market_ew30"]
        reordered = pd.concat(
            [symbolic.iloc[: rank - 1], control, symbolic.iloc[rank - 1 :]],
            ignore_index=True,
        )
        gate = run_gate(
            testing,
            reordered,
            level=0.10,
            inspect_every=5,
            scale_window=63,
            scale_multiple=4.0,
            fixed_lambdas=primary.lambdas,
        )
        rows.append(_summarize_gate("market_control_budget_rank", str(rank), gate))
    output = pd.DataFrame(rows)
    output.to_csv(
        results_dir / "public_design_sensitivity.csv",
        index=False,
        float_format="%.12g",
    )
    return output


def _spa_diagnostics(
    candidate_returns: pd.DataFrame,
    proposals: pd.DataFrame,
    results_dir: Path,
) -> dict[str, object]:
    sample = candidate_returns.loc["2007-01-03":"2026-05-29"].dropna()
    symbolic_ids = proposals.loc[
        proposals["role"] == "adaptive_symbolic_proposal", "strategy_id"
    ].tolist()
    output = {
        "full_89_rule_panel": hansen_spa(
            sample,
            repetitions=1999,
            mean_block_length=20.0,
            seed=20260730,
        ),
        "without_market_positive_control": hansen_spa(
            sample.drop(columns=["market_ew30"]),
            repetitions=1999,
            mean_block_length=20.0,
            seed=20260730,
        ),
        "adaptive_symbolic_proposals_only": hansen_spa(
            sample[symbolic_ids],
            repetitions=1999,
            mean_block_length=20.0,
            seed=20260730,
        ),
        "convention": (
            "Studentized Hansen SPA against zero; Politis-Romano stationary "
            "bootstrap; lower/consistent/upper sample-dependent recentering"
        ),
    }
    _write_json(results_dir / "spa_diagnostics.json", output)
    return output


def run(root: Path) -> None:
    results_dir = root / "results"
    industries, factors, risk_free = load_panel(root / "data" / "raw")
    candidate_returns, _, _ = build_candidates(
        industries, risk_free, cost_bps=2.0
    )
    proposals = pd.read_csv(
        results_dir / "proposal_ledger.csv", parse_dates=["proposal_date"]
    )
    factor_results = _factor_alpha_diagnostics(
        candidate_returns, factors, results_dir
    )
    _, winner_summary = _winner_curse_diagnostics(
        candidate_returns, proposals, results_dir
    )
    _, residual_summary = _predictable_residual_gate(
        candidate_returns, factors, proposals, results_dir
    )
    scale_motivation = _scale_motivation_diagnostics(
        candidate_returns, proposals, results_dir
    )
    sensitivity = _public_design_sensitivity(
        candidate_returns, proposals, results_dir
    )
    spa = _spa_diagnostics(candidate_returns, proposals, results_dir)
    output_names = [
        "factor_regression_diagnostics.csv",
        "winner_curse_by_proposal.csv",
        "winner_curse_summary.json",
        "predictable_residual_gate_ledger.csv",
        "predictable_residual_gate_summary.json",
        "scale_motivation_diagnostics.csv",
        "public_design_sensitivity.csv",
        "spa_diagnostics.json",
    ]
    manifest = {
        "factor_regressions": {
            "deployment_start": "first trading day after 2018-01-19 certification",
            "dependent_variable": "2-bp net excess return of market_ew30",
            "models": ["CAPM", "FF5+Mom"],
            "hac_lags": 5,
            "rows": len(factor_results),
        },
        "winner_curse": winner_summary,
        "predictable_residual_gate": residual_summary,
        "scale_motivation": {
            "definition": (
                "clipping over all finite scores in the full 89-rule panel; "
                "upper-cap incidence and median over 77 proposal-time lambdas"
            ),
            "rows": scale_motivation.to_dict(orient="records"),
        },
        "public_design_sensitivity_rows": len(sensitivity),
        "spa": spa,
        "code_sha256": _sha256(root / "code" / "empirical_diagnostics.py"),
        "output_sha256": {
            name: _sha256(results_dir / name) for name in output_names
        },
    }
    _write_json(results_dir / "empirical_diagnostics_manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    arguments = parser.parse_args()
    run(arguments.root.resolve())
