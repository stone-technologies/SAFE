from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from inference import run_gate, spending_weights, weighted_ebh  # noqa: E402
from baselines import hansen_spa  # noqa: E402
from empirical_diagnostics import predictable_factor_residuals  # noqa: E402
from portfolios import safe_alpha_portfolio  # noqa: E402
from power_upgrade import (  # noqa: E402
    campaign_spending_weights,
    run_mixture_gate,
)
from simulation import _run_common_evidence, matched_seed_blocks  # noqa: E402
from strategies import _signal_portfolio, _time_series_portfolio  # noqa: E402


class InferenceTests(unittest.TestCase):
    def test_spending_budget_is_summable(self) -> None:
        weights = spending_weights(10_000)
        self.assertTrue(np.all(weights > 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0 - 1.0 / 10_001)

    def test_fixed_cap_campaign_has_unit_infinite_mass(self) -> None:
        eta, campaign_cap = 0.65, 50
        for realized_count in (17, campaign_cap, 137):
            weights = campaign_spending_weights(
                realized_count, eta, campaign_cap
            )
            telescoping_tail = (1.0 - eta) / (realized_count + 1.0)
            unspent_campaign_mass = (
                eta * max(campaign_cap - realized_count, 0) / campaign_cap
            )
            self.assertAlmostEqual(
                float(weights.sum())
                + telescoping_tail
                + unspent_campaign_mass,
                1.0,
            )

    def test_fixed_cap_campaign_is_prefix_invariant(self) -> None:
        eta, campaign_cap = 0.65, 50
        short = campaign_spending_weights(19, eta, campaign_cap)
        long = campaign_spending_weights(91, eta, campaign_cap)
        np.testing.assert_allclose(short, long[: len(short)], rtol=0.0, atol=0.0)
        ranks = np.arange(campaign_cap + 1, len(long) + 1, dtype=float)
        expected_tail = (1.0 - eta) / (ranks * (ranks + 1.0))
        np.testing.assert_allclose(
            long[campaign_cap:], expected_tail, rtol=1e-14, atol=0.0
        )

    def test_upward_geometric_mixture_is_a_null_e_process(self) -> None:
        # Exhaust the 2^3 paths of three independent Rademacher increments.
        # Every fixed component has mean-one terminal wealth; their fixed
        # arithmetic mixture therefore does too and stays nonnegative.
        terminal_e_values = []
        for increments in product((-1.0, 1.0), repeat=3):
            values = np.array([[0.0], *([x] for x in increments)])
            outcome = run_mixture_gate(
                values=values,
                arrivals=np.array([0]),
                base_lambdas=np.array([0.1]),
                gammas=np.array([1.0]),
                bet_multipliers=(1.0, 2.0, 4.0),
                level=1e-6,
                inspect_every=1,
            )
            self.assertGreaterEqual(float(outcome.raw_e[0]), 0.0)
            self.assertAlmostEqual(
                float(outcome.safe_e[0]), float(outcome.raw_e[0])
            )
            terminal_e_values.append(float(outcome.raw_e[0]))
        self.assertAlmostEqual(float(np.mean(terminal_e_values)), 1.0)

    def test_weighted_ebh_step_up(self) -> None:
        values = np.array([25.0, 20.0, 1.0])
        weights = np.array([0.5, 0.25, 0.125])
        rejected = weighted_ebh(
            values, weights, level=0.10, admission_floor=10.0
        )
        np.testing.assert_array_equal(rejected, np.array([True, True, False]))

    def test_matched_terminal_rule_uses_telescoping_weights(self) -> None:
        # With three uniform weights, E_1=25 misses the one-rejection boundary
        # 1/(.10/3)=30.  The predeclared gamma_1=1/2 boundary is 20.
        values = np.array([25.0, 1.0, 1.0])
        telescoping = spending_weights(3)
        matched = weighted_ebh(
            values, telescoping, level=0.10, admission_floor=10.0
        )
        uniform = weighted_ebh(
            values,
            np.repeat(1.0 / 3.0, 3),
            level=0.10,
            admission_floor=10.0,
        )
        np.testing.assert_array_equal(matched, np.array([True, False, False]))
        np.testing.assert_array_equal(uniform, np.array([False, False, False]))

    def test_new_admission_floor_blocks_weak_passenger(self) -> None:
        values = np.r_[5.0, np.repeat(100.0, 9)]
        weights = np.r_[0.5, np.repeat(1.0 / 18.0, 9)]
        old = np.r_[False, np.repeat(True, 9)]
        without_floor = weighted_ebh(
            values, weights, level=0.10, previously_rejected=old
        )
        with_floor = weighted_ebh(
            values,
            weights,
            level=0.10,
            previously_rejected=old,
            admission_floor=10.0,
        )
        self.assertTrue(without_floor[0])
        self.assertFalse(with_floor[0])
        np.testing.assert_array_equal(with_floor[1:], np.repeat(True, 9))

    def test_safe_and_terminal_share_raw_path_but_only_safe_freezes(self) -> None:
        values = np.array([[0.0], [1.0], [-1.0], [-1.0]])
        comparison = _run_common_evidence(
            values=values,
            arrivals=np.array([0]),
            lambdas=np.array([0.5]),
            gammas=np.array([1.0]),
            levels=np.array([1.0]),
            inspect_every=1,
        )[0]
        self.assertTrue(comparison.safe_promoted[0])
        self.assertAlmostEqual(float(comparison.safe_terminal[0]), 1.5)
        self.assertAlmostEqual(float(comparison.raw_terminal[0]), 0.375)
        self.assertFalse(comparison.terminal_rejected[0])

    def test_matched_terminal_set_is_contained_in_safe_terminal_set(self) -> None:
        # At the common terminal inspection, SAFE has every old frozen member
        # plus the untouched raw path for every unpromoted member.  Any rank
        # feasible for the one-shot terminal rule is therefore feasible for
        # SAFE as well.  Exercise this on seeded multihypothesis paths with
        # staggered arrivals, heterogeneous bets, and the mandatory floor.
        terminal_discoveries = 0
        for seed in range(32):
            rng = np.random.default_rng(20260730 + seed)
            observations, proposals = 140, 12
            values = rng.normal(0.0, 0.45, size=(observations, proposals))
            values[:, :3] += 0.35
            values = np.clip(values, -1.0, 1.0)
            comparison = _run_common_evidence(
                values=values,
                arrivals=np.arange(proposals) * 4,
                lambdas=rng.uniform(0.10, 0.50, size=proposals),
                gammas=spending_weights(proposals),
                levels=np.array([0.10]),
                inspect_every=5,
            )[0]
            self.assertFalse(
                np.any(comparison.terminal_rejected & ~comparison.safe_promoted)
            )
            terminal_discoveries += int(comparison.terminal_rejected.sum())
        self.assertGreater(terminal_discoveries, 0)

    def test_hybrid_dense_seed_map_preserves_archived_blocks(self) -> None:
        mapping = matched_seed_blocks(repetitions=250)
        anchor = mapping[
            (mapping["correlation"] == 0.5)
            & (mapping["annual_sharpe"] == 1.5)
        ].iloc[0]
        inserted = mapping[
            (mapping["correlation"] == 0.0)
            & (mapping["annual_sharpe"] == 0.75)
        ].iloc[0]
        self.assertEqual(int(anchor.seed_first), 315909)
        self.assertEqual(int(anchor.seed_last), 316158)
        self.assertEqual(int(inserted.seed_first), 20260730)
        intervals = [
            set(range(int(row.seed_first), int(row.seed_last) + 1))
            for row in mapping.itertuples()
            if row.seed_source == "inserted_20260730_block"
        ]
        self.assertEqual(sum(map(len, intervals)), len(set().union(*intervals)))

    def test_first_update_is_strictly_postproposal_and_freezes(self) -> None:
        dates = pd.date_range("2020-01-01", periods=80, freq="B")
        returns = pd.DataFrame({"candidate": np.repeat(0.01, len(dates))}, index=dates)
        proposals = pd.DataFrame(
            {
                "strategy_id": ["candidate"],
                "proposal_date": [dates[29]],
            }
        )
        result = run_gate(
            returns,
            proposals,
            level=0.50,
            inspect_every=1,
            scale_window=20,
            scale_multiple=1.0,
            fixed_lambdas=pd.Series({"candidate": 0.5}),
        )
        self.assertEqual(float(result.e_values.loc[dates[29], "candidate"]), 1.0)
        self.assertGreater(float(result.e_values.loc[dates[30], "candidate"]), 1.0)
        date = result.promotion_dates["candidate"]
        self.assertFalse(pd.isna(date))
        frozen_path = result.e_values.loc[date:, "candidate"]
        self.assertTrue(np.allclose(frozen_path, frozen_path.iloc[0]))

    def test_future_candidate_is_never_promoted(self) -> None:
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        returns = pd.DataFrame(
            {"early": np.ones(30), "future": np.ones(30)}, index=dates
        )
        proposals = pd.DataFrame(
            {
                "strategy_id": ["early", "future"],
                "proposal_date": [dates[0], dates[-1]],
            }
        )
        result = run_gate(
            returns,
            proposals,
            level=1.0,
            inspect_every=1,
            scale_window=20,
            scale_multiple=1.0,
            fixed_lambdas=pd.Series({"early": 0.5, "future": 0.5}),
        )
        before_arrival = result.rejection_sets.get(dates[-2], ())
        self.assertNotIn("future", before_arrival)

    def test_predictable_factor_residual_excludes_current_fit(self) -> None:
        rng = np.random.default_rng(123)
        dates = pd.date_range("2020-01-01", periods=80, freq="B")
        factors = pd.DataFrame(
            rng.normal(size=(80, 6)),
            index=dates,
            columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"],
        )
        coefficients = np.array([0.8, -0.2, 0.1, 0.3, -0.1, 0.25])
        returns = pd.DataFrame(
            {"candidate": 0.002 + factors.to_numpy() @ coefficients},
            index=dates,
        )
        # A current-date shock must affect the residual but not the beta used
        # to form it; the beta is fit only to the preceding 40 rows.
        returns.iloc[60, 0] += 1.0
        residual = predictable_factor_residuals(
            returns, factors, lookback=40, minimum=20
        )
        history_x = np.column_stack(
            [np.ones(40), factors.iloc[20:60].to_numpy()]
        )
        history_y = returns.iloc[20:60, 0].to_numpy()
        fitted = np.linalg.lstsq(history_x, history_y, rcond=None)[0]
        expected = (
            returns.iloc[60, 0]
            - factors.iloc[60].to_numpy() @ fitted[1:]
        )
        self.assertAlmostEqual(float(residual.iloc[60, 0]), float(expected))


class ExecutionTests(unittest.TestCase):
    def test_signal_is_lagged_one_bar(self) -> None:
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        returns = pd.DataFrame(
            [[0.10, -0.10], [0.20, -0.20], [0.30, -0.30]],
            index=dates,
            columns=["a", "b"],
        )
        signal = pd.DataFrame(
            [[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]],
            index=dates,
            columns=returns.columns,
        )
        realized, _ = _signal_portfolio(
            returns,
            pd.Series(0.0, index=dates),
            signal,
            breadth=1,
            direction=1,
            rebalance="daily",
            cost_bps=0.0,
        )
        self.assertEqual(float(realized.iloc[0]), 0.0)
        self.assertAlmostEqual(float(realized.iloc[1]), 0.20)
        self.assertAlmostEqual(float(realized.iloc[2]), -0.30)

    def test_directional_sleeve_is_excess_return(self) -> None:
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        risk_free = pd.Series(0.001, index=dates)
        returns = pd.DataFrame(0.001, index=dates, columns=["a", "b"])
        signal = pd.DataFrame(1.0, index=dates, columns=returns.columns)
        realized, _ = _time_series_portfolio(
            returns,
            risk_free,
            signal,
            direction=1,
            rebalance="daily",
            cost_bps=0.0,
        )
        self.assertTrue(np.allclose(realized, 0.0))

    def test_certified_portfolio_starts_next_bar_and_costs_allocation(self) -> None:
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        candidate_returns = pd.DataFrame(
            {"candidate": [0.50, 0.10, 0.0]}, index=dates
        )
        candidate_turnover = pd.DataFrame(
            {"candidate": [0.0, 0.0, 0.0]}, index=dates
        )
        gate = type("Gate", (), {})()
        gate.promotion_dates = pd.Series({"candidate": dates[0]})
        returns, turnover, weights = safe_alpha_portfolio(
            candidate_returns,
            candidate_turnover,
            gate,
            pd.Series(0.0, index=dates),
            cost_bps=10.0,
        )
        self.assertEqual(float(weights.loc[dates[0], "candidate"]), 0.0)
        self.assertEqual(float(returns.loc[dates[0]]), 0.0)
        self.assertAlmostEqual(float(turnover.loc[dates[1]]), 0.5)
        self.assertAlmostEqual(float(returns.loc[dates[1]]), 0.10 - 0.0005)


class BaselineTests(unittest.TestCase):
    def test_spa_is_deterministic_and_recenterings_are_ordered(self) -> None:
        rng = np.random.default_rng(321)
        returns = pd.DataFrame(rng.normal(size=(120, 4)))
        first = hansen_spa(
            returns, repetitions=99, mean_block_length=10.0, seed=20260730
        )
        second = hansen_spa(
            returns, repetitions=99, mean_block_length=10.0, seed=20260730
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first["p_value_lower"], first["p_value_consistent"])
        self.assertLessEqual(first["p_value_consistent"], first["p_value_upper"])


if __name__ == "__main__":
    unittest.main()
