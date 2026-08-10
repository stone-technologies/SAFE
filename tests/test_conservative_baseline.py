from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from conservative_baseline import (  # noqa: E402
    proposal_alpha_spending,
    run_joint_and_alpha_spending,
)
from power_upgrade import run_mixture_gate  # noqa: E402


class ConservativeBaselineTests(unittest.TestCase):
    def test_proposal_thresholds_are_exact(self) -> None:
        rejected = proposal_alpha_spending(
            np.array([20.0, 59.999, 60.0]),
            np.array([0.5, 1.0 / 6.0, 1.0 / 6.0]),
            0.10,
        )
        np.testing.assert_array_equal(rejected, [True, False, True])

    def test_budget_above_one_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            proposal_alpha_spending(
                np.array([10.0, 10.0]), np.array([0.6, 0.5]), 0.10
            )

    def test_paired_joint_copy_matches_campaign_implementation(self) -> None:
        rng = np.random.default_rng(20260809)
        values = np.clip(rng.normal(0.035, 0.20, size=(360, 7)), -1.0, 1.0)
        arrivals = np.array([20, 35, 50, 65, 80, 95, 110])
        lambdas = np.linspace(0.08, 0.28, len(arrivals))
        gammas = 1.0 / (
            np.arange(1, len(arrivals) + 1)
            * np.arange(2, len(arrivals) + 2)
        )
        comparison = run_joint_and_alpha_spending(
            values,
            arrivals,
            lambdas,
            gammas,
            bet_multipliers=(1.0,),
            level=0.10,
            inspect_every=1,
        )
        reference = run_mixture_gate(
            values,
            arrivals,
            lambdas,
            gammas,
            bet_multipliers=(1.0,),
            level=0.10,
            inspect_every=1,
        )
        np.testing.assert_array_equal(comparison.joint.promoted, reference.promoted)
        np.testing.assert_array_equal(
            comparison.joint.promotion_day, reference.promotion_day
        )
        np.testing.assert_allclose(comparison.joint.stopped_e, reference.safe_e)
        np.testing.assert_allclose(comparison.raw_terminal_e, reference.raw_e)

    def test_alpha_spending_is_contained_in_joint_gate(self) -> None:
        # Two strong alternatives make the joint k-dependent threshold useful.
        values = np.zeros((180, 3), dtype=float)
        values[21:, 0] = 0.45
        values[21:, 1] = 0.40
        arrivals = np.array([20, 20, 20])
        lambdas = np.repeat(0.5, 3)
        gammas = np.array([0.30, 0.30, 0.30])
        comparison = run_joint_and_alpha_spending(
            values, arrivals, lambdas, gammas, inspect_every=1
        )
        self.assertTrue(
            np.all(
                ~comparison.alpha_spending.promoted
                | comparison.joint.promoted
            )
        )

    def test_future_candidate_is_ineligible(self) -> None:
        values = np.ones((40, 2), dtype=float)
        comparison = run_joint_and_alpha_spending(
            values,
            arrivals=np.array([0, 100]),
            base_lambdas=np.array([0.5, 0.5]),
            gammas=np.array([0.5, 0.25]),
            inspect_every=1,
        )
        self.assertFalse(comparison.alpha_spending.promoted[1])
        self.assertEqual(comparison.alpha_spending.stopped_e[1], 1.0)


if __name__ == "__main__":
    unittest.main()
