from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_panel
from inference import bounded_scores
from simulation import _run_null_stream, _wilson
from strategies import build_candidates


_SCORES: np.ndarray | None = None
_START = 0


def _task(seed: int) -> dict[str, int]:
    if _SCORES is None:
        raise RuntimeError("worker scores were not initialized")
    return {
        "seed": seed,
        **asdict(
            _run_null_stream(
                _SCORES,
                start_location=_START,
                proposal_gap=63,
                maximum_proposals=77,
                level=0.10,
                inspect_every=5,
                seed=seed,
            )
        ),
    }


def run(root: Path, jobs: int) -> None:
    global _SCORES, _START
    industries, _, risk_free = load_panel(root / "data" / "raw")
    candidate_returns, _, _ = build_candidates(
        industries, risk_free, cost_bps=2.0
    )
    scores = bounded_scores(candidate_returns).loc["2002-01-01":"2026-05-29"]
    _SCORES = scores.to_numpy()
    _START = int(scores.index.searchsorted(pd.Timestamp("2007-01-03")))
    seeds = list(range(271828, 271828 + 500))
    if jobs == 1:
        outcomes = [_task(seed) for seed in seeds]
    else:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
            outcomes = list(executor.map(_task, seeds, chunksize=4))
    per_seed = pd.DataFrame(outcomes)
    results = root / "results"
    per_seed.to_csv(results / "null_calibration_floor_seed_results.csv", index=False)
    rows: list[dict[str, object]] = []
    for method, count_field, day_field in [
        ("SAFE-ALPHA", "safe_count", "safe_first_day"),
        ("Same-bar leakage", "leakage_count", "leakage_first_day"),
        ("Repeated 5% t-test", "sequential_count", "sequential_first_day"),
        ("Unadjusted one-year holdout", "holdout_count", "holdout_first_day"),
    ]:
        counts = per_seed[count_field].to_numpy()
        days = per_seed[day_field].to_numpy()
        successes = int(np.sum(counts > 0))
        low, high = _wilson(successes, len(per_seed))
        rows.append(
            {
                "method": method,
                "false_discovery_probability": successes / len(per_seed),
                "ci_low": low,
                "ci_high": high,
                "mean_false_discoveries": float(np.mean(counts)),
                "median_first_day_if_any": (
                    float(np.median(days[days >= 0]))
                    if np.any(days >= 0)
                    else np.nan
                ),
                "repetitions": len(per_seed),
                "new_admission_floor": (
                    10.0 if method in {"SAFE-ALPHA", "Same-bar leakage"} else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows)
    for name in ["null_calibration_floor.csv", "null_calibration.csv"]:
        summary.to_csv(results / name, index=False, float_format="%.12g")
    manifest = {
        "seed_range": [271828, 272327],
        "repetitions": 500,
        "q": 0.10,
        "inspection_frequency_trading_days": 5,
        "proposal_gap_trading_days": 63,
        "maximum_proposals": 77,
        "new_admission_floor": {
            "SAFE-ALPHA": 10.0,
            "Same-bar leakage": 10.0,
            "Repeated 5% t-test": None,
            "Unadjusted one-year holdout": None,
        },
        "dependence": "one Rademacher sign per date multiplies all 89 scores",
    }
    (results / "null_calibration_floor_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    arguments = parser.parse_args()
    if arguments.jobs < 1:
        raise ValueError("--jobs must be positive")
    run(arguments.root.resolve(), arguments.jobs)
