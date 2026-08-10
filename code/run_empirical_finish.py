from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_panel
from inference import bounded_scores
from simulation import (
    Q_SENSITIVITY_LEVELS,
    _heavy_tailed_null_repetition,
    _null_level_repetition,
    _planted_level_repetition,
    _planted_repetition,
    _wilson,
    matched_seed_blocks,
)
from strategies import build_candidates


_GLOBAL_NULL_SCORES: np.ndarray | None = None
_GLOBAL_NULL_START = 0


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


def _planted_task(task: tuple[float, float, int, str]) -> dict[str, object]:
    annual_sharpe, correlation, seed, source = task
    outcome = _planted_repetition(
        annual_sharpe=annual_sharpe,
        correlation=correlation,
        seed=seed,
    )
    return {
        "annual_sharpe": annual_sharpe,
        "correlation": correlation,
        "seed": seed,
        "seed_source": source,
        **outcome,
    }


def _planted_q_task(seed: int) -> list[dict[str, object]]:
    rows = _planted_level_repetition(
        annual_sharpe=1.5,
        correlation=0.5,
        seed=seed,
        levels=Q_SENSITIVITY_LEVELS,
    )
    for row in rows:
        row["seed"] = seed
    return rows


def _null_q_task(seed: int) -> list[dict[str, object]]:
    if _GLOBAL_NULL_SCORES is None:
        raise RuntimeError("null-score worker was not initialized")
    rows = _null_level_repetition(
        _GLOBAL_NULL_SCORES,
        start_location=_GLOBAL_NULL_START,
        proposal_gap=63,
        maximum_proposals=77,
        levels=Q_SENSITIVITY_LEVELS,
        inspect_every=5,
        seed=seed,
    )
    for row in rows:
        row["seed"] = seed
    return rows


def _heavy_task(seed: int) -> dict[str, int]:
    return {"seed": seed, **_heavy_tailed_null_repetition(seed)}


def _parallel_map(function, tasks: list, jobs: int, chunksize: int = 8) -> list:
    if jobs == 1:
        return [function(task) for task in tasks]
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
        return list(executor.map(function, tasks, chunksize=chunksize))


def _dense_power(results_dir: Path, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_map = matched_seed_blocks(repetitions=250)
    seed_map.to_csv(results_dir / "matched_timing_seed_map.csv", index=False)
    tasks: list[tuple[float, float, int, str]] = []
    for row in seed_map.itertuples(index=False):
        tasks.extend(
            (
                float(row.annual_sharpe),
                float(row.correlation),
                seed,
                str(row.seed_source),
            )
            for seed in range(int(row.seed_first), int(row.seed_last) + 1)
        )
    raw = pd.DataFrame(_parallel_map(_planted_task, tasks, jobs, chunksize=10))
    per_seed_rows: list[dict[str, object]] = []
    for row in raw.itertuples(index=False):
        shared = {
            "annual_sharpe": row.annual_sharpe,
            "correlation": row.correlation,
            "seed": row.seed,
            "seed_source": row.seed_source,
            "proposal_recall": row.proposal_recall,
        }
        for method, prefix in [
            ("SAFE-ALPHA", "safe"),
            ("Matched terminal weighted e-BH", "terminal_ebh"),
        ]:
            per_seed_rows.append(
                {
                    **shared,
                    "method": method,
                    "fdp": getattr(row, f"{prefix}_fdp"),
                    "conditional_power": getattr(row, f"{prefix}_power"),
                    "end_to_end_power": getattr(
                        row, f"{prefix}_end_to_end_power"
                    ),
                    "discoveries": getattr(row, f"{prefix}_discoveries"),
                    "delay_days": getattr(row, f"{prefix}_delay"),
                }
            )
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(
        results_dir / "matched_timing_seed_results.csv",
        index=False,
        float_format="%.12g",
    )
    grouped = per_seed.groupby(
        ["method", "annual_sharpe", "correlation"], sort=False
    )
    summary = grouped.agg(
        mean_fdp=("fdp", "mean"),
        mean_conditional_power=("conditional_power", "mean"),
        mean_end_to_end_power=("end_to_end_power", "mean"),
        mean_proposal_recall=("proposal_recall", "mean"),
        mean_discoveries=("discoveries", "mean"),
        mean_delay_days=("delay_days", "mean"),
        repetitions=("seed", "nunique"),
    ).reset_index()
    summary.to_csv(
        results_dir / "matched_timing_power.csv",
        index=False,
        float_format="%.12g",
    )
    return per_seed, summary


def _anchor_audit(results_dir: Path, summary: pd.DataFrame) -> pd.DataFrame:
    archived = pd.read_csv(results_dir / "planted_power.csv")
    archived = archived[archived["method"] == "SAFE-ALPHA"].copy()
    current = summary[summary["method"] == "SAFE-ALPHA"].copy()
    renamed = {
        "mean_power": "mean_conditional_power",
        "mean_end_to_end_power": "mean_end_to_end_power",
        "mean_proposal_recall": "mean_proposal_recall",
        "mean_discoveries": "mean_discoveries",
        "mean_delay_days": "mean_delay_days",
        "mean_fdp": "mean_fdp",
    }
    archived = archived.rename(
        columns={name: f"archived_{name}" for name in renamed}
    )
    current = current.rename(
        columns={name: f"matched_{name}" for name in renamed.values()}
    )
    merged = archived.merge(current, on=["annual_sharpe", "correlation"])
    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        for archived_name, current_name in renamed.items():
            old = float(getattr(row, f"archived_{archived_name}"))
            new = float(getattr(row, f"matched_{current_name}"))
            rows.append(
                {
                    "annual_sharpe": row.annual_sharpe,
                    "correlation": row.correlation,
                    "metric": archived_name,
                    "archived_value": old,
                    "matched_run_value": new,
                    "absolute_difference": abs(old - new),
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(
        results_dir / "matched_timing_anchor_audit.csv",
        index=False,
        float_format="%.17g",
    )
    if len(audit) != 60:
        raise AssertionError("archived planted SAFE anchor audit is incomplete")
    return audit


def _q_sensitivity(root: Path, results_dir: Path, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    global _GLOBAL_NULL_SCORES, _GLOBAL_NULL_START

    industries, _, risk_free = load_panel(root / "data" / "raw")
    candidate_returns, _, _ = build_candidates(
        industries, risk_free, cost_bps=2.0
    )
    scores = bounded_scores(candidate_returns).loc["2002-01-01":"2026-05-29"]
    _GLOBAL_NULL_SCORES = scores.to_numpy()
    _GLOBAL_NULL_START = int(scores.index.searchsorted(pd.Timestamp("2007-01-03")))
    null_nested = _parallel_map(
        _null_q_task,
        list(range(271828, 271828 + 500)),
        jobs,
        chunksize=4,
    )
    null_seed = pd.DataFrame([row for rows in null_nested for row in rows])
    null_seed.to_csv(
        results_dir / "q_sensitivity_null_seed_results.csv",
        index=False,
    )
    null_rows: list[dict[str, object]] = []
    for level, group in null_seed.groupby("q", sort=True):
        successes = int(group["any_false"].sum())
        low, high = _wilson(successes, len(group))
        null_rows.append(
            {
                "q": level,
                "method": "SAFE-ALPHA",
                "false_discovery_probability": successes / len(group),
                "ci_low": low,
                "ci_high": high,
                "mean_false_discoveries": group["false_discoveries"].mean(),
                "repetitions": group["seed"].nunique(),
            }
        )
    null_summary = pd.DataFrame(null_rows)
    null_summary.to_csv(
        results_dir / "q_sensitivity_null.csv",
        index=False,
        float_format="%.12g",
    )
    planted_nested = _parallel_map(
        _planted_q_task,
        list(range(315909, 316159)),
        jobs,
        chunksize=5,
    )
    planted_seed = pd.DataFrame([row for rows in planted_nested for row in rows])
    planted_seed.to_csv(
        results_dir / "q_sensitivity_power_seed_results.csv",
        index=False,
        float_format="%.12g",
    )
    planted_summary = (
        planted_seed.groupby(["q", "method"], sort=False)
        .agg(
            mean_fdp=("fdp", "mean"),
            mean_end_to_end_power=("end_to_end_power", "mean"),
            mean_discoveries=("discoveries", "mean"),
            mean_delay_days=("delay_days", "mean"),
            repetitions=("seed", "nunique"),
        )
        .reset_index()
    )
    planted_summary.insert(1, "annual_sharpe", 1.5)
    planted_summary.insert(2, "correlation", 0.5)
    planted_summary.to_csv(
        results_dir / "q_sensitivity_power.csv",
        index=False,
        float_format="%.12g",
    )
    return null_summary, planted_summary


def _heavy_null(results_dir: Path, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = pd.DataFrame(
        _parallel_map(
            _heavy_task,
            list(range(20260730, 20260730 + 500)),
            jobs,
            chunksize=5,
        )
    )
    per_seed.to_csv(results_dir / "heavy_tailed_null_seed_results.csv", index=False)
    rows: list[dict[str, object]] = []
    for method, count_field, day_field in [
        ("SAFE-ALPHA", "safe_count", "safe_first_day"),
        ("Same-bar leakage", "leakage_count", "leakage_first_day"),
        ("Repeated 5% t-test", "sequential_count", "sequential_first_day"),
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
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        results_dir / "heavy_tailed_null.csv",
        index=False,
        float_format="%.12g",
    )
    return per_seed, summary


def run(
    root: Path,
    jobs: int,
    skip_dense: bool,
    skip_q: bool,
    skip_heavy: bool,
) -> None:
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if skip_dense:
        dense = pd.read_csv(results_dir / "matched_timing_power.csv")
        audit = pd.read_csv(results_dir / "matched_timing_anchor_audit.csv")
    else:
        _, dense = _dense_power(results_dir, jobs)
        audit = _anchor_audit(results_dir, dense)
    if not skip_q:
        _q_sensitivity(root, results_dir, jobs)
    if not skip_heavy:
        _heavy_null(results_dir, jobs)

    output_names = [
        "matched_timing_seed_map.csv",
        "matched_timing_seed_results.csv",
        "matched_timing_power.csv",
        "matched_timing_anchor_audit.csv",
    ]
    if not skip_q:
        output_names.extend(
            [
                "q_sensitivity_null_seed_results.csv",
                "q_sensitivity_null.csv",
                "q_sensitivity_power_seed_results.csv",
                "q_sensitivity_power.csv",
            ]
        )
    if not skip_heavy:
        output_names.extend(
            ["heavy_tailed_null_seed_results.csv", "heavy_tailed_null.csv"]
        )
    manifest = {
        "schema_version": 1,
        "primary_q": 0.10,
        "new_admission_floor": "h=1/q at every q; mandatory for primary results",
        "inspection_frequency_trading_days": 5,
        "proposal_weights": "gamma_j=1/[j(j+1)]",
        "dense_annual_sharpes": sorted(dense["annual_sharpe"].unique().tolist()),
        "correlations": sorted(dense["correlation"].unique().tolist()),
        "repetitions_per_power_cell": 250,
        "anchor_seed_stream": {
            "base": 314159,
            "cells": "published {0.5,1,1.5,2,3} x {0,0.5}",
        },
        "inserted_seed_stream": {
            "base": 20260730,
            "cells": "{0.75,1.25,1.75,2.25,2.5,2.75} x {0,0.5}",
        },
        "q_sensitivity_levels": list(Q_SENSITIVITY_LEVELS),
        "q_sensitivity_null_seed_range": [271828, 272327],
        "q_sensitivity_power_cell": {"annual_sharpe": 1.5, "correlation": 0.5},
        "q_sensitivity_power_seed_range": [315909, 316158],
        "heavy_tailed_null": {
            "label": "new untuned diagnostic; not the unrecoverable prior run",
            "student_t_df": 3,
            "innovation_standardization": "divide by sqrt(3)",
            "variance_recursion": "h[0]=1; h[t]=0.05+0.10*u[t-1]^2+0.85*h[t-1]",
            "score": "clip(0.25*sqrt(h[t])*(sqrt(.5)*u[t]+sqrt(.5)*v[j,t]),-1,1)",
            "seed_range": [20260730, 20261229],
            "repetitions": 500,
        },
        "pre_floor_anchor_audit_max_absolute_difference": float(
            audit["absolute_difference"].max()
        ),
        "code_sha256": {
            name: _sha256(root / "code" / name)
            for name in ["inference.py", "simulation.py", "run_empirical_finish.py"]
        },
        "output_sha256": {
            name: _sha256(results_dir / name) for name in output_names
        },
    }
    _write_json(results_dir / "empirical_finish_manifest.json", manifest)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-q", action="store_true")
    parser.add_argument("--skip-heavy", action="store_true")
    arguments = parser.parse_args()
    if arguments.jobs < 1:
        raise ValueError("--jobs must be positive")
    run(
        arguments.root.resolve(),
        jobs=arguments.jobs,
        skip_dense=arguments.skip_dense,
        skip_q=arguments.skip_q,
        skip_heavy=arguments.skip_heavy,
    )
