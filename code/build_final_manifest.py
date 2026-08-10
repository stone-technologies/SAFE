"""Hash the files supporting the post-archive acceptance revision.

The original run manifests are historical records and are intentionally not
rewritten.  This manifest covers the added design locks, prospective power
experiments, conservative comparator, revised figures, paper source, and final
PDFs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SUPPORTING_FILES = [
    "ACCEPTANCE_REVISION_REPORT.md",
    "README.md",
    "SAFE_ALPHA_empirical_finish_report.md",
    "SAFE_ALPHA_referee_report.md",
    "submission_checklist.md",
    "code/conservative_baseline.py",
    "code/make_empirical_finish_figures.py",
    "code/power_upgrade.py",
    "code/run_conservative_baseline.py",
    "code/run_power_upgrade.py",
    "code/run_power_upgrade_final_heavy_null.py",
    "code/run_power_upgrade_v2_holdout.py",
    "code/run_power_upgrade_validation.py",
    "tests/test_conservative_baseline.py",
    "tests/test_core.py",
    "results/conservative_baseline_campaign75-geometric-daily_manifest.json",
    "results/conservative_baseline_campaign75-geometric-daily_null_seed_results.csv",
    "results/conservative_baseline_campaign75-geometric-daily_null_summary.csv",
    "results/conservative_baseline_campaign75-geometric-daily_power_seed_results.csv",
    "results/conservative_baseline_campaign75-geometric-daily_power_summary.csv",
    "results/conservative_baseline_campaign75-geometric-daily_primary_paired.csv",
    "results/power_upgrade_development_v4.csv",
    "results/power_upgrade_development_v4_summary.csv",
    "results/power_upgrade_final_heavy_null_seed_results.csv",
    "results/power_upgrade_final_heavy_null_summary.csv",
    "results/power_upgrade_geometric_development.csv",
    "results/power_upgrade_geometric_development.json",
    "results/power_upgrade_lock.json",
    "results/power_upgrade_v2_holdout_seed_results.csv",
    "results/power_upgrade_v2_holdout_summary.csv",
    "results/power_upgrade_v2_lock.json",
    "results/power_upgrade_v2_primary_paired.csv",
    "results/power_upgrade_v2_rank_seed_results.csv",
    "results/power_upgrade_v2_rank_summary.csv",
    "paper/main.tex",
    "paper/body.tex",
    "paper/references.bib",
    "paper/appendix.tex",
    "paper/extended.tex",
    "paper/figures/calibration_power.pdf",
    "paper/figures/evidence_path_repaired.pdf",
    "paper/main.pdf",
    "paper/extended.pdf",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in SUPPORTING_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing supporting files: {missing}")

    manifest = {
        "manifest_role": (
            "post-archive acceptance revision; original run manifests remain frozen"
        ),
        "locked_design": "campaign75-geometric-daily",
        "declared_campaign_cap": 50,
        "campaign_share": 0.75,
        "inspection_frequency_trading_days": 1,
        "bet_multipliers": [1.0, 2.0, 4.0],
        "component_cap": 0.5,
        "holdout_seed_policy": (
            "six disjoint blocks beginning at 52000000; 250 paths per cell"
        ),
        "unit_tests": {"passed": 22, "failed": 0},
        "submission_pages": 8,
        "extended_report_pages": 14,
        "sha256": {name: sha256(ROOT / name) for name in SUPPORTING_FILES},
    }
    destination = ROOT / "results" / "final_revision_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
