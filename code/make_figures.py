from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "SAFE-ALPHA": "#1B5E8C",
    "Terminal BH": "#D97904",
    "Uncorrected 5%": "#8C3B72",
    "Top-5 backtest": "#5B6770",
    "Same-bar leakage": "#B33A3A",
    "Repeated 5% t-test": "#8C3B72",
    "Unadjusted one-year holdout": "#7A7A7A",
}


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def calibration_power(results: Path, output: Path) -> None:
    calibration = pd.read_csv(results / "null_calibration.csv")
    power = pd.read_csv(results / "planted_power.csv")
    power = power[power["correlation"] == 0.5]

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.25))
    axis = axes[0]
    labels = [
        "SAFE-ALPHA",
        "Same-bar leakage",
        "Repeated 5% t-test",
        "Unadjusted one-year holdout",
    ]
    subset = calibration.set_index("method").loc[labels]
    x = np.arange(len(labels))
    values = subset["false_discovery_probability"].to_numpy()
    errors = np.vstack(
        [
            values - subset["ci_low"].to_numpy(),
            subset["ci_high"].to_numpy() - values,
        ]
    )
    axis.bar(
        x,
        values,
        yerr=errors,
        width=0.70,
        capsize=2,
        color=[COLORS[label] for label in labels],
        linewidth=0,
    )
    axis.axhline(0.10, color="black", linewidth=0.8, linestyle="--", label=r"$q=0.10$")
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Probability of any false promotion")
    axis.set_xticks(x)
    axis.set_xticklabels(["SAFE", "Leakage", "Repeated\nt-test", "One\nholdout"])
    axis.set_title("(a) Wild-sign global-null stress test")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8)

    axis = axes[1]
    for method in ["SAFE-ALPHA", "Terminal BH", "Uncorrected 5%"]:
        group = power[power["method"] == method].sort_values("annual_sharpe")
        axis.plot(
            group["annual_sharpe"],
            group[
                (
                    "mean_end_to_end_power"
                    if "mean_end_to_end_power" in group
                    else "mean_power"
                )
            ],
            marker="o",
            markersize=3.2,
            linewidth=1.35,
            color=COLORS[method],
            label=method,
        )
    axis.set_xlim(0.4, 3.1)
    axis.set_ylim(0, 1.03)
    axis.set_xlabel("Planted annualized Sharpe")
    axis.set_ylabel("End-to-end discovery power")
    axis.set_title(r"(b) Power with cross-strategy correlation $\rho=0.5$")
    axis.grid(color="#D9D9D9", linewidth=0.45, alpha=0.8)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout(w_pad=1.7)
    figure.savefig(output / "calibration_power.pdf")
    figure.savefig(output / "calibration_power.png", dpi=300)
    plt.close(figure)


def evidence_path(results: Path, output: Path) -> None:
    evidence = pd.read_csv(results / "e_value_paths.csv", index_col=0, parse_dates=True)
    ledger = pd.read_csv(results / "certification_ledger.csv")
    ledger = ledger.set_index("strategy_id")
    scaled = evidence.copy()
    for column in scaled:
        scaled[column] = (
            0.10 * float(ledger.loc[column, "gamma"]) * scaled[column]
        )
    selected = scaled.iloc[-1].nlargest(5).index.tolist()
    if "market_ew30" not in selected:
        selected[-1] = "market_ew30"

    figure, axis = plt.subplots(figsize=(3.45, 2.15))
    palette = ["#1B5E8C", "#D97904", "#4D7C52", "#8C3B72", "#5B6770"]
    for color, strategy in zip(palette, selected):
        label = strategy.replace("_", " ")
        axis.plot(
            scaled.index,
            scaled[strategy].clip(lower=1e-8),
            linewidth=1.15,
            color=color,
            label=label,
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_yscale("log")
    axis.set_ylabel(r"Weighted evidence $q\gamma_jE_{j,t}$")
    axis.set_xlabel("Date")
    axis.set_title("Post-proposal evidence and the one-discovery boundary")
    axis.grid(color="#D9D9D9", linewidth=0.4, alpha=0.7)
    axis.legend(frameon=False, fontsize=5.8, loc="lower left")
    figure.tight_layout()
    figure.savefig(output / "evidence_path.pdf")
    figure.savefig(output / "evidence_path.png", dpi=300)
    plt.close(figure)


def tables(results: Path, output: Path) -> None:
    matched = results / "economic_metrics_matched.csv"
    metrics = pd.read_csv(
        matched if matched.exists() else results / "economic_metrics.csv"
    )
    main = metrics[metrics["cost_bps"] == 2.0].copy()
    main["annual_return"] *= 100
    main["annual_volatility"] *= 100
    main["certainty_equivalent"] *= 100
    main["max_drawdown"] *= 100
    main["cvar_5_daily"] *= 100
    columns = [
        "method",
        "selected",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "annual_turnover",
    ]
    main[columns].to_csv(output / "economic_table.csv", index=False)

    calibration = pd.read_csv(results / "null_calibration.csv")
    calibration.to_csv(output / "calibration_table.csv", index=False)

    power = pd.read_csv(results / "planted_power.csv")
    power[
        (power["correlation"] == 0.5)
        & (power["annual_sharpe"].isin([1.0, 2.0, 3.0]))
    ].to_csv(output / "power_table.csv", index=False)

    diagnostics = json.loads((results / "classical_diagnostics.json").read_text())
    pd.DataFrame(
        [
            {
                "diagnostic": "White Reality Check",
                "value": diagnostics["white_reality_check"]["p_value"],
            },
            {
                "diagnostic": "Deflated Sharpe probability",
                "value": diagnostics["deflated_sharpe_ratio"][
                    "deflated_sharpe_probability"
                ],
            },
            {
                "diagnostic": "PBO",
                "value": diagnostics["probability_backtest_overfitting"]["pbo"],
            },
        ]
    ).to_csv(output / "classical_table.csv", index=False)


def main(root: Path) -> None:
    setup()
    results = root / "results"
    output = root / "paper" / "figures"
    table_output = root / "paper" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    table_output.mkdir(parents=True, exist_ok=True)
    calibration_power(results, output)
    evidence_path(results, output)
    tables(results, table_output)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
