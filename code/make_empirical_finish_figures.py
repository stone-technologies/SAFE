from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/safe-alpha-mpl")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COLORS = {
    "SAFE-ALPHA": "#1B5E8C",
    "Matched terminal weighted e-BH": "#D97904",
    "Same-bar leakage": "#B33A3A",
    "Repeated 5% t-test": "#8C3B72",
}

CALIBRATION_POWER_CAPTION = (
    "Calibration, power, and proposal-rank tradeoffs. Panel (a) shows the "
    "nine-level market-shaped sign-null curve for the original protocol and "
    "the q=.10 points for the frozen final joint gate and proposal-wise FWER "
    "comparator; bands and bars are Wilson 95% intervals. Panel (b) reports "
    "250 paired replications per effect at rho=.5, with seed-level normal 95% "
    "bands. Panel (c) uses the newly seeded central holdout and shows that the "
    "finite-campaign prior trades early-rank power for middle- and late-rank "
    "power; the geometric betting mixture recovers part of the early loss."
)

EVIDENCE_PATH_CAPTION = (
    "Public replay and selection decay. Panel (a) shows weighted evidence for "
    "the predeclared market control and the three symbolic proposals with the "
    "largest recorded path maxima. Triangles mark proposal dates, the star "
    "marks certification, and the dashed line is the one-discovery boundary. "
    "Panel (b) compares the five-year selection Sharpe with the strictly later "
    "one-year Sharpe for 74 horizon-complete symbolic proposals; the diamond "
    "marks the coordinate-wise medians and the dashed diagonal denotes no "
    "selection decay. Display choices do not enter the testing rule."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def matched_timing(results: Path, output: Path) -> None:
    power = pd.read_csv(results / "matched_timing_power.csv")
    q_power = pd.read_csv(results / "q_sensitivity_power.csv")
    q_null = pd.read_csv(results / "q_sensitivity_null.csv")
    methods = ["SAFE-ALPHA", "Matched terminal weighted e-BH"]
    labels = {
        "SAFE-ALPHA": "SAFE-ALPHA",
        "Matched terminal weighted e-BH": "Matched terminal weighted e-BH",
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 4.7))
    for axis, correlation, title in [
        (axes[0, 0], 0.0, r"(a) Independent streams ($\rho=0$)"),
        (axes[0, 1], 0.5, r"(b) Correlated streams ($\rho=0.5$)"),
    ]:
        for method in methods:
            group = power[
                (power["method"] == method)
                & (power["correlation"] == correlation)
            ].sort_values("annual_sharpe")
            axis.plot(
                group["annual_sharpe"],
                group["mean_end_to_end_power"],
                marker="o",
                markersize=2.8,
                linewidth=1.35,
                color=COLORS[method],
                label=labels[method],
            )
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Planted annualized Sharpe")
        axis.set_ylabel("End-to-end power")
        axis.set_title(title)
        axis.grid(color="#D9D9D9", linewidth=0.45, alpha=0.8)
    axes[0, 0].legend(frameon=False, loc="lower right")

    axis = axes[1, 0]
    for method in methods:
        group = power[
            (power["method"] == method) & (power["correlation"] == 0.5)
        ].sort_values("annual_sharpe")
        axis.plot(
            group["annual_sharpe"],
            group["mean_delay_days"],
            marker="o",
            markersize=2.8,
            linewidth=1.35,
            color=COLORS[method],
            label=labels[method],
        )
    axis.set_xlabel("Planted annualized Sharpe")
    axis.set_ylabel("Mean true-discovery delay (days)")
    axis.set_title(r"(c) Decision timing at $\rho=0.5$")
    axis.grid(color="#D9D9D9", linewidth=0.45, alpha=0.8)

    axis = axes[1, 1]
    for method in methods:
        group = q_power[q_power["method"] == method].sort_values("q")
        axis.plot(
            group["q"],
            group["mean_end_to_end_power"],
            marker="o",
            markersize=2.8,
            linewidth=1.35,
            color=COLORS[method],
            label=labels[method],
        )
    axis.plot(
        q_null["q"],
        q_null["false_discovery_probability"],
        color="#222222",
        linestyle="--",
        marker="s",
        markersize=2.5,
        linewidth=1.0,
        label="SAFE all-null any-false",
    )
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Target q (admission floor = 1/q)")
    axis.set_ylabel("Power / probability")
    axis.set_title(r"(d) q sensitivity, SR=1.5 and $\rho=0.5$")
    axis.grid(color="#D9D9D9", linewidth=0.45, alpha=0.8)
    axis.legend(frameon=False, loc="upper left")
    figure.suptitle(
        "Matched evidence and proposal weights isolate decision timing",
        fontsize=10.0,
        y=1.01,
    )
    figure.tight_layout(w_pad=1.5, h_pad=1.5)
    figure.savefig(output / "matched_timing_comparison.pdf")
    figure.savefig(output / "matched_timing_comparison.png", dpi=300)
    plt.close(figure)


def calibration_power(results: Path, output: Path) -> None:
    q_null = pd.read_csv(results / "q_sensitivity_null.csv")
    original_seed = pd.read_csv(results / "matched_timing_seed_results.csv")
    final_seed = pd.read_csv(
        results / "conservative_baseline_campaign75-geometric-daily_power_seed_results.csv"
    )
    final_null = pd.read_csv(
        results / "conservative_baseline_campaign75-geometric-daily_null_summary.csv"
    )
    rank = pd.read_csv(results / "power_upgrade_v2_rank_summary.csv")

    original_seed = original_seed[original_seed["correlation"] == 0.5].copy()
    final_seed = final_seed[final_seed["correlation"] == 0.5].copy()

    def mean_band(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
        grouped = frame.groupby(group_columns)["end_to_end_power"]
        summary = grouped.agg(["mean", "std", "count"]).reset_index()
        summary["half"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
        return summary

    original_curve = mean_band(original_seed, ["method", "annual_sharpe"])
    final_curve = mean_band(final_seed, ["method", "annual_sharpe"])

    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.36))

    axis = axes[0]
    axis.fill_between(
        q_null["q"],
        q_null["ci_low"],
        q_null["ci_high"],
        color="#1B5E8C",
        alpha=0.16,
        linewidth=0,
    )
    axis.plot(
        q_null["q"],
        q_null["false_discovery_probability"],
        color="#1B5E8C",
        marker="o",
        markersize=2.8,
        linewidth=1.25,
        label="Original SAFE",
    )
    axis.plot([0, 0.205], [0, 0.205], color="black", linestyle="--", linewidth=0.8)
    null_colors = {
        "Persistent weighted e-BH": "#2E7D32",
        "Proposal e-alpha-spending (FWER)": "#7B4F9D",
    }
    null_labels = {
        "Persistent weighted e-BH": "Final joint",
        "Proposal e-alpha-spending (FWER)": "Proposal FWER",
    }
    for row in final_null.itertuples(index=False):
        axis.errorbar(
            0.10,
            row.false_discovery_probability,
            yerr=[[row.false_discovery_probability - row.ci_low],
                  [row.ci_high - row.false_discovery_probability]],
            marker="s" if row.method == "Persistent weighted e-BH" else "D",
            markersize=4.0,
            color=null_colors[row.method],
            capsize=2,
            linewidth=0.9,
            label=null_labels[row.method],
        )
    axis.set_xlim(0.0, 0.205)
    axis.set_ylim(0.0, 0.21)
    axis.set_xlabel(r"Target $q$")
    axis.set_ylabel("Any false promotion")
    axis.set_title("(a) Sign-null calibration")
    axis.grid(color="#D9D9D9", linewidth=0.4, alpha=0.75)
    axis.legend(frameon=False, fontsize=5.8, loc="upper left")

    axis = axes[1]
    curve_specs = [
        (
            original_curve,
            "SAFE-ALPHA",
            "Original persistent",
            "#1B5E8C",
            "o",
        ),
        (
            original_curve,
            "Matched terminal weighted e-BH",
            "Matched terminal",
            "#D97904",
            "o",
        ),
        (
            final_curve,
            "Persistent weighted e-BH",
            "Final joint",
            "#2E7D32",
            "s",
        ),
        (
            final_curve,
            "Proposal e-alpha-spending (FWER)",
            "Proposal FWER",
            "#7B4F9D",
            "D",
        ),
    ]
    for frame, method, label, color, marker in curve_specs:
        group = frame[frame["method"] == method].sort_values("annual_sharpe")
        x = group["annual_sharpe"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        half = group["half"].to_numpy(dtype=float)
        axis.fill_between(x, mean - half, mean + half, color=color, alpha=0.09)
        axis.plot(
            x,
            mean,
            color=color,
            marker=marker,
            markersize=2.4,
            linewidth=1.15,
            label=label,
        )
    axis.set_xlim(0.45, 3.05)
    axis.set_ylim(-0.02, 1.03)
    axis.set_xlabel("Planted annualized Sharpe")
    axis.set_ylabel("End-to-end power")
    axis.set_title(r"(b) Paired power, $\rho=.5$")
    axis.grid(color="#D9D9D9", linewidth=0.4, alpha=0.75)
    axis.legend(frameon=False, fontsize=5.6, loc="lower right")

    axis = axes[2]
    labels = ["1-5", "6-10", "11-20", "21-35"]
    rank = rank[rank["rank_bin"].isin(labels)].copy()
    design_specs = [
        ("telescoping-fixed", "Original", "#1B5E8C"),
        ("campaign75-fixed-daily", "Campaign", "#8CB369"),
        ("campaign75-geometric-daily", "Final", "#2E7D32"),
    ]
    x = np.arange(len(labels), dtype=float)
    width = 0.25
    for offset, (design, label, color) in zip((-width, 0.0, width), design_specs):
        values = (
            rank[rank["design"] == design]
            .set_index("rank_bin")
            .reindex(labels)["power"]
            .to_numpy(dtype=float)
        )
        axis.bar(x + offset, values, width=width, color=color, label=label)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 0.9)
    axis.set_xlabel("Proposal rank")
    axis.set_ylabel("Power among true proposals")
    axis.set_title(r"(c) Rank profile, SR $=1.5$")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.75)
    axis.legend(frameon=False, fontsize=5.8, loc="upper right")

    figure.tight_layout(w_pad=1.05)
    figure.savefig(output / "calibration_power.pdf")
    figure.savefig(output / "calibration_power.png", dpi=300)
    plt.close(figure)


def repaired_evidence_path(results: Path, output: Path) -> None:
    evidence = pd.read_csv(results / "e_value_paths.csv", index_col=0, parse_dates=True)
    proposals = pd.read_csv(
        results / "proposal_ledger.csv", parse_dates=["proposal_date"]
    ).set_index("strategy_id")
    ledger = pd.read_csv(
        results / "certification_ledger.csv", parse_dates=["promotion_date"]
    ).set_index("strategy_id")
    weighted = evidence.mul(0.10 * ledger["gamma"], axis=1)
    symbolic = [column for column in weighted if column != "market_ew30"]
    representatives = weighted[symbolic].max().nlargest(3).index.tolist()
    selected = ["market_ew30", *representatives]
    colors = ["#1B5E8C", "#D97904", "#4D7C52", "#8C3B72"]
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.42))
    axis = axes[0]
    for color, strategy in zip(colors, selected):
        proposal_date = pd.Timestamp(proposals.loc[strategy, "proposal_date"])
        path = weighted.loc[weighted.index > proposal_date, strategy].clip(lower=1e-10)
        proposal_value = 0.10 * float(ledger.loc[strategy, "gamma"])
        plotted_path = pd.concat(
            [pd.Series([proposal_value], index=[proposal_date]), path]
        )
        label = {
            "market_ew30": "Market control",
            "time_series_momentum_lb126_monthly": (
                "126-day time-series momentum"
            ),
            "time_series_momentum_lb252_monthly": (
                "252-day time-series momentum"
            ),
            "high_volatility_lb63_b3_monthly": "63-day high-volatility",
        }.get(strategy, strategy.replace("_", " "))
        axis.plot(
            plotted_path.index,
            plotted_path,
            color=color,
            linewidth=1.15,
            label=label,
        )
        if len(path):
            axis.scatter(
                proposal_date,
                proposal_value,
                marker="^",
                s=24,
                facecolor="white",
                edgecolor=color,
                linewidth=0.9,
                zorder=4,
            )
            axis.axvline(
                proposal_date,
                color=color,
                linestyle=":",
                linewidth=0.45,
                alpha=0.5,
            )
        promotion_date = ledger.loc[strategy, "promotion_date"]
        if pd.notna(promotion_date):
            promotion_date = pd.Timestamp(promotion_date)
            axis.scatter(
                promotion_date,
                weighted.loc[promotion_date, strategy],
                marker="*",
                s=46,
                facecolor=color,
                edgecolor="black",
                linewidth=0.45,
                zorder=5,
            )
            axis.axvline(
                promotion_date,
                color=color,
                linestyle="--",
                linewidth=0.65,
                alpha=0.8,
            )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.85)
    axis.set_yscale("log")
    axis.set_ylabel(r"Weighted evidence $q\gamma_jE_{j,t}$", fontsize=8.0)
    axis.set_xlabel("Date", fontsize=8.0)
    axis.set_title("(a) Locked public replay")
    axis.grid(color="#D9D9D9", linewidth=0.4, alpha=0.7)
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(
        [
            Line2D(
                [0], [0], marker="^", color="none", markerfacecolor="white",
                markeredgecolor="black", markersize=5.0, label="Proposal"
            ),
            Line2D(
                [0], [0], marker="*", color="none", markerfacecolor="#1B5E8C",
                markeredgecolor="black", markersize=6.0, label="Certification"
            ),
        ]
    )
    labels.extend(["Proposal", "Certification"])
    axis.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        handlelength=1.6,
        columnspacing=0.9,
    )

    selection = pd.read_csv(results / "winner_curse_by_proposal.csv")
    selection = selection[selection["complete_252d_holdout"].astype(bool)]
    before = selection["preproposal_1260d_sharpe"].to_numpy(dtype=float)
    after = selection["postproposal_252d_sharpe"].to_numpy(dtype=float)
    axis = axes[1]
    axis.scatter(
        before,
        after,
        s=17,
        color="#1B5E8C",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.78,
    )
    lower = float(min(before.min(), after.min(), -1.5))
    upper = float(max(before.max(), after.max(), 2.0))
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=0.8)
    axis.axhline(0.0, color="#777777", linewidth=0.55)
    axis.axvline(0.0, color="#777777", linewidth=0.55)
    median_before = float(np.median(before))
    median_after = float(np.median(after))
    axis.scatter(
        median_before,
        median_after,
        marker="D",
        s=38,
        facecolor="#D97904",
        edgecolor="black",
        linewidth=0.5,
        zorder=4,
        label="Coordinate-wise medians",
    )
    axis.text(
        0.03,
        0.97,
        "Median: 0.449 to -0.037\nCorrelation: 0.034",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Five-year selection Sharpe")
    axis.set_ylabel("Strictly later one-year Sharpe")
    axis.set_title("(b) Adaptive symbolic proposals")
    axis.grid(color="#D9D9D9", linewidth=0.4, alpha=0.7)
    axis.legend(frameon=False, loc="lower right", fontsize=6.5)

    figure.subplots_adjust(bottom=0.29, left=0.08, right=0.99, top=0.93, wspace=0.28)
    figure.savefig(output / "evidence_path_repaired.pdf")
    figure.savefig(output / "evidence_path_repaired.png", dpi=300)
    plt.close(figure)


def heavy_tailed_null(results: Path, output: Path) -> None:
    data = pd.read_csv(results / "heavy_tailed_null.csv")
    labels = ["SAFE-ALPHA", "Same-bar leakage", "Repeated 5% t-test"]
    data = data.set_index("method").loc[labels]
    values = data["false_discovery_probability"]
    errors = [values - data["ci_low"], data["ci_high"] - values]
    figure, axis = plt.subplots(figsize=(3.55, 2.35))
    axis.bar(
        range(len(labels)),
        values,
        yerr=errors,
        capsize=2,
        width=0.68,
        color=[COLORS[label] for label in labels],
    )
    axis.axhline(0.10, color="black", linestyle="--", linewidth=0.8)
    axis.set_xticks(range(len(labels)), ["SAFE", "Same-bar\nleakage", "Repeated\nt-test"])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Probability of any false promotion")
    axis.set_title("Untuned heavy-tail and predictable-volatility null")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    figure.tight_layout()
    figure.savefig(output / "heavy_tailed_null.pdf")
    figure.savefig(output / "heavy_tailed_null.png", dpi=300)
    plt.close(figure)


def main(root: Path) -> None:
    _setup()
    results = root / "results"
    output = root / "paper" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    calibration_power(results, output)
    matched_timing(results, output)
    heavy_tailed_null(results, output)
    repaired_evidence_path(results, output)
    caption_path = results / "calibration_power_caption.txt"
    caption_path.write_text(CALIBRATION_POWER_CAPTION + "\n", encoding="utf-8")
    evidence_caption_path = results / "evidence_path_caption.txt"
    evidence_caption_path.write_text(
        EVIDENCE_PATH_CAPTION + "\n", encoding="utf-8"
    )
    output_names = [
        "calibration_power.pdf",
        "calibration_power.png",
        "matched_timing_comparison.pdf",
        "matched_timing_comparison.png",
        "heavy_tailed_null.pdf",
        "heavy_tailed_null.png",
        "evidence_path_repaired.pdf",
        "evidence_path_repaired.png",
    ]
    source_names = [
        "null_calibration.csv",
        "matched_timing_power.csv",
        "q_sensitivity_power.csv",
        "q_sensitivity_null.csv",
        "heavy_tailed_null.csv",
        "e_value_paths.csv",
        "proposal_ledger.csv",
        "certification_ledger.csv",
        "winner_curse_by_proposal.csv",
        "conservative_baseline_campaign75-geometric-daily_power_seed_results.csv",
        "conservative_baseline_campaign75-geometric-daily_null_summary.csv",
        "power_upgrade_v2_rank_summary.csv",
    ]
    manifest = {
        "code_sha256": _sha256(root / "code" / "make_empirical_finish_figures.py"),
        "source_sha256": {
            name: _sha256(results / name) for name in source_names
        },
        "caption_sha256": _sha256(caption_path),
        "captions_sha256": {
            "calibration_power_caption.txt": _sha256(caption_path),
            "evidence_path_caption.txt": _sha256(evidence_caption_path),
        },
        "output_sha256": {
            name: _sha256(output / name) for name in output_names
        },
    }
    (results / "empirical_finish_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
