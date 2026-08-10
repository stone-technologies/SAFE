from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inference import spending_weights, weighted_ebh
from simulation import _adaptive_order, _initial_lambdas


@dataclass(frozen=True)
class UpgradeDesign:
    name: str
    campaign_share: float
    bet_multipliers: tuple[float, ...]
    anchor_first: bool = False
    inspect_every: int = 5


BASELINE = UpgradeDesign(
    name="telescoping-fixed",
    campaign_share=0.0,
    bet_multipliers=(1.0,),
)

HEDGED = UpgradeDesign(
    name="telescoping-hedged",
    campaign_share=0.0,
    bet_multipliers=(0.25, 0.50, 1.0),
)

CAMPAIGN_25 = UpgradeDesign(
    name="campaign25-fixed",
    campaign_share=0.25,
    bet_multipliers=(1.0,),
)

CAMPAIGN = UpgradeDesign(
    name="campaign50-fixed",
    campaign_share=0.5,
    bet_multipliers=(1.0,),
)

CAMPAIGN_75 = UpgradeDesign(
    name="campaign75-fixed",
    campaign_share=0.75,
    bet_multipliers=(1.0,),
)

UPGRADED = UpgradeDesign(
    name="campaign-hedged",
    campaign_share=0.5,
    bet_multipliers=(0.25, 0.50, 1.0),
)

UPWARD_GEOMETRIC_HEDGE = UpgradeDesign(
    name="campaign50-geometric-upward",
    campaign_share=0.5,
    bet_multipliers=(1.0, 2.0, 4.0),
)

ANCHORED_25 = UpgradeDesign(
    name="anchored25-fixed",
    campaign_share=0.25,
    bet_multipliers=(1.0,),
    anchor_first=True,
)

ANCHORED_50 = UpgradeDesign(
    name="anchored50-fixed",
    campaign_share=0.50,
    bet_multipliers=(1.0,),
    anchor_first=True,
)

ANCHORED_75 = UpgradeDesign(
    name="anchored75-fixed",
    campaign_share=0.75,
    bet_multipliers=(1.0,),
    anchor_first=True,
)

BASELINE_DAILY = UpgradeDesign(
    name="telescoping-fixed-daily",
    campaign_share=0.0,
    bet_multipliers=(1.0,),
    inspect_every=1,
)

CAMPAIGN_75_DAILY = UpgradeDesign(
    name="campaign75-fixed-daily",
    campaign_share=0.75,
    bet_multipliers=(1.0,),
    inspect_every=1,
)

CAMPAIGN_75_GEOMETRIC_DAILY = UpgradeDesign(
    name="campaign75-geometric-daily",
    campaign_share=0.75,
    bet_multipliers=(1.0, 2.0, 4.0),
    inspect_every=1,
)

ANCHORED_75_DAILY = UpgradeDesign(
    name="anchored75-fixed-daily",
    campaign_share=0.75,
    bet_multipliers=(1.0,),
    anchor_first=True,
    inspect_every=1,
)

DEVELOPMENT_DESIGNS = (
    BASELINE,
    HEDGED,
    CAMPAIGN_25,
    CAMPAIGN,
    CAMPAIGN_75,
    UPGRADED,
    UPWARD_GEOMETRIC_HEDGE,
    ANCHORED_25,
    ANCHORED_50,
    ANCHORED_75,
    BASELINE_DAILY,
    CAMPAIGN_75_DAILY,
    CAMPAIGN_75_GEOMETRIC_DAILY,
    ANCHORED_75_DAILY,
)


def campaign_spending_weights(
    realized_count: int,
    eta: float,
    campaign_cap: int,
) -> np.ndarray:
    """Return an online-safe prefix of a fixed-cap campaign schedule.

    ``campaign_cap`` (M) and ``eta`` must be fixed before the campaign.  For
    proposal rank j, the schedule is

        gamma_j = (1-eta) / [j(j+1)] + eta/M,  j <= M,
        gamma_j = (1-eta) / [j(j+1)],          j > M.

    Consequently, extending ``realized_count`` never changes an earlier
    weight, proposals beyond M retain the telescoping component, and the
    infinite sum is one.  If fewer than M proposals arrive, the unused
    uniform campaign mass remains reserved rather than being reallocated.
    """
    if isinstance(realized_count, (bool, np.bool_)) or not isinstance(
        realized_count, (int, np.integer)
    ):
        raise TypeError("realized_count must be a nonnegative integer")
    if realized_count < 0:
        raise ValueError("realized_count must be nonnegative")
    if isinstance(campaign_cap, (bool, np.bool_)) or not isinstance(
        campaign_cap, (int, np.integer)
    ):
        raise TypeError("campaign_cap must be a positive integer")
    if campaign_cap <= 0:
        raise ValueError("campaign_cap must be positive")
    eta = float(eta)
    if not np.isfinite(eta) or not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    weights = (1.0 - eta) * spending_weights(realized_count)
    weights[: min(realized_count, campaign_cap)] += eta / float(campaign_cap)
    return weights


def anchored_campaign_spending_weights(
    realized_count: int,
    eta: float,
    campaign_cap: int,
) -> np.ndarray:
    """Keep half the budget on the declared anchor and spread the remainder.

    Slot one retains gamma_1=1/2.  A fraction eta of the residual half-budget
    is uniform over slots 2,...,M, while the other fraction retains the
    telescoping schedule and its unbounded tail.  Thus the infinite sum is
    one and the anchor's original boundary is unchanged.
    """
    if isinstance(realized_count, (bool, np.bool_)) or not isinstance(
        realized_count, (int, np.integer)
    ):
        raise TypeError("realized_count must be a nonnegative integer")
    if realized_count < 0:
        raise ValueError("realized_count must be nonnegative")
    if isinstance(campaign_cap, (bool, np.bool_)) or not isinstance(
        campaign_cap, (int, np.integer)
    ):
        raise TypeError("campaign_cap must be an integer at least two")
    if campaign_cap < 2:
        raise ValueError("anchored campaign_cap must be at least two")
    eta = float(eta)
    if not np.isfinite(eta) or not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    if realized_count == 0:
        return np.empty(0, dtype=float)
    if realized_count == 1:
        return np.array([0.5], dtype=float)
    weights = (1.0 - eta) * spending_weights(realized_count)
    weights[0] = 0.5
    final_campaign_slot = min(realized_count, campaign_cap)
    weights[1:final_campaign_slot] += eta / (2.0 * float(campaign_cap - 1))
    return weights


def design_spending_weights(
    realized_count: int,
    design: UpgradeDesign,
    campaign_cap: int,
) -> np.ndarray:
    if design.anchor_first:
        return anchored_campaign_spending_weights(
            realized_count, design.campaign_share, campaign_cap
        )
    return campaign_spending_weights(
        realized_count, design.campaign_share, campaign_cap
    )


@dataclass
class DesignOutcome:
    promoted: np.ndarray
    promotion_day: np.ndarray
    terminal_rejected: np.ndarray
    safe_e: np.ndarray
    raw_e: np.ndarray


def run_mixture_gate(
    values: np.ndarray,
    arrivals: np.ndarray,
    base_lambdas: np.ndarray,
    gammas: np.ndarray,
    bet_multipliers: tuple[float, ...],
    level: float = 0.10,
    inspect_every: int = 5,
) -> DesignOutcome:
    """Run SAFE and a matched terminal gate on one mixture e-process.

    Every multiplier and base bet must be fixed or predictable at proposal
    time.  Under a null for which X_t lies in [-1, 1] and has nonpositive
    conditional mean, each component factor

        1 + min(0.5, multiplier * base_lambda) * X_t

    defines a nonnegative e-process.  The fixed arithmetic mean of those
    component wealths is again an e-process.  It also retains pathwise
    log-wealth regret of at most log(number of components) relative to the
    best included component.  Stopping all components when their mixture is
    admitted preserves validity by optional stopping.
    """
    values = np.asarray(values, dtype=float)
    arrivals = np.asarray(arrivals, dtype=int)
    base_lambdas = np.asarray(base_lambdas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    multipliers = np.asarray(bet_multipliers, dtype=float)
    if (
        arrivals.ndim != 1
        or values.ndim != 2
        or values.shape[1] != len(arrivals)
    ):
        raise ValueError("values and arrivals must align")
    if (
        base_lambdas.ndim != 1
        or gammas.ndim != 1
        or len(base_lambdas) != len(arrivals)
        or len(gammas) != len(arrivals)
    ):
        raise ValueError("lambdas, gammas, and arrivals must align")
    if np.any(~np.isfinite(values)) or np.any(np.abs(values) > 1.0):
        raise ValueError("values must be finite and lie in [-1, 1]")
    if np.any(~np.isfinite(base_lambdas)) or np.any(base_lambdas < 0.0):
        raise ValueError("base lambdas must be finite and nonnegative")
    if (
        multipliers.ndim != 1
        or len(multipliers) == 0
        or np.any(~np.isfinite(multipliers))
        or np.any(multipliers <= 0.0)
    ):
        raise ValueError("bet multipliers must be finite and positive")
    if (
        np.any(~np.isfinite(gammas))
        or np.any(gammas < 0.0)
        or float(gammas.sum()) > 1.0 + 1e-12
    ):
        raise ValueError(
            "gammas must be finite, nonnegative, and sum to at most one"
        )
    if not isinstance(inspect_every, (int, np.integer)) or inspect_every <= 0:
        raise ValueError("inspect_every must be a positive integer")

    component_lambdas = np.minimum(
        0.5, base_lambdas[:, None] * multipliers[None, :]
    )
    raw_components = np.ones_like(component_lambdas)
    safe_components = np.ones_like(component_lambdas)
    promoted = np.zeros(len(arrivals), dtype=bool)
    promotion_day = np.full(len(arrivals), -1, dtype=int)

    first_day = max(int(arrivals.min()), 0) if len(arrivals) else 0
    for t in range(first_day, len(values)):
        active = arrivals < t
        factors = 1.0 + component_lambdas * values[t, :, None]
        raw_components[active] *= factors[active]
        updating = active & ~promoted
        safe_components[updating] *= factors[updating]

        if t % inspect_every == 0 or t == len(values) - 1:
            safe_e = safe_components.mean(axis=1)
            eligible = active | promoted
            gate_input = np.zeros(len(arrivals), dtype=float)
            gate_input[eligible] = safe_e[eligible]
            gate = weighted_ebh(
                gate_input,
                gammas,
                level,
                previously_rejected=promoted,
                admission_floor=1.0 / level,
            ) & eligible
            new = gate & ~promoted
            promotion_day[new] = t
            promoted |= new

    raw_e = raw_components.mean(axis=1)
    safe_e = safe_components.mean(axis=1)
    terminal = weighted_ebh(
        raw_e,
        gammas,
        level,
        admission_floor=1.0 / level,
    )
    return DesignOutcome(
        promoted=promoted,
        promotion_day=promotion_day,
        terminal_rejected=terminal,
        safe_e=safe_e,
        raw_e=raw_e,
    )


def planted_path(
    annual_sharpe: float,
    correlation: float,
    seed: int,
    candidates: int = 80,
    alternatives: int = 16,
    pre_days: int = 504,
    post_days: int = 1260,
    proposal_gap: int = 15,
    maximum_proposals: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one path shared by every development design."""
    rng = np.random.default_rng(seed)
    truth = np.zeros(candidates, dtype=bool)
    truth[rng.choice(candidates, alternatives, replace=False)] = True
    total_days = pre_days + post_days
    common = rng.normal(size=(total_days, 1))
    idiosyncratic = rng.normal(size=(total_days, candidates))
    sigma = 0.25
    drift = annual_sharpe * sigma / np.sqrt(252.0)
    scores = sigma * (
        np.sqrt(correlation) * common
        + np.sqrt(max(1.0 - correlation, 0.0)) * idiosyncratic
    )
    scores += truth[None, :] * drift
    scores = np.clip(scores, -1.0, 1.0)
    proposal_times = np.arange(
        pre_days, total_days - 63, proposal_gap, dtype=int
    )
    selected, arrivals = _adaptive_order(
        scores,
        proposal_times,
        maximum=maximum_proposals,
        forced_first=int(np.argmax(scores[:pre_days].mean(axis=0))),
        lookback=pre_days,
    )
    values = scores[:, selected]
    lambdas = _initial_lambdas(
        scores,
        selected,
        arrivals,
        lookback=pre_days,
    )
    return values, arrivals, lambdas, truth[selected], truth


def summarize_design(
    outcome: DesignOutcome,
    selected_truth: np.ndarray,
    all_truth: np.ndarray,
    arrivals: np.ndarray,
) -> dict[str, float]:
    discoveries = outcome.promoted
    count = int(discoveries.sum())
    true = int(np.sum(discoveries & selected_truth))
    false = int(np.sum(discoveries & ~selected_truth))
    delays = outcome.promotion_day[discoveries & selected_truth] - arrivals[
        discoveries & selected_truth
    ]
    terminal = outcome.terminal_rejected
    terminal_true = int(np.sum(terminal & selected_truth))
    terminal_false = int(np.sum(terminal & ~selected_truth))
    return {
        "fdp": false / max(count, 1),
        "end_to_end_power": true / max(int(all_truth.sum()), 1),
        "conditional_power": true / max(int(selected_truth.sum()), 1),
        "proposal_recall": int(selected_truth.sum()) / max(int(all_truth.sum()), 1),
        "discoveries": count,
        "delay": float(np.mean(delays)) if len(delays) else np.nan,
        "terminal_fdp": terminal_false / max(int(terminal.sum()), 1),
        "terminal_end_to_end_power": terminal_true / max(int(all_truth.sum()), 1),
    }
