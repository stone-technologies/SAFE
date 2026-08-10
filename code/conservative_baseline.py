from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inference import weighted_ebh


@dataclass
class ProcedureOutcome:
    promoted: np.ndarray
    promotion_day: np.ndarray
    stopped_e: np.ndarray


@dataclass
class JointFWERComparison:
    """Paired decisions from identical score paths and betting factors."""

    joint: ProcedureOutcome
    alpha_spending: ProcedureOutcome
    raw_terminal_e: np.ndarray


def proposal_alpha_spending(
    e_values: np.ndarray,
    gammas: np.ndarray,
    level: float,
) -> np.ndarray:
    """Proposal-wise e-Bonferroni decisions at levels ``level * gamma_j``.

    If every null stream is an e-process in the global filtration and the
    proposal weights sum to at most one, Ville's inequality and a union bound
    give strong FWER control under arbitrary dependence.  Because each
    ``gamma_j <= 1``, the boundary also implies the individual evidence floor
    ``1 / level`` used by SAFE-ALPHA.
    """
    e_values = np.asarray(e_values, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    if e_values.ndim != 1 or gammas.ndim != 1 or len(e_values) != len(gammas):
        raise ValueError("e_values and gammas must be aligned vectors")
    if not 0.0 < level <= 1.0:
        raise ValueError("level must lie in (0, 1]")
    if np.any(~np.isfinite(gammas)) or np.any(gammas <= 0.0):
        raise ValueError("gammas must be finite and positive")
    if float(gammas.sum()) > 1.0 + 1e-12:
        raise ValueError("proposal weights must sum to at most one")
    boundary = 1.0 / (level * gammas)
    return np.isfinite(e_values) & (e_values >= boundary)


def run_joint_and_alpha_spending(
    values: np.ndarray,
    arrivals: np.ndarray,
    base_lambdas: np.ndarray,
    gammas: np.ndarray,
    bet_multipliers: tuple[float, ...] = (1.0,),
    level: float = 0.10,
    inspect_every: int = 1,
) -> JointFWERComparison:
    """Run persistent e-BH and proposal alpha spending on one evidence path.

    Both procedures receive the same candidates, arrival times, predictable
    proposal-time bets, mixture components, proposal weights, and inspection
    dates.  They maintain separate stopped copies only because each procedure
    freezes a strategy at its own first promotion.  The untouched terminal
    e-values are returned as an implementation audit.
    """
    values = np.asarray(values, dtype=float)
    arrivals = np.asarray(arrivals, dtype=int)
    base_lambdas = np.asarray(base_lambdas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    multipliers = np.asarray(bet_multipliers, dtype=float)
    proposals = len(arrivals)
    if values.ndim != 2 or values.shape[1] != proposals:
        raise ValueError("values and arrivals must describe the same proposals")
    if len(base_lambdas) != proposals or len(gammas) != proposals:
        raise ValueError("lambdas, gammas, and arrivals must align")
    if inspect_every < 1:
        raise ValueError("inspect_every must be positive")
    if len(multipliers) == 0 or np.any(~np.isfinite(multipliers)) or np.any(
        multipliers <= 0.0
    ):
        raise ValueError("bet multipliers must be finite and positive")
    if np.any(~np.isfinite(gammas)) or np.any(gammas <= 0.0):
        raise ValueError("gammas must be finite and positive")
    if float(gammas.sum()) > 1.0 + 1e-12:
        raise ValueError("proposal weights must sum to at most one")
    if not 0.0 < level <= 1.0:
        raise ValueError("level must lie in (0, 1]")

    component_lambdas = np.minimum(
        0.5, base_lambdas[:, None] * multipliers[None, :]
    )
    raw_components = np.ones_like(component_lambdas)
    joint_components = np.ones_like(component_lambdas)
    alpha_components = np.ones_like(component_lambdas)
    joint_promoted = np.zeros(proposals, dtype=bool)
    alpha_promoted = np.zeros(proposals, dtype=bool)
    joint_day = np.full(proposals, -1, dtype=int)
    alpha_day = np.full(proposals, -1, dtype=int)

    first_day = max(int(arrivals.min()), 0) if proposals else 0
    for t in range(first_day, len(values)):
        active = arrivals < t
        factors = 1.0 + component_lambdas * values[t, :, None]
        raw_components[active] *= factors[active]
        joint_update = active & ~joint_promoted
        alpha_update = active & ~alpha_promoted
        joint_components[joint_update] *= factors[joint_update]
        alpha_components[alpha_update] *= factors[alpha_update]

        if t % inspect_every != 0 and t != len(values) - 1:
            continue
        eligible = active | joint_promoted | alpha_promoted

        joint_e = joint_components.mean(axis=1)
        joint_input = np.zeros(proposals, dtype=float)
        joint_input[eligible] = joint_e[eligible]
        joint_gate = weighted_ebh(
            joint_input,
            gammas,
            level,
            previously_rejected=joint_promoted,
            admission_floor=1.0 / level,
        ) & eligible
        joint_new = joint_gate & ~joint_promoted
        joint_day[joint_new] = t
        joint_promoted |= joint_new

        alpha_e = alpha_components.mean(axis=1)
        alpha_gate = proposal_alpha_spending(alpha_e, gammas, level) & active
        alpha_new = alpha_gate & ~alpha_promoted
        alpha_day[alpha_new] = t
        alpha_promoted |= alpha_new

    return JointFWERComparison(
        joint=ProcedureOutcome(
            promoted=joint_promoted,
            promotion_day=joint_day,
            stopped_e=joint_components.mean(axis=1),
        ),
        alpha_spending=ProcedureOutcome(
            promoted=alpha_promoted,
            promotion_day=alpha_day,
            stopped_e=alpha_components.mean(axis=1),
        ),
        raw_terminal_e=raw_components.mean(axis=1),
    )
