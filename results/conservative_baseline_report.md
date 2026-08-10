# Valid proposal-wise FWER comparator

## Definition

Candidate (j) receives the same proposal-time weight (gamma_j), predictable
bet, score stream, and inspection dates as the persistent weighted e-BH gate.
The conservative comparator promotes (j) at the first declared inspection
at which

\[
E_{j,t}\ge \frac{1}{q\gamma_j}.
\]

This is proposal-wise e-alpha-spending, equivalently e-Bonferroni with
proposal level (q\gamma_j).  The individual floor (E_{j,t}\ge 1/q) is
automatic because (gamma_j\le 1).  Ville's inequality and
(sum_j\gamma_j\le1) give FWER at most (q), without an independence
assumption.  The comparator is deliberately conservative but fully valid for
adaptive proposal identities and arrival times under the same global
conditional e-process assumptions as SAFE-ALPHA.

## Locked paired result

The archived run uses the anchored 75 percent campaign schedule, daily
inspection, the original fixed proposal-time bet, (q=.10), and the
mandatory admission floor.  Both procedures read identical simulated scores,
adaptive proposals, bets, weights, and inspection dates.

At annualized Sharpe (1.5) and correlation (0.5), over 250 paired paths,
the persistent weighted e-BH gate attained end-to-end power (0.588), versus
(0.360) for proposal e-alpha-spending.  The paired difference was (0.228),
with a seed-level normal 95 percent interval ([0.2096,0.2464]).  The joint
gate had higher power on 218 paths and tied on 32, with no loss.  Mean FDP was
(0.00044) for the joint gate and zero for proposal e-alpha-spending.

In 500 same-date sign-null replays, the probability of any false promotion
was (0.024), with Wilson interval ([0.0138,0.0415]), for persistent
weighted e-BH, and (0.022), with Wilson interval ([0.0123,0.0390]), for
proposal e-alpha-spending.  Mean false discoveries were (0.070) and
(0.036), respectively.

The complete 22-cell estimates and all 5,500 paired seed-level results appear
in `conservative_baseline_anchored75-fixed-daily_power_summary.csv` and
`conservative_baseline_anchored75-fixed-daily_power_seed_results.csv`.

## FWER proposition

```latex
\begin{proposition}[Proposal-wise e-alpha-spending]
Suppose that, on the event that candidate $j$ is null, $(E_{j,t})_{t\ge
\tau_j}$ is an e-process in the global filtration, and let
$\gamma_j\in\mathcal F_{\tau_j}$ satisfy $\gamma_j\ge0$ and
$\sum_{j\ge1}\gamma_j\le1$ almost surely.  Promote $j$ at the first declared
inspection $S_j$ for which $E_{j,S_j}\ge(q\gamma_j)^{-1}$.  Then, under
arbitrary serial and cross-candidate dependence,
\[
P\{\text{at least one true null is promoted}\}\le q.
\]
\end{proposition}

\begin{proof}
Conditional Ville inequality at the proposal time gives
$P(S_j<\infty\mid\mathcal F_{\tau_j})\le q\gamma_j$ on the event that
candidate $j$ is null.  A union bound, followed by conditioning at each
proposal time and Tonelli's theorem, therefore yields
\[
P\{\exists j:N_j=1,\ S_j<\infty\}
\le \sum_j E[N_j q\gamma_j]
\le q E\!\left[\sum_j\gamma_j\right]\le q.
\]
No independence step is used.
\end{proof}
```

## Inspection-refinement proposition

```latex
\begin{proposition}[Inspection refinement]
Fix the eligible candidates, raw e-processes, proposal weights, level, and
individual evidence floor.  Run the irreversible self-consistent gate on two
inspection schedules $\mathcal I\subseteq\mathcal I'$, freezing a candidate's
evidence when that run first promotes it.  Then
$D_t^{\mathcal I}\subseteq D_t^{\mathcal I'}$ at every
$t\in\mathcal I$.  Consequently, on daily data, daily inspection weakly
increases discoveries pathwise and weakly decreases the promotion time of
every candidate found by a coarser schedule, while preserving the same
error-control theorem.
\end{proposition}

\begin{proof}
Proceed by induction over the coarse inspection dates.  Immediately before a
common inspection $t$, let $A$ be the fine-schedule discovery set and let
$B$ be the set produced by the coarse gate at $t$; the induction hypothesis
contains the old coarse set in $A$.  Put $C=A\cup B$.  Every member of $A$
passes the floor and its weighted boundary at size $|A|$, hence also at size
$|C|$.  If $j\in B\setminus A$, it has not been frozen by the fine run, so
both runs use the same raw value for $j$ at $t$; that value passes the floor
and the coarse boundary at size $|B|$, hence also the weaker boundary at size
$|C|$.  Thus every member of $C$ passes the fine gate's size-$|C|$ boundary,
so $C$ is feasible.  Maximality of the fine step-up set implies that its new
set contains $C$, and in particular contains $B$.  Extra fine inspections
between common dates only enlarge its persistent set, completing the
induction.
\end{proof}
```
