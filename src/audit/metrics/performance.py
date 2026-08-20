"""A thin PnL layer: predictions to positions to performance statistics.

Why this exists, given that IC is the better metric for the work
-----------------------------------------------------------------
For model validation, a cross-sectional IC is the more informative statistic: it
is bounded, comparable across targets, and decomposes cleanly into the free and
earned components the rest of this framework separates. A Sharpe ratio does none
of those things.

It is nevertheless the number people read. "Sharpe 3.2 with a 4% drawdown" lands
with a reader who would skim past "demeaned IC 0.0006", including readers whose
approval a piece of research needs. So this layer exists to state the same
finding in the register the audience uses — not because the register is better.

Nothing beneath it changes. The layer consumes a `Panel` and produces summary
statistics; the audit modules continue to operate on predictions and labels and
know nothing about positions.

The construction, and what it deliberately leaves out
-----------------------------------------------------
Positions are cross-sectionally demeaned prediction ranks, scaled to unit gross
exposure per date:

1. Rank predictions within each date
2. Centre the ranks, so the book is dollar-neutral by construction
3. Scale so gross exposure sums to 1

Then `pnl(t) = Σ_i position(i, t) · label(i, t)`.

This is a *scoring device*, not a trading strategy, and the omissions are the
point:

* **No transaction costs.** A realistic cost model would improve the simulation
  and obscure the comparison, since costs depend on turnover, which depends on
  the signal. The framework compares two versions of the same pipeline, and an
  unmodelled cost that applies equally to both cancels.
* **No capacity, borrow, or liquidity constraints.** Same reasoning.
* **No position sizing by conviction.** Rank-based weights make the result depend
  on ordering alone, which is what a cross-sectional IC measures. Sizing by
  predicted magnitude would introduce a second thing being tested.

The consequence is that **the Sharpe ratios here are not achievable returns**.
They are the same information as the IC, in different units, and the report says
so. A number presented as achievable would be the exact overstatement this
project exists to detect.

Raw and demeaned PnL
--------------------
The layer computes both, and the pair is the whole reason it earns its place.

On a persistent, sign-constant target — realized volatility, range, volume — a
long-short book built from prediction ranks is long the high-level names and
short the low-level ones. Because the level barely moves, that book wins *every
single day*. Measured on the synthetic panels, a prediction with **exactly zero**
skill produces an annualised Sharpe of **147** with a 100% hit rate and no
drawdown at all.

That is not a strategy. It is the free component of the target, expressed in
Sharpe units — the same thing raw IC reports, and just as misleading.

Scoring the same positions against **demeaned** labels removes the level and
leaves the PnL that came from predicting deviations. The gap between the two
Sharpe figures is the level effect, stated in the units a reader recognises. A
demo that showed only the first number would be reproducing the deception this
project exists to expose; showing both is the point.

Annualisation
-------------
Sharpe is annualised by `sqrt(periods_per_year)` with a default of 252. That
assumes independent daily returns — and the effective-sample-size machinery in
`metrics/significance.py` exists precisely because that assumption usually fails.
The annualised figure is therefore reported alongside the per-period figure and
the autocorrelation of the PnL series, so the assumption stays visible rather
than being buried in a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metrics.significance import newey_west_tstat
from ..panel import DATE, ENTITY, LABEL, PRED, Panel

TRADING_DAYS = 252


@dataclass
class PerformanceStats:
    """Summary of a PnL series produced from a panel of predictions."""

    n_periods: int
    mean_return: float
    volatility: float
    sharpe: float
    sharpe_annualised: float
    max_drawdown: float
    hit_rate: float
    equity_curve: pd.Series
    pnl: pd.Series
    turnover: float
    sharpe_tstat: float
    sharpe_tstat_hac: float
    pnl_autocorr: float
    effective_n: float
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_periods": self.n_periods,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sharpe_annualised": self.sharpe_annualised,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "turnover": self.turnover,
            "sharpe_tstat": self.sharpe_tstat,
            "sharpe_tstat_hac": self.sharpe_tstat_hac,
            "pnl_autocorr": self.pnl_autocorr,
            "effective_n": self.effective_n,
            **self.detail,
        }


def rank_positions(frame: pd.DataFrame, pred_col: str = PRED) -> pd.Series:
    """Dollar-neutral, unit-gross positions from within-date prediction ranks.

    Returns zeros for a date whose cross-section is too small or whose
    predictions are constant. Zero is the correct position there: with no
    ordering to act on, a scoring device that took a position would be reporting
    the result of an arbitrary tie-break.
    """
    out = pd.Series(0.0, index=frame.index)

    for _, g in frame.groupby(DATE, sort=False):
        if len(g) < 2:
            continue
        p = g[pred_col].to_numpy(float)
        if not np.isfinite(p).all() or np.nanstd(p) <= 0:
            continue

        ranks = pd.Series(p).rank(method="average").to_numpy()
        centred = ranks - ranks.mean()
        gross = np.abs(centred).sum()
        if gross <= 0:
            continue
        out.loc[g.index] = centred / gross

    return out


def compute_pnl(
    panel: Panel,
    scope: str = "test",
    pred_col: str = PRED,
    demean_labels: bool = False,
) -> pd.DataFrame:
    """Per-date PnL and turnover from rank positions.

    ``demean_labels`` subtracts each entity's training-period mean label before
    computing PnL, so the result reflects predicted *deviations* rather than the
    persistent level. See the module docstring: on a sign-constant target the
    undemeaned figure is dominated by the level and flatters a skill-free
    prediction beyond any plausible reading.

    Turnover is the gross change in positions between consecutive dates. It is
    reported but not charged: see the module docstring on why costs are left out
    of a scoring device.
    """
    view = panel.evaluation_view(scope).copy()
    if demean_labels:
        mu = panel.per_entity_train_mean(LABEL)
        view = view.loc[view[ENTITY].map(mu).notna()].copy()
        view[LABEL] = view[LABEL] - view[ENTITY].map(mu)
    view = view.sort_values([DATE, ENTITY], kind="mergesort")
    view["position"] = rank_positions(view, pred_col=pred_col)

    rows = []
    prev: pd.Series | None = None
    for date, g in view.groupby(DATE, sort=True):
        pos = g.set_index(ENTITY)["position"]
        pnl = float((pos * g.set_index(ENTITY)[LABEL]).sum())

        if prev is None:
            turnover = float(pos.abs().sum())
        else:
            aligned = pos.reindex(prev.index.union(pos.index)).fillna(0.0)
            prev_aligned = prev.reindex(aligned.index).fillna(0.0)
            turnover = float((aligned - prev_aligned).abs().sum())
        prev = pos

        rows.append({"date": date, "pnl": pnl, "turnover": turnover, "n": len(g)})

    return pd.DataFrame(rows).set_index("date")


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of the equity curve, as a positive fraction.

    Computed on a cumulative-sum curve rather than a compounded one, consistent
    with the additive PnL this layer produces. Returns 0.0 for a curve that never
    declines.
    """
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    # Guard the early part of the curve, where a peak near zero would make a
    # relative drawdown explode; the denominator is floored at 1.0 since the
    # curve starts there.
    drawdown = (running_max - equity) / running_max.clip(lower=1.0)
    return float(drawdown.max())


def performance(
    panel: Panel,
    scope: str = "test",
    pred_col: str = PRED,
    periods_per_year: int = TRADING_DAYS,
    demean_labels: bool = False,
) -> PerformanceStats:
    """Full performance summary for a panel of predictions."""
    table = compute_pnl(
        panel, scope=scope, pred_col=pred_col, demean_labels=demean_labels
    )
    pnl = table["pnl"]

    if len(pnl) < 2:
        raise ValueError("need at least two dates of PnL to summarise performance")

    mean = float(pnl.mean())
    vol = float(pnl.std(ddof=1))
    sharpe = mean / vol if vol > 0 else float("nan")
    equity = 1.0 + pnl.cumsum()

    sig = newey_west_tstat(pnl)

    return PerformanceStats(
        n_periods=len(pnl),
        mean_return=mean,
        volatility=vol,
        sharpe=sharpe,
        sharpe_annualised=(
            sharpe * np.sqrt(periods_per_year) if np.isfinite(sharpe) else float("nan")
        ),
        max_drawdown=max_drawdown(equity),
        hit_rate=float((pnl > 0).mean()),
        equity_curve=equity,
        pnl=pnl,
        turnover=float(table["turnover"].mean()),
        # The naive t-statistic on the PnL series, and the HAC-corrected one.
        # Reporting both keeps visible the gap that annualising by sqrt(252)
        # silently assumes away.
        sharpe_tstat=sig.naive_tstat,
        sharpe_tstat_hac=sig.hac_tstat,
        pnl_autocorr=sig.lag1_autocorr,
        effective_n=sig.effective_n,
        detail={
            "periods_per_year": periods_per_year,
            "scope": scope,
            "demeaned_labels": demean_labels,
            "mean_cross_section": float(table["n"].mean()),
        },
    )


def compare_performance(
    panels: dict[str, Panel],
    scope: str = "test",
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Performance table for several panels, in both raw and demeaned form.

    Both columns are reported for every panel. The raw Sharpe is what a
    conventional backtest would print; the demeaned Sharpe is what remains once
    the persistent level is removed. Showing only the first would reproduce the
    deception the framework exists to expose.
    """
    rows = {}
    for name, panel in panels.items():
        try:
            raw = performance(panel, scope=scope, periods_per_year=periods_per_year)
            dm = performance(
                panel, scope=scope, periods_per_year=periods_per_year, demean_labels=True
            )
            rows[name] = {
                "sharpe_raw": raw.sharpe_annualised,
                "sharpe_demeaned": dm.sharpe_annualised,
                "hit_rate_raw": raw.hit_rate,
                "hit_rate_demeaned": dm.hit_rate,
                "max_drawdown_demeaned": dm.max_drawdown,
                "turnover": raw.turnover,
                "n_periods": raw.n_periods,
            }
        except Exception as exc:  # a panel that cannot be scored is shown, not fatal
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return pd.DataFrame(rows).T
