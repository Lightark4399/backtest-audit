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
skill produces a Sharpe of **147** annualised at 252 periods per year, with a
100% hit rate and a score series that never declines at all.

That is not a strategy. It is the free component of the target, expressed in
Sharpe units — the same thing raw IC reports, and just as misleading.

Scoring the same positions against **demeaned** labels removes the level and
leaves the PnL that came from predicting deviations. The gap between the two
Sharpe figures is the level effect, stated in the units a reader recognises. A
demo that showed only the first number would be reproducing the deception this
project exists to expose; showing both is the point.

Annualisation
-------------
`periods_per_year` has no default. The panel contract is
`(entity_id, event_date, prediction, label)` and says nothing about observation
frequency, so a hardcoded 252 would silently assert daily data about a panel
that might be weekly or monthly. Without it, only the per-period Sharpe is
reported and the annualised figure is stated as unavailable — which is not the
same claim as zero, and not the same claim as a computation that failed.

When a caller does supply it, `sqrt(periods_per_year)` scaling assumes i.i.d.
returns. Under autocorrelation the error has a direction that depends on its
sign: positive autocorrelation inflates the annualised figure, negative
autocorrelation deflates it (Lo, 2002, "The Statistics of Sharpe Ratios",
*Financial Analysts Journal* 58(4), 36-52). It is therefore not an upper bound —
that would only hold for one sign. The per-period figure, the lag-1
autocorrelation and the HAC t-statistic are all reported alongside it so the
assumption stays visible rather than being buried in a constant.

What the score series is
------------------------
`pnl` accumulates into a series that starts at 1.0, but it is a *score*, not
capital: positions are unit-gross by construction and nothing compounds. Its
peak-to-trough decline is therefore reported in the units of that series, as
`additive_peak_to_trough`, and never as a percentage drawdown of invested
capital — a quantity this layer has no basis to compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metrics.significance import newey_west_tstat
from ..panel import DATE, ENTITY, LABEL, PRED, Panel

# Offered for callers that know their panel is daily; never applied by default.
TRADING_DAYS = 252


@dataclass
class PerformanceStats:
    """Summary of a PnL series produced from a panel of predictions."""

    n_periods: int
    mean_return: float
    volatility: float
    sharpe: float
    # None when no observation frequency was supplied: not applicable rather
    # than zero, and distinct from a NaN produced by a failed computation.
    sharpe_annualised: float | None
    additive_peak_to_trough: float
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
            "sharpe_per_period": self.sharpe,
            "sharpe_annualised": self.sharpe_annualised,
            "additive_peak_to_trough": self.additive_peak_to_trough,
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


def additive_peak_to_trough(score_curve: pd.Series) -> float:
    """Largest peak-to-trough decline of the additive score series, in its units.

    This is deliberately *not* a maximum drawdown. Positions here are
    dollar-neutral and unit-gross and nothing is compounded, so the series this
    reads is a cumulative score rather than the net asset value of an invested
    book. Dividing by a running peak would express the decline as a fraction of
    a quantity that is not capital, and the result can exceed 100% -- as it did,
    reading 186.8%, on the demo's level-only panel. A reader who saw that would
    reasonably understand a conventional equity-curve drawdown, which it is not.

    So the decline is reported as an absolute difference in score units. That is
    invariant to the arbitrary 1.0 the curve starts at, and it needs no floor on
    the denominator, since there is no denominator.

    A caller who has real capital and a compounded NAV can compute a
    conventional maximum drawdown from it and report that under a separate key.
    This layer does not, because it has no capital to speak of.

    Returns 0.0 for a curve that never declines.
    """
    if score_curve.empty:
        return float("nan")
    return float((score_curve.cummax() - score_curve).max())


def annualisation_label(periods_per_year: int | None) -> str:
    """How ``annualise_sharpe`` scales, as text, so a label cannot disagree with it.

    Every renderer takes its factor string from here rather than composing one.
    Four descriptions of this single fact had already drifted apart -- a lag
    table headed "ann. Sharpe", a comparison table headed "raw (ann.)", a block
    headed "annualised x 252" and a sentence reading "annualised at 252" -- all
    of them describing one multiplication by sqrt(252). Two named no factor and
    two named the period count as though it were the multiplier.

    Sharing the string is what makes them agree. An assertion that each names
    *a* factor cannot tell whether it named the right one, and the third and
    fourth of those survived exactly such an assertion.
    """
    if periods_per_year is None:
        return "not annualised"
    return f"sqrt {periods_per_year}"


def annualise_sharpe(sharpe: float, periods_per_year: int | None) -> float | None:
    """Scale a per-period Sharpe by ``sqrt(periods_per_year)``.

    Returns ``None`` when no frequency was supplied. That is a different claim
    from 0.0 and from NaN: the figure is not applicable, rather than nil or
    uncomputable. See the module docstring on why there is no default, and on
    the direction of the i.i.d. error under autocorrelation.

    There is deliberately no upper clamp, and no threshold above which the
    figure is suppressed as implausible -- the demo's zero-skill panel annualises
    to 147 and that number is printed. Incident 12 in ``AI_NOTES.md`` is that
    same figure: the temptation was to treat it as a bug and adjust until it
    looked reasonable, and it was correct. Suppression would also gate on
    "implausibly large", which cannot be estimated without bias from the data at
    hand, so by the repository's own rule it may be context but never a gate --
    and the gate would sit on the path real panels take, hiding the diagnostic
    exactly when a pipeline error makes it fire hardest. Callers who need a
    plausibility judgement should make it against the raw/demeaned pair, which
    is where the information is.
    """
    if periods_per_year is None:
        return None
    if not np.isfinite(sharpe):
        return float("nan")
    return float(sharpe * np.sqrt(periods_per_year))


def performance(
    panel: Panel,
    scope: str = "test",
    pred_col: str = PRED,
    periods_per_year: int | None = None,
    demean_labels: bool = False,
) -> PerformanceStats:
    """Full performance summary for a panel of predictions.

    ``periods_per_year`` has no default: the panel contract does not carry an
    observation frequency, so one cannot be assumed. Supplied, it adds an
    annualised Sharpe alongside the per-period figure; omitted, the annualised
    figure is ``None`` and the report says so explicitly.
    """
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
        sharpe_annualised=annualise_sharpe(sharpe, periods_per_year),
        additive_peak_to_trough=additive_peak_to_trough(equity),
        hit_rate=float((pnl > 0).mean()),
        equity_curve=equity,
        pnl=pnl,
        turnover=float(table["turnover"].mean()),
        # The naive t-statistic on the PnL series, and the HAC-corrected one.
        # Reporting both keeps visible the gap that sqrt-T annualisation
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
    periods_per_year: int | None = None,
) -> pd.DataFrame:
    """Performance table for several panels, in both raw and demeaned form.

    Both columns are reported for every panel. The raw Sharpe is what a
    conventional backtest would print; the demeaned Sharpe is what remains once
    the persistent level is removed. Showing only the first would reproduce the
    deception the framework exists to expose.

    Sharpe columns are per period and named as such. Annualised columns appear
    only when ``periods_per_year`` is supplied, so a reader can never mistake
    one for the other by reading a bare ``sharpe_raw`` heading.
    """
    rows = {}
    for name, panel in panels.items():
        try:
            raw = performance(panel, scope=scope, periods_per_year=periods_per_year)
            dm = performance(
                panel, scope=scope, periods_per_year=periods_per_year, demean_labels=True
            )
            rows[name] = {
                "sharpe_raw_per_period": raw.sharpe,
                "sharpe_demeaned_per_period": dm.sharpe,
                "hit_rate_raw": raw.hit_rate,
                "hit_rate_demeaned": dm.hit_rate,
                "additive_peak_to_trough_demeaned": dm.additive_peak_to_trough,
                "turnover": raw.turnover,
                "n_periods": raw.n_periods,
            }
            if periods_per_year is not None:
                rows[name]["sharpe_raw_annualised"] = raw.sharpe_annualised
                rows[name]["sharpe_demeaned_annualised"] = dm.sharpe_annualised
        except Exception as exc:  # a panel that cannot be scored is shown, not fatal
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return pd.DataFrame(rows).T
