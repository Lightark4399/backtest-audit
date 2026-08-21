"""Execution-timing audit: could the signal have been traded when it was scored?

A signal computed from Friday's close cannot be traded at Friday's close. The
close is the last observable price of the session; by the time it exists, the
opportunity to transact at it has passed. A backtest that assumes otherwise is
not slightly optimistic — it is measuring a trade that could not have happened.

The check
---------
Re-score the same predictions against returns realised over progressively later
windows:

* ``lag=0`` — trade at the same close the signal was computed from. Impossible.
* ``lag=1`` — trade at the next available price. The honest assumption for a
  close-computed daily signal.
* ``lag=2`` — one further period of delay, as a robustness check.

Three profiles, three conclusions. The third was added after testing, because the
first version of this module treated the *healthiest* profile as unreadable:

**Concentrated at lag 0, collapsing at lag 1.** The signal was reading the
outcome, not forecasting it. Direct evidence of look-ahead, and the clearest
diagnostic in the framework because the mechanism is unambiguous: the only thing
that changed was *when* the trade happened.

**Nothing at lag 0, ability at lag 1.** The best possible profile, and easy to
mistake for a failure. The lag-0 return is the one contemporaneous with the
signal's own computation — a pure forecast should have *no* relationship with it,
because that return has already happened. Ability appearing only once execution
moves forward is exactly what forecasting looks like.

**Gradual decay across lags.** Normal for a signal with some contemporaneous
overlap. Any real signal decays as execution is delayed, since the market
consumes part of the information in the interim. What matters is whether it
survives the delay it would actually face.

Why this is not the alignment audit
-----------------------------------
The alignment audit asks whether prediction and label are paired to the correct
date. This one accepts that the pairing is right and asks a different question:
given that the signal describes date t, was there still a tradeable price left on
date t after the signal existed? A pipeline can be perfectly aligned and still
assume an impossible fill.

Requirement
-----------
Unlike every other module, this one needs a **return series** rather than the
framework's generic label — the question is about execution, which only makes
sense against something tradeable. Panels carrying a non-return target skip the
audit rather than producing a meaningless number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metrics.ic import MIN_CROSS_SECTION, _corr
from ..metrics.performance import performance
from ..panel import DATE, ENTITY, LABEL, PRED, Panel

# A lag-1 score below this fraction of the lag-0 score means the result depended
# on an execution that was not available.
COLLAPSE_RATIO = 0.25

# Below this |IC| a score is indistinguishable from noise on a panel of this size.
MIN_TESTABLE_IC = 0.02


@dataclass
class LagResult:
    """Score achieved when execution is delayed by a given number of periods."""

    lag: int
    ic: float
    sharpe: float
    n_dates: int
    n_rows: int

    def to_dict(self) -> dict:
        return {
            "lag": self.lag,
            "ic": self.ic,
            "sharpe": self.sharpe,
            "n_dates": self.n_dates,
            "n_rows": self.n_rows,
        }


@dataclass
class ExecutionTimingResult:
    """Decay profile of a signal as execution is delayed."""

    results: list[LagResult]
    passed: bool | None
    verdict: str
    return_column: str
    detail: dict = field(default_factory=dict)

    def at_lag(self, lag: int) -> LagResult | None:
        return next((r for r in self.results if r.lag == lag), None)

    @property
    def decay_ratio(self) -> float:
        """Fraction of the lag-0 IC that survives to lag 1."""
        a, b = self.at_lag(0), self.at_lag(1)
        if a is None or b is None or not np.isfinite(a.ic) or abs(a.ic) < 1e-12:
            return float("nan")
        return b.ic / a.ic

    def to_dict(self) -> dict:
        return {
            "lags": [r.to_dict() for r in self.results],
            "decay_ratio": self.decay_ratio,
            "passed": self.passed,
            "verdict": self.verdict,
            "return_column": self.return_column,
            **self.detail,
        }


def shift_returns(panel: Panel, return_col: str, lag: int) -> pd.DataFrame:
    """Pair each prediction with the return realised ``lag`` periods later.

    Shifted per entity over its own observed dates, so an entity missing a day
    does not silently borrow another's return. Rows with no counterpart are
    dropped rather than filled: a fabricated return would make the result depend
    on the fill rule instead of on the execution assumption.
    """
    d = panel.data.sort_values([ENTITY, DATE], kind="mergesort").copy()
    if lag == 0:
        d["_ret"] = d[return_col]
    else:
        d["_ret"] = d.groupby(ENTITY, sort=False)[return_col].shift(-lag)
    return d.loc[d["_ret"].notna()].reset_index(drop=True)


def _score_lag(panel: Panel, return_col: str, lag: int, scope: str) -> LagResult:
    shifted = shift_returns(panel, return_col, lag)
    if shifted.empty:
        return LagResult(lag=lag, ic=float("nan"), sharpe=float("nan"), n_dates=0, n_rows=0)

    tmp = Panel(data=shifted, train_end=panel.train_end, label_name=panel.label_name)
    view = tmp.evaluation_view(scope)

    ics = []
    for _, g in view.groupby(DATE, sort=True):
        if len(g) < MIN_CROSS_SECTION:
            continue
        c = _corr(g[PRED].to_numpy(float), g["_ret"].to_numpy(float), "spearman")
        if c is not None:
            ics.append(c)

    # Sharpe comes from the shared PnL layer with the shifted return standing in
    # as the label, so the execution comparison uses exactly the same position
    # construction as the rest of the framework.
    pnl_frame = shifted.copy()
    pnl_frame[LABEL] = pnl_frame["_ret"]
    try:
        sharpe = performance(
            Panel(data=pnl_frame, train_end=panel.train_end), scope=scope
        ).sharpe_annualised
    except Exception:
        sharpe = float("nan")

    return LagResult(
        lag=lag,
        ic=float(np.mean(ics)) if ics else float("nan"),
        sharpe=sharpe,
        n_dates=len(ics),
        n_rows=len(view),
    )


def audit_execution_timing(
    panel: Panel,
    return_col: str = "forward_return",
    lags: tuple[int, ...] = (0, 1, 2),
    scope: str = "test",
) -> ExecutionTimingResult:
    """Score the same signal under progressively delayed execution.

    ``return_col`` must name a column of realised returns. The default follows
    the common convention; a panel without it skips the audit, since delaying
    execution against a non-tradeable target measures nothing.
    """
    if return_col not in panel.data.columns:
        raise ValueError(
            f"column {return_col!r} not found. The execution-timing audit needs a "
            "return series, because the question is whether a trade could have "
            "happened -- which only has meaning against something tradeable."
        )

    results = [_score_lag(panel, return_col, lag, scope) for lag in lags]
    comp = ExecutionTimingResult(
        results=results, passed=None, verdict="", return_column=return_col
    )

    zero, one = comp.at_lag(0), comp.at_lag(1)
    ratio = comp.decay_ratio
    later = [r for r in results if r.lag > 0 and np.isfinite(r.ic)]
    best_later = max((r.ic for r in later), default=float("nan"))

    if zero is None or one is None or not np.isfinite(zero.ic) or not np.isfinite(one.ic):
        comp.passed = None
        comp.verdict = "INCONCLUSIVE: the lag-0 and lag-1 scores could not both be computed."
    elif (
        abs(zero.ic) < MIN_TESTABLE_IC
        and np.isfinite(best_later)
        and best_later > MIN_TESTABLE_IC
    ):
        # The healthiest profile there is, and the one a naive reading of a
        # "decay test" would mistake for a failure to decay.
        comp.passed = True
        comp.verdict = (
            f"PASS (clean forecast profile): the signal has essentially no "
            f"relationship with the contemporaneous return ({zero.ic:+.4f} at "
            f"lag 0) and only becomes informative once execution moves forward "
            f"({one.ic:+.4f} at lag 1). The lag-0 return had already happened "
            "when the signal was computed, so having no edge on it is precisely "
            "what a forecast, as opposed to a reading, looks like."
        )
    elif abs(zero.ic) < MIN_TESTABLE_IC:
        comp.passed = None
        comp.verdict = (
            f"INCONCLUSIVE: the signal scores {zero.ic:+.4f} at lag 0 and no "
            "later lag is informative either. There is no decay profile to read."
        )
    elif ratio < COLLAPSE_RATIO:
        comp.passed = False
        comp.verdict = (
            f"FAIL: IC falls from {zero.ic:+.4f} at lag 0 to {one.ic:+.4f} at "
            f"lag 1 -- only {ratio:.0%} survives a single period of delay. A "
            "signal computed from a close cannot trade at that close, so the "
            "lag-0 figure describes a fill that was never available. What "
            "remains at lag 1 is what the signal is actually worth."
        )
    else:
        comp.passed = True
        comp.verdict = (
            f"PASS: IC falls from {zero.ic:+.4f} to {one.ic:+.4f} "
            f"({ratio:.0%} retained) when execution moves one period later. The "
            "signal survives the delay it would actually face; some decay is "
            "expected, since the market consumes part of the information in the "
            "interim."
        )

    comp.detail = {
        "collapse_ratio_threshold": COLLAPSE_RATIO,
        "scope": scope,
        "lags_tested": list(lags),
    }
    return comp
