"""Survivorship audit: were the failures ever in the sample?

Every other module in this framework examines rows that exist. This one asks
which rows exist at all, and that difference makes it the hardest bias to notice:
there is no anomalous value to spot, no correlation that behaves oddly, no test
that fires. The data looks clean because the inconvenient observations were never
loaded.

How the bias gets in
--------------------
A universe assembled from a current constituent list and backfilled through
history contains only entities that made it to the end. Everything that was
delisted, acquired, or wound up is absent -- and those are disproportionately the
entities that performed badly. The backtest is then run on a sample selected, in
part, on the outcome.

The correction is to decide membership by date rather than by present existence:
an entity belongs to the universe on date d if it had listed by d and had not yet
delisted. That is what ``universe_asof`` implements, and reconstructing the
universe this way keeps the failures in the sample for the dates they actually
traded.

What this module measures
-------------------------
Two scores on the same model and the same dates:

* **survivors-only** -- the universe restricted to entities present on the final
  date, which is what backfilling a current constituent list produces.
* **point-in-time universe** -- membership by listing and delisting dates.

The gap is the inflation attributable to survivorship. As with the point-in-time
audit, both arms are scored over the same evaluation dates so the difference
isolates universe composition rather than sampling.

An honest caveat about magnitude
--------------------------------
The size of this effect depends entirely on how delisting relates to the target.
For a *return* target the bias is severe and well documented. For a *volatility*
or *range* target it can go either way: entities heading for delisting often
become more volatile, so excluding them may remove high-volatility observations
rather than low-performing ones, and the direction of the bias is then an
empirical question rather than a foregone conclusion. The module reports the
measured direction rather than assuming it, and the report says which way it went.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..metrics.ic import cross_sectional_ic, demeaned_ic
from ..panel import DATE, ENTITY, Panel

# Below this the difference is indistinguishable from estimation noise.
MATERIAL_GAP = 0.01


@dataclass
class SurvivorshipResult:
    """Comparison of a survivors-only universe against a point-in-time one."""

    survivors_ic: float
    pit_ic: float
    survivors_demeaned_ic: float
    pit_demeaned_ic: float
    n_entities_total: int
    n_entities_surviving: int
    n_entities_delisted: int
    survivor_rate: float
    passed: bool | None
    verdict: str
    detail: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        """Survivors-only score minus point-in-time score, on the demeaned IC."""
        return self.survivors_demeaned_ic - self.pit_demeaned_ic

    def to_dict(self) -> dict:
        return {
            "survivors_ic": self.survivors_ic,
            "pit_ic": self.pit_ic,
            "survivors_demeaned_ic": self.survivors_demeaned_ic,
            "pit_demeaned_ic": self.pit_demeaned_ic,
            "gap": self.gap,
            "n_entities_total": self.n_entities_total,
            "n_entities_surviving": self.n_entities_surviving,
            "n_entities_delisted": self.n_entities_delisted,
            "survivor_rate": self.survivor_rate,
            "passed": self.passed,
            "verdict": self.verdict,
            **self.detail,
        }


def surviving_entities(panel: Panel, tail_dates: int = 1) -> set:
    """Entities observed on the final ``tail_dates`` dates of the panel.

    This reconstructs what a current-constituent-list universe would contain.
    ``tail_dates > 1`` tolerates an entity missing the very last day for reasons
    unrelated to delisting -- a holiday, a data gap -- which would otherwise be
    misclassified as a failure and overstate the bias.
    """
    dates = panel.dates
    if len(dates) == 0:
        return set()
    tail = set(dates[-tail_dates:])
    return set(panel.data.loc[panel.data[DATE].isin(tail), ENTITY].unique())


def delisted_entities(panel: Panel, tail_dates: int = 1) -> set:
    return set(panel.entities) - surviving_entities(panel, tail_dates)


def restrict_to_entities(panel: Panel, entities: set) -> Panel:
    """Panel restricted to the given entities, preserving the train boundary."""
    data = panel.data.loc[panel.data[ENTITY].isin(entities)].reset_index(drop=True)
    if data.empty:
        raise ValueError("restriction left no rows")
    return Panel(data=data, train_end=panel.train_end, label_name=panel.label_name)


def run_survivorship_audit(
    panel: Panel,
    tail_dates: int = 1,
    scope: str = "test",
) -> SurvivorshipResult:
    """Score the same predictions on a survivors-only and a point-in-time universe.

    The panel supplied must already be the point-in-time one -- containing every
    entity for the dates it actually traded. The survivors-only arm is derived
    from it by dropping entities absent at the end, which is the operation a
    backfilled constituent list performs implicitly.
    """
    survivors = surviving_entities(panel, tail_dates)
    delisted = delisted_entities(panel, tail_dates)

    pit_ic = cross_sectional_ic(panel, method="spearman", scope=scope).mean
    pit_dm = demeaned_ic(panel).mean

    n_total = len(panel.entities)
    rate = len(survivors) / n_total if n_total else float("nan")

    if not delisted:
        return SurvivorshipResult(
            survivors_ic=pit_ic,
            pit_ic=pit_ic,
            survivors_demeaned_ic=pit_dm,
            pit_demeaned_ic=pit_dm,
            n_entities_total=n_total,
            n_entities_surviving=len(survivors),
            n_entities_delisted=0,
            survivor_rate=rate,
            passed=None,
            verdict=(
                "NO ATTRITION: every entity is present on the final date, so a "
                "survivors-only universe is identical to the point-in-time one. "
                "This says nothing about a real universe -- a panel assembled "
                "without delisted entities in the first place would look exactly "
                "like this, and the absence would be invisible here. Check that "
                "the source universe was built from listing and delisting dates."
            ),
            detail={"tail_dates": tail_dates},
        )

    surv_panel = restrict_to_entities(panel, survivors)
    surv_ic = cross_sectional_ic(surv_panel, method="spearman", scope=scope).mean
    surv_dm = demeaned_ic(surv_panel).mean
    gap = surv_dm - pit_dm

    if np.isfinite(gap) and gap > MATERIAL_GAP:
        passed = False
        verdict = (
            f"FAIL: restricting to the {len(survivors)} entities present at the "
            f"end raises the demeaned IC by {gap:+.4f} ({surv_dm:+.4f} vs "
            f"{pit_dm:+.4f}). The {len(delisted)} entities that disappeared "
            f"({1 - rate:.1%} of the universe) were harder to predict, and a "
            "backtest that omitted them was scored on a sample selected partly "
            "on outcome."
        )
    elif np.isfinite(gap) and gap < -MATERIAL_GAP:
        passed = True
        verdict = (
            f"PASS (opposite direction): the survivors-only universe scores "
            f"{-gap:+.4f} LOWER. The entities that disappeared were easier to "
            "predict, so excluding them understates performance rather than "
            "flattering it. For a volatility-style target this is plausible -- "
            "entities heading for delisting are often more volatile and more "
            "persistent -- but it is worth confirming the delisting pattern is "
            "what you expect."
        )
    else:
        passed = True
        verdict = (
            f"PASS: dropping the {len(delisted)} non-surviving entities moves the "
            f"demeaned IC by only {gap:+.4f}. Their exclusion would not have "
            "materially changed the reported score on this metric."
        )

    return SurvivorshipResult(
        survivors_ic=surv_ic,
        pit_ic=pit_ic,
        survivors_demeaned_ic=surv_dm,
        pit_demeaned_ic=pit_dm,
        n_entities_total=n_total,
        n_entities_surviving=len(survivors),
        n_entities_delisted=len(delisted),
        survivor_rate=rate,
        passed=passed,
        verdict=verdict,
        detail={"tail_dates": tail_dates, "scope": scope},
    )
