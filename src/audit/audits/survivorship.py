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
delisted.

How this module reconstructs the universe, and what that costs
--------------------------------------------------------------
It does not consult a listing and delisting calendar. It derives membership from
the panel it was given: the tail window is the panel's final ``tail_dates``
dates (the last date by default), an entity counts as absent when it appears on
none of them, and the survivors-only arm is that panel with those entities
dropped. Absence from the tail window is not verified delisting.

The consequence is a blind spot, and it is the worst-placed one available. **On
a panel backfilled from a survivor list this audit reports no attrition** --
because the entities that failed were never in the panel to be missing from its
tail window -- and that is precisely the data whose survivorship bias most needs
catching. The module says so in its own verdict rather than leaving the reader
to infer it: the NO ATTRITION result states that a universe assembled without
its delisted entities looks exactly like an honest one from here, and asks the
reader to check how the source universe was built.

A calendar-based reconstruction has no such blind spot. The SQL macro
``universe_asof`` in ``sql/duckdb/001_schema.sql``, reachable through
``BitemporalStore.universe()``, implements one -- but this module does not call
it, and nothing in the audit pipeline does. It is available to callers who
already hold a bitemporal store. ``PLAN.md`` records what would make wiring it
worth doing, and ``CROSS_REPO_AUDIT.md`` records why it was left unwired here.

What this module measures
-------------------------
Two scores of the same model over the same nominal evaluation window:

* **survivors-only** -- the panel restricted to entities seen at least once in
  the tail window. This is a filter on the input panel, not a reconstruction
  of a current constituent list: with ``tail_dates > 1`` an entity absent on
  the last date is still kept if it appears earlier in the window.
* **as-supplied panel** -- the panel exactly as supplied. It is point-in-time
  only if the caller built it from dated membership; this module neither checks
  nor reconstructs that.

The gap is the sensitivity of the score to restricting the input panel to the
entities seen in its tail window. It reads as survivorship bias only once the
caller has established how the input panel's membership was built; this module
cannot establish that.

Both arms cover the same nominal evaluation window, but each IC is computed
independently, and a date is scored only where that arm's cross-section is
usable (enough entities, not constant). The dates each arm actually scores need
not coincide, so the gap can mix a composition difference with a difference in
scorable dates. That boundary is recorded, not corrected, here.

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

# Fixed diagnostic cut-off on survivors-only minus as-supplied demeaned IC:
# above +MATERIAL_GAP is FAIL, anything else finite is PASS, a non-finite gap is
# INCONCLUSIVE. A chosen materiality level, not calibrated to sampling error and
# not a significance bound.
MATERIAL_GAP = 0.01


@dataclass
class SurvivorshipResult:
    """Comparison of a survivors-only universe against the panel as supplied.

    Field names are kept for compatibility and read as follows: ``pit_*`` hold
    the as-supplied panel's scores, point-in-time only if the caller built it
    that way; ``survivors_*`` hold the scores of the entities seen in the tail
    window (the final ``tail_dates`` dates); ``n_entities_delisted`` counts
    entities absent from the whole tail window, not verified delistings;
    ``gap`` is survivors minus as-supplied.
    """

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
        """Survivors-only score minus as-supplied score, on the demeaned IC."""
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


def tail_window_phrases(tail_dates: int) -> tuple[str, str]:
    """How the tail window reads in a sentence: (present ..., absent ...)."""
    if tail_dates == 1:
        return "on the final date", "absent on the final date"
    return (
        f"at least once in the final {tail_dates} dates",
        f"absent from all of the final {tail_dates} dates",
    )


def surviving_entities(panel: Panel, tail_dates: int = 1) -> set:
    """Entities observed at least once on the final ``tail_dates`` dates of the panel.

    A filter on the input panel, not a reconstruction of a current constituent
    list. ``tail_dates > 1`` keeps an entity missing the very last day -- for a
    holiday, a data gap, or any other reason -- if it appears earlier in the
    window.
    """
    dates = panel.dates
    if len(dates) == 0:
        return set()
    # A slice with 0, a negative N or N beyond the panel would not select "the
    # final N dates" that the verdict and report describe.
    if not 1 <= tail_dates <= len(dates):
        raise ValueError(
            f"tail_dates must be between 1 and the panel's {len(dates)} dates; "
            f"got {tail_dates}"
        )
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
    """Score the same predictions on a survivors-only panel and the panel as supplied.

    The survivors-only arm is the panel supplied with the entities absent from
    the tail window dropped. Both arms cover the same nominal evaluation window;
    the dates each can actually score need not coincide. The gap measures the
    score's sensitivity to that filter, and reads as survivorship bias only if
    the panel supplied was built from historical membership of a defined
    research universe.
    """
    present, absent = tail_window_phrases(tail_dates)
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
                f"NO ATTRITION: every entity is present {present}, so a "
                "survivors-only universe is identical to the panel as supplied. "
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

    if not np.isfinite(gap):
        # After the no-attrition path, which returns early: with nothing absent
        # at the end there is no survivors-only arm to compare.
        undefined = [
            name
            for name, v in (("survivors-only", surv_dm), ("as-supplied", pit_dm))
            if not np.isfinite(v)
        ]
        passed = None
        verdict = (
            f"INCONCLUSIVE: the {' and '.join(undefined)} demeaned IC could not be "
            f"computed, so there is no gap to compare despite {len(delisted)} "
            f"entities {absent}. A demeaned IC is undefined when no "
            "evaluation date has a usable cross-section (too few entities, or no "
            "variation). This is neither a zero gap nor a gap within the threshold."
        )
    elif gap > MATERIAL_GAP:
        passed = False
        verdict = (
            f"FAIL: the {len(survivors)} entities present {present} score "
            f"{surv_dm:+.4f} on the demeaned IC against {pit_dm:+.4f} for the "
            f"panel as supplied, a gap of {gap:+.4f}, above the fixed "
            f"{MATERIAL_GAP} materiality threshold. {len(delisted)} entities "
            f"({1 - rate:.1%}) are {absent}; absence is not "
            "verified delisting. The gap alone does not show why those entities "
            "score differently or that the sample was selected on outcome; it "
            "reads as survivorship bias only if the input panel was built from "
            "dated membership."
        )
    elif gap < -MATERIAL_GAP:
        passed = True
        verdict = (
            f"PASS (opposite direction): the entities present {present} "
            f"score {-gap:.4f} LOWER on the demeaned IC than the panel as "
            f"supplied ({surv_dm:+.4f} vs {pit_dm:+.4f}), beyond the fixed "
            f"{MATERIAL_GAP} materiality threshold. The gap alone does not show "
            "why the absent entities score differently, and absence from the "
            "tail window is not verified delisting."
        )
    else:
        passed = True
        verdict = (
            f"PASS: the entities present {present} and the panel as "
            f"supplied differ by {gap:+.4f} on the demeaned IC, within the fixed "
            f"{MATERIAL_GAP} materiality threshold -- a diagnostic cut-off, not a "
            f"significance test. {len(delisted)} entities {absent} "
            "are excluded from the first figure."
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
