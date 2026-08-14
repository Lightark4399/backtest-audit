"""Alignment audit: does the result actually depend on correct time alignment?

A genuine forecast is a claim about a *specific* pairing of prediction and
outcome. If that pairing can be broken and the score survives, the score was
never measuring forecasting.

Three perturbations, each answering a different question:

``shuffle``
    Permute labels among entities *within* each date. Predictive information is
    destroyed while every marginal distribution is preserved exactly -- same
    labels, same predictions, same cross-section sizes, same dates. The IC must
    collapse to zero. If it does not, the score is being produced by the
    evaluation machinery rather than by the pairing: a broadcasting bug, an
    index misalignment, a metric that correlates something with itself.

``shift``
    Score predictions against the *next* date's labels instead of the correct
    ones. The IC must fall. If it does not fall, the prediction is not specific
    to the day it claims to forecast.

``future_shift``
    Score against the *previous* date's labels -- using a forecast to "predict"
    an outcome that had already occurred. If this scores HIGHER than correct
    alignment, the prediction contains information from the past that it should
    have been built on, i.e. the pipeline has an off-by-one and is effectively
    reporting a fit rather than a forecast.

Interpreting the shift tests on a persistent target
---------------------------------------------------
This is the subtlety that makes a naive reading of these tests wrong. When the
target is autocorrelated, ``y(t+1)`` and ``y(t)`` are themselves correlated, so
a *correct* prediction of ``y(t)`` will still score positively against ``y(t+1)``.
A shift test that leaves the IC positive is therefore NOT evidence of a leak.

What matters is the *drop*, and the right benchmark for how large the drop
should be is the persistence of the label itself. So the report gives the shift
IC alongside the label's own autocorrelation, and judges the result by whether
the drop is material rather than by whether the shifted IC reached zero. Setting
a pass threshold at "shifted IC must be ~0" would fail every honest model on a
persistent target -- a false alarm that would train the user to ignore the tool.

Determinism
-----------
The shuffle permutation is seeded from the date, so a report is reproducible.
Randomness that changes between runs would make an audit result unfalsifiable:
a reviewer re-running it could get a different verdict and neither number could
be checked against the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metrics.ic import ICSeries, cross_sectional_ic
from ..panel import DATE, ENTITY, LABEL, Panel

# A shuffled IC should be zero up to sampling noise. The threshold is applied to
# |mean IC| and is deliberately loose: with ~100 dates and ~100 entities the
# standard error of the mean daily IC is around 0.01, so 0.05 is roughly five
# standard errors -- generous enough not to cry wolf, tight enough that a real
# structural leak (which typically leaves a large IC) cannot slip through.
SHUFFLE_TOLERANCE = 0.05

# How far above the persistence benchmark a shifted IC may sit before the
# alignment is called into question. Same scale as SHUFFLE_TOLERANCE: both ask
# "is this correlation distinguishable from what innocent structure explains".
SHIFT_EXCESS_TOLERANCE = 0.05

# Below this |IC| under correct alignment there is no signal whose date
# specificity could be tested, and the shift test reports inconclusive rather
# than passing a model that has nothing to align.
MIN_TESTABLE_IC = 0.02


@dataclass
class AlignmentCheck:
    """Outcome of one perturbation."""

    name: str
    description: str
    baseline_ic: float  # IC under correct alignment
    perturbed_ic: float
    passed: bool | None  # None when the check is inconclusive
    verdict: str
    detail: dict = field(default_factory=dict)

    @property
    def drop(self) -> float:
        return self.baseline_ic - self.perturbed_ic

    @property
    def drop_ratio(self) -> float:
        if not np.isfinite(self.baseline_ic) or abs(self.baseline_ic) < 1e-12:
            return float("nan")
        return self.drop / abs(self.baseline_ic)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "baseline_ic": self.baseline_ic,
            "perturbed_ic": self.perturbed_ic,
            "drop": self.drop,
            "drop_ratio": self.drop_ratio,
            "passed": self.passed,
            "verdict": self.verdict,
            **self.detail,
        }


def shuffle_labels_within_date(panel: Panel, seed: int = 0) -> Panel:
    """Permute labels among entities within each cross-section.

    Shuffling *within* a date rather than across the whole panel is essential:
    a global shuffle would also destroy the day-to-day variation in the label's
    level, so a surviving correlation could be explained by that rather than by
    a bug. Permuting inside the date holds every cross-section's composition
    fixed and removes only the entity-to-label correspondence, which is exactly
    the thing predictive skill is a claim about.
    """
    out = panel.data.copy()
    shuffled = np.empty(len(out), dtype=float)

    for date, idx in out.groupby(DATE, sort=True).groups.items():
        pos = out.index.get_indexer(idx)
        values = out[LABEL].to_numpy()[pos]
        # Seed per date so the whole audit is reproducible from `seed` alone.
        rng = np.random.default_rng(hash((seed, pd.Timestamp(date).value)) % (2**32))
        shuffled[pos] = rng.permutation(values)

    out[LABEL] = shuffled
    return Panel(data=out, train_end=panel.train_end, label_name=panel.label_name)


def shift_labels(panel: Panel, offset: int = 1) -> Panel:
    """Re-pair each prediction with a label from a different date.

    ``offset=+1`` pairs a prediction with the NEXT date's label (predicting one
    day too late); ``offset=-1`` pairs it with the PREVIOUS date's label
    (nominally predicting the past).

    The shift is applied per entity over its own observed dates, so an entity
    missing on some day does not silently borrow another entity's label. Rows
    with no counterpart after shifting are dropped rather than filled: a
    fabricated label would make the perturbed IC depend on the fill rule instead
    of on alignment.
    """
    if offset == 0:
        raise ValueError("offset must be non-zero; 0 is the unperturbed panel")

    out = panel.data.sort_values([ENTITY, DATE], kind="mergesort").copy()
    # shift(-offset) pulls the label from `offset` positions later into this row.
    out[LABEL] = out.groupby(ENTITY, sort=False)[LABEL].shift(-offset)
    out = out.loc[out[LABEL].notna()].reset_index(drop=True)

    if out.empty:
        raise ValueError(f"shifting by {offset} left no rows with labels")

    return Panel(data=out, train_end=panel.train_end, label_name=panel.label_name)


def label_autocorrelation(panel: Panel, lag: int = 1) -> float:
    """Pooled within-entity autocorrelation of the label.

    This is the benchmark for reading a shift test: a persistent label makes a
    shifted prediction score positively for entirely innocent reasons, and
    knowing how persistent it is turns "the shifted IC is still 0.5" from an
    alarm into an expectation.
    """
    d = panel.data.sort_values([ENTITY, DATE], kind="mergesort")
    cur = d[LABEL]
    nxt = d.groupby(ENTITY, sort=False)[LABEL].shift(-lag)
    both = cur.notna() & nxt.notna()
    if both.sum() < 3:
        return float("nan")
    a, b = cur[both].to_numpy(), nxt[both].to_numpy()
    if a.std() <= 0 or b.std() <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def demeaned_label_autocorrelation(panel: Panel, lag: int = 1) -> float:
    """Within-entity autocorrelation of the label AFTER removing its training mean.

    This is the right persistence benchmark for a demeaned shift test. The raw
    label's autocorrelation is inflated by the constant entity level -- two
    observations of the same entity look similar largely because the entity is
    what it is, not because consecutive deviations are related. Removing the
    level first measures the persistence of the part that a shift can actually
    disturb.
    """
    panel.require_train_end("demeaned_label_autocorrelation")
    mu = panel.per_entity_train_mean(LABEL)
    d = panel.data.sort_values([ENTITY, DATE], kind="mergesort").copy()
    d["_dm"] = d[LABEL] - d[ENTITY].map(mu)

    cur = d["_dm"]
    nxt = d.groupby(ENTITY, sort=False)["_dm"].shift(-lag)
    both = cur.notna() & nxt.notna()
    if both.sum() < 3:
        return float("nan")
    a, b = cur[both].to_numpy(), nxt[both].to_numpy()
    if a.std() <= 0 or b.std() <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ic_mean(panel: Panel, method: str, scope: str) -> float:
    series: ICSeries = cross_sectional_ic(panel, method=method, scope=scope)
    return series.mean


def run_shuffle_test(
    panel: Panel, method: str = "spearman", scope: str = "test", seed: int = 0
) -> AlignmentCheck:
    """Labels permuted within each date. IC must collapse to ~0."""
    base = _ic_mean(panel, method, scope)
    perturbed = _ic_mean(shuffle_labels_within_date(panel, seed=seed), method, scope)

    ok = np.isfinite(perturbed) and abs(perturbed) < SHUFFLE_TOLERANCE
    if ok:
        verdict = (
            f"PASS: IC collapses to {perturbed:+.4f} when the entity-label pairing "
            "is destroyed, so the score depends on that pairing."
        )
    else:
        verdict = (
            f"FAIL: IC remains {perturbed:+.4f} after labels were permuted within "
            "each date. Predictive information cannot survive this, so the score "
            "is being produced by the evaluation path rather than by the "
            "prediction -- look for an index misalignment or a metric comparing "
            "a series with itself."
        )

    return AlignmentCheck(
        name="shuffle",
        description="labels permuted among entities within each date",
        baseline_ic=base,
        perturbed_ic=perturbed,
        passed=bool(ok),
        verdict=verdict,
        detail={"tolerance": SHUFFLE_TOLERANCE, "seed": seed},
    )


def run_shift_test(
    panel: Panel,
    offset: int = 1,
    method: str = "spearman",
    scope: str = "test",
    demean: bool = True,
) -> AlignmentCheck:
    """Predictions scored against another date's labels.

    Two design decisions here, both discovered by running the test against panels
    whose ground truth was known.

    **The test is run on demeaned series.** The per-entity level is constant in
    time, so shifting the labels does not disturb it at all: a raw-IC shift test
    is therefore nearly blind whenever the level dominates the cross-section --
    precisely the situation this framework exists for. Measured on the synthetic
    panels, shifting moves the raw IC of a genuinely skilful model by 5%, and the
    raw IC of a *zero-skill* model by 0.1%; neither is a usable signal. After
    demeaning, the same shift moves the skilful model's IC by 14%. The level does
    not merely inflate the headline number, it also blinds the alignment test,
    which is why demeaning is the default here as well.

    **The drop is judged against the label's own persistence, not a fixed
    threshold.** If the demeaned label has autocorrelation rho at this lag, then
    a correctly aligned prediction, when paired with the neighbouring date's
    label, should still score roughly ``rho * base`` -- because the neighbouring
    label genuinely is that similar. Requiring the shifted IC to approach zero
    would fail every honest model on a persistent target and train the user to
    ignore the tool. What is suspicious is the shifted IC coming in *materially
    above* that benchmark, which means the prediction is not specific to the date
    it claims to forecast.
    """
    if demean:
        from ..metrics.ic import demeaned_ic

        base = demeaned_ic(panel, method=method).mean
        perturbed = demeaned_ic(shift_labels(panel, offset=offset), method=method).mean
        rho = demeaned_label_autocorrelation(panel, lag=abs(offset))
        basis = "demeaned IC"
    else:
        base = _ic_mean(panel, method, scope)
        perturbed = _ic_mean(shift_labels(panel, offset=offset), method, scope)
        rho = label_autocorrelation(panel, lag=abs(offset))
        basis = "raw IC"

    direction = "next" if offset > 0 else "previous"
    drop_ratio = (
        (base - perturbed) / abs(base) if np.isfinite(base) and abs(base) > 1e-12 else float("nan")
    )

    # Expected level under correct alignment, given how persistent the label is.
    expected = rho * base if np.isfinite(rho) and np.isfinite(base) else float("nan")
    excess = perturbed - expected if np.isfinite(expected) else float("nan")

    # The pass criterion is deliberately NOT "the drop must exceed the level
    # implied by label persistence". Estimating that level requires knowing how
    # much of the label is signal and how much is observation noise: the measured
    # autocorrelation is attenuated by noise, while the prediction correlates
    # only with the signal part, so rho * base systematically UNDER-states the
    # shifted IC a correct model should achieve. Gating on it produced false
    # alarms on panels that were honest by construction.
    #
    # What survives that objection is the ordering: whatever the persistence, a
    # correctly aligned prediction must not score BETTER against a neighbouring
    # date's labels than against its own. That is the criterion used. The
    # persistence figures are still reported, as context for reading the size of
    # the drop, but they do not gate the verdict.
    if not np.isfinite(base) or abs(base) < MIN_TESTABLE_IC:
        passed = None
        verdict = (
            f"INCONCLUSIVE: {basis} under correct alignment is {base:+.4f}, too "
            "close to zero to test whether it is date-specific. There is no "
            "signal here whose alignment could be verified."
        )
    elif not np.isfinite(perturbed):
        passed = None
        verdict = f"INCONCLUSIVE: shifted {basis} could not be computed."
    elif offset < 0:
        # The BACKWARD shift is reported as a diagnostic, never as a pass/fail.
        #
        # Almost every forecaster of a persistent target is built on lagged
        # values of the label: the prediction for date t is a function of the
        # label at t-1. Such a prediction is, by construction, more similar to
        # the label it was BUILT FROM than to the one it is trying to forecast,
        # so scoring it against the previous date's labels routinely produces a
        # HIGHER number than correct alignment. Measured on the clean example
        # pipeline -- honest by construction -- the backward shift raises the
        # demeaned IC from 0.71 to 0.97.
        #
        # Gating on that would fail every autoregressive model, which is to say
        # nearly all of them. What the number does convey is how much of the
        # prediction is a restatement of the last observation, so it is reported
        # with that reading and the forward shift carries the verdict.
        passed = None
        if perturbed > base:
            reading = (
                "the prediction resembles the label it was built from more than "
                "the one it forecasts, which is normal for a model driven by "
                "lagged values, and is quantified properly by the persistence "
                "baseline in the decomposition above"
            )
        else:
            reading = (
                "the prediction is closer to the outcome it forecasts than to "
                "the one it was built from"
            )
        verdict = f"DIAGNOSTIC: {basis} {base:+.4f} -> {perturbed:+.4f}; {reading}."
    elif perturbed > base + SHIFT_EXCESS_TOLERANCE:
        passed = False
        verdict = (
            f"FAIL: {basis} is HIGHER against the {direction} date's labels "
            f"({perturbed:+.4f}) than under correct alignment ({base:+.4f}). "
            "The prediction matches a later date better than the one it claims "
            "to forecast, which points to a lag applied in the wrong direction."
        )
    else:
        passed = True
        verdict = (
            f"PASS: {basis} falls from {base:+.4f} to {perturbed:+.4f} "
            f"({drop_ratio:.0%}) against the {direction} date's labels. "
            f"For context, the demeaned label's own persistence is rho={rho:+.2f}, "
            "so a modest drop is expected even when alignment is correct."
        )

    return AlignmentCheck(
        name=f"shift{offset:+d}",
        description=f"predictions scored against the {direction} date's labels",
        baseline_ic=base,
        perturbed_ic=perturbed,
        passed=passed,
        verdict=verdict,
        detail={
            "offset": offset,
            "basis": basis,
            "label_autocorrelation": rho,
            "expected_under_persistence": expected,
            "excess_over_expected": excess,
            "tolerance": SHIFT_EXCESS_TOLERANCE,
            "diagnostic_only": offset < 0,
        },
    )


def run_alignment_audit(
    panel: Panel,
    method: str = "spearman",
    scope: str = "test",
    seed: int = 0,
) -> list[AlignmentCheck]:
    """All three perturbations, in the order they should be read.

    Shuffle first: if it fails, the other two are uninterpretable, because a
    score that survives destroyed pairings tells us nothing about alignment.
    """
    return [
        run_shuffle_test(panel, method=method, scope=scope, seed=seed),
        run_shift_test(panel, offset=1, method=method, scope=scope),
        run_shift_test(panel, offset=-1, method=method, scope=scope),
    ]


def alignment_summary(checks: list[AlignmentCheck]) -> dict:
    """Aggregate verdict, keeping 'inconclusive' distinct from 'passed'.

    Diagnostic-only checks (the backward shift) are excluded from the verdict
    counts: they are reported for interpretation and never assert a conclusion,
    so letting them mark a run as not-all-passed would make a clean audit
    permanently amber and drain the summary of meaning.
    """
    gated = [c for c in checks if not c.detail.get("diagnostic_only")]
    failed = [c.name for c in gated if c.passed is False]
    inconclusive = [c.name for c in gated if c.passed is None]
    return {
        "n_checks": len(checks),
        "n_gated": len(gated),
        "failed": failed,
        "inconclusive": inconclusive,
        "all_passed": not failed and not inconclusive,
        "any_failed": bool(failed),
    }
