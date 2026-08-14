"""Group decomposition: how much of the score comes from ranking across groups?

A cross-sectional IC computed over a mixed universe answers a question nobody
asked. If entities fall into groups with systematically different levels --
sectors, exchanges, market-cap bands, asset classes -- then a prediction that
merely identifies which group an entity belongs to will rank the full
cross-section well while having no ability to rank *within* any group.

That distinction is not academic. A forecast is used inside a group far more
often than across one: which of these tech names, which of these small caps,
which of these bonds. A score that only works across groups will not survive
contact with that use.

The decomposition
-----------------
Three numbers on the same predictions:

``pooled``
    IC over the whole cross-section, the conventionally reported figure.

``within-group``
    IC computed inside each group separately, then averaged across groups
    weighted by group size. This is the usable ability.

``between-group``
    IC of the group-average prediction against the group-average label. This is
    the part that comes from ranking groups, which a group dummy would capture
    for free.

The gap between pooled and within-group is the level effect, expressed the same
way as everything else in the report.

Why weighting by size and not equally
--------------------------------------
An unweighted average lets a group of four entities count as much as a group of
four hundred. Since small groups produce noisy per-date correlations, that would
make the within-group figure dominated by the noisiest estimates. Size weighting
answers "how well does this rank a typical entity within its own group", which is
the question the number is used to answer. The unweighted figure is reported
alongside so the difference between the two -- itself a sign of heterogeneous
group quality -- stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metrics.ic import MIN_CROSS_SECTION, cross_sectional_ic
from ..panel import DATE, ENTITY, LABEL, PRED, Panel

# Below this the pooled and within-group figures agree to within noise.
MATERIAL_GAP = 0.02


@dataclass
class GroupDecomposition:
    """Pooled, within-group and between-group views of the same predictions."""

    pooled_ic: float
    within_ic_weighted: float
    within_ic_unweighted: float
    between_ic: float
    per_group: pd.DataFrame
    n_groups: int
    group_column: str
    passed: bool | None
    verdict: str
    detail: dict = field(default_factory=dict)

    @property
    def level_effect(self) -> float:
        """Pooled minus within-group: the part attributable to ranking groups."""
        return self.pooled_ic - self.within_ic_weighted

    def to_dict(self) -> dict:
        return {
            "pooled_ic": self.pooled_ic,
            "within_ic_weighted": self.within_ic_weighted,
            "within_ic_unweighted": self.within_ic_unweighted,
            "between_ic": self.between_ic,
            "level_effect": self.level_effect,
            "n_groups": self.n_groups,
            "group_column": self.group_column,
            "passed": self.passed,
            "verdict": self.verdict,
            "per_group": self.per_group.reset_index().to_dict(orient="records"),
            **self.detail,
        }


def _within_group_ic(
    panel: Panel, group_col: str, method: str, scope: str
) -> tuple[pd.DataFrame, float, float]:
    """Per-group IC, plus size-weighted and unweighted averages.

    Each group is scored through ``cross_sectional_ic`` on a panel restricted to
    it, so a group's IC is computed exactly the way the pooled figure is. Using a
    different routine for the parts than for the whole would make the
    decomposition non-comparable with the number it is decomposing.
    """
    rows = []
    view = panel.evaluation_view(scope)

    for group, g in view.groupby(group_col, sort=True):
        # A group must be large enough on a typical date for a within-date
        # correlation to mean anything; below that the estimate is noise and is
        # reported as such rather than folded into the average.
        typical_size = g.groupby(DATE)[ENTITY].size().median()
        if typical_size < MIN_CROSS_SECTION:
            rows.append(
                {
                    "group": group,
                    "n_rows": len(g),
                    "typical_cross_section": float(typical_size),
                    "ic": float("nan"),
                    "note": "cross-section too small to estimate",
                }
            )
            continue

        sub = Panel(
            data=panel.data.loc[panel.data[group_col] == group].reset_index(drop=True),
            train_end=panel.train_end,
            label_name=panel.label_name,
        )
        series = cross_sectional_ic(sub, method=method, scope=scope)
        rows.append(
            {
                "group": group,
                "n_rows": len(g),
                "typical_cross_section": float(typical_size),
                "ic": series.mean,
                "note": "",
            }
        )

    table = pd.DataFrame(rows).set_index("group")
    usable = table.loc[table["ic"].notna()]
    if usable.empty:
        return table, float("nan"), float("nan")

    weights = usable["n_rows"].to_numpy(float)
    ics = usable["ic"].to_numpy(float)
    weighted = float(np.average(ics, weights=weights))
    unweighted = float(np.mean(ics))
    return table, weighted, unweighted


def _between_group_ic(panel: Panel, group_col: str, method: str, scope: str) -> float:
    """IC of group-average prediction against group-average label, per date.

    This is the score a predictor would achieve if it knew only which group each
    entity belonged to -- the free component that a group dummy supplies.
    """
    view = panel.evaluation_view(scope)
    agg = (
        view.groupby([DATE, group_col])[[PRED, LABEL]]
        .mean()
        .reset_index()
        .rename(columns={group_col: ENTITY})
    )
    if agg.empty:
        return float("nan")
    tmp = Panel(data=agg, train_end=panel.train_end, label_name=panel.label_name)
    return cross_sectional_ic(tmp, method=method, scope="all").mean


def decompose_by_group(
    panel: Panel,
    group_col: str = "group",
    method: str = "spearman",
    scope: str = "test",
) -> GroupDecomposition:
    """Split the cross-sectional IC into within-group and between-group parts."""
    if group_col not in panel.data.columns:
        raise ValueError(
            f"column {group_col!r} not in the panel. Group decomposition needs a "
            "grouping key (sector, exchange, size band) carried alongside the "
            "predictions."
        )

    pooled = cross_sectional_ic(panel, method=method, scope=scope).mean
    table, within_w, within_u = _within_group_ic(panel, group_col, method, scope)
    between = _between_group_ic(panel, group_col, method, scope)
    n_groups = int(table["ic"].notna().sum())

    gap = pooled - within_w

    if n_groups < 2:
        passed = None
        verdict = (
            f"INCONCLUSIVE: only {n_groups} group has a cross-section large "
            "enough to score. With one group there is no between-group component "
            "to separate, so the pooled figure is already a within-group one."
        )
    elif not np.isfinite(gap):
        passed = None
        verdict = "INCONCLUSIVE: the within-group figure could not be computed."
    elif gap > MATERIAL_GAP:
        passed = False
        verdict = (
            f"FAIL: pooled IC is {pooled:+.4f} but the size-weighted within-group "
            f"IC is only {within_w:+.4f}, a gap of {gap:+.4f}. Much of the score "
            f"comes from ranking {n_groups} groups against each other "
            f"(between-group IC {between:+.4f}), which a group dummy would supply "
            "for free. Inside a group -- where a forecast is normally used -- the "
            "prediction is materially weaker than the headline suggests."
        )
    else:
        passed = True
        verdict = (
            f"PASS: within-group IC ({within_w:+.4f}) is close to pooled "
            f"({pooled:+.4f}). The ranking ability survives inside groups, so it "
            "is not an artefact of group-level differences."
        )

    return GroupDecomposition(
        pooled_ic=pooled,
        within_ic_weighted=within_w,
        within_ic_unweighted=within_u,
        between_ic=between,
        per_group=table,
        n_groups=n_groups,
        group_column=group_col,
        passed=passed,
        verdict=verdict,
        detail={
            "method": method,
            "scope": scope,
            "material_gap_threshold": MATERIAL_GAP,
            "n_groups_too_small": int(table["ic"].isna().sum()),
        },
    )
