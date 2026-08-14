"""Tests for the group decomposition.

The construction that matters is a prediction which knows only which group an
entity belongs to. Pooled IC rates it highly; within-group IC must rate it at
zero. If the decomposition cannot separate those, it is not measuring anything
the pooled figure does not already say.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit.audits.grouping import MATERIAL_GAP, decompose_by_group
from audit.panel import ENTITY, LABEL, PRED, Panel
from audit.synthetic import generate_panel


def _group_only_panel(offset_scale: float = 3.0, noise: float = 0.01) -> Panel:
    """Prediction encodes the group and nothing else.

    Labels get a large group-level offset, and the prediction is that offset plus
    negligible noise. Within any group the prediction is therefore essentially
    constant and carries no ranking information at all -- while across the pooled
    cross-section it tracks the label closely.
    """
    p, _ = generate_panel(skill=0.5)
    d = p.data.copy()
    offset = d["group"].astype(float) * offset_scale
    rng = np.random.default_rng(0)
    d[LABEL] = d[LABEL] + offset
    d[PRED] = offset + rng.normal(0.0, noise, len(d))
    return Panel(data=d, train_end=p.train_end, label_name=p.label_name)


# ----------------------------------------------------------------------
# The central case
# ----------------------------------------------------------------------
def test_group_only_prediction_scores_high_pooled_and_zero_within():
    res = decompose_by_group(_group_only_panel())
    assert res.pooled_ic > 0.8, "pooled IC should look strong"
    assert abs(res.within_ic_weighted) < 0.1, "within-group ability should be nil"
    assert res.between_ic > 0.9, "the score comes from ranking groups"
    assert res.level_effect > MATERIAL_GAP
    assert res.passed is False


def test_genuine_within_group_skill_survives_decomposition():
    """No false alarm when the grouping is unrelated to the prediction.

    The synthetic generator assigns groups at random, so a skilful prediction
    should score about the same pooled and within-group.
    """
    p, _ = generate_panel(skill=0.5)
    res = decompose_by_group(p)
    assert res.pooled_ic > 0.5
    assert abs(res.level_effect) < MATERIAL_GAP
    assert res.passed is True


def test_level_effect_grows_with_the_group_offset():
    """Dose-response in how strongly groups differ."""
    gaps = []
    for scale in (0.0, 1.0, 3.0):
        res = decompose_by_group(_group_only_panel(offset_scale=scale, noise=0.5))
        gaps.append(res.level_effect)
    assert gaps == sorted(gaps), f"level effect not monotone: {gaps}"


# ----------------------------------------------------------------------
# Averaging choices
# ----------------------------------------------------------------------
def test_weighted_and_unweighted_within_ic_are_both_reported():
    """Their difference signals heterogeneous group quality and must stay visible."""
    res = decompose_by_group(_group_only_panel())
    assert np.isfinite(res.within_ic_weighted)
    assert np.isfinite(res.within_ic_unweighted)


def test_small_groups_are_excluded_not_averaged_in():
    """A group too small to score contributes noise, not information.

    Folding a two-entity group's correlation into the average would let the
    noisiest possible estimate move the headline number.
    """
    p, _ = generate_panel(skill=0.5)
    d = p.data.copy()
    # Carve a two-entity group out of the universe
    tiny = sorted(d[ENTITY].unique())[:2]
    d.loc[d[ENTITY].isin(tiny), "group"] = 99
    panel = Panel(data=d, train_end=p.train_end)

    res = decompose_by_group(panel)
    assert 99 in res.per_group.index
    assert pd.isna(res.per_group.loc[99, "ic"])
    assert "too small" in res.per_group.loc[99, "note"]
    assert res.detail["n_groups_too_small"] >= 1


def test_per_group_table_carries_sizes_for_interpretation():
    res = decompose_by_group(_group_only_panel())
    assert {"n_rows", "typical_cross_section", "ic"} <= set(res.per_group.columns)
    assert (res.per_group["n_rows"] > 0).all()


# ----------------------------------------------------------------------
# Degenerate inputs
# ----------------------------------------------------------------------
def test_single_group_is_inconclusive_not_a_pass():
    """With one group there is no between-group component to separate."""
    p, _ = generate_panel(skill=0.5)
    d = p.data.copy()
    d["group"] = 0
    res = decompose_by_group(Panel(data=d, train_end=p.train_end))
    assert res.n_groups == 1
    assert res.passed is None
    assert "INCONCLUSIVE" in res.verdict


def test_missing_group_column_raises_with_guidance():
    p, _ = generate_panel(skill=0.5)
    d = p.data.drop(columns=["group"])
    with pytest.raises(ValueError, match="grouping key"):
        decompose_by_group(Panel(data=d, train_end=p.train_end))


def test_result_serialises_including_the_per_group_table():
    res = decompose_by_group(_group_only_panel())
    d = res.to_dict()
    for key in ("pooled_ic", "within_ic_weighted", "between_ic", "level_effect", "per_group"):
        assert key in d
    assert isinstance(d["per_group"], list)
    assert len(d["per_group"]) >= 2
