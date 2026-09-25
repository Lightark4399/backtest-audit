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
from audit.report.text import WIDTH, format_group_decomposition
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


def test_within_above_pooled_is_not_described_as_close_to_pooled():
    """Opposite direction: the prediction ranks groups backwards but entities
    within a group correctly, so within-group IC exceeds pooled by far more than
    the threshold. The result stays a PASS (unchanged behaviour) but the verdict
    must not claim the two figures are close."""
    p, _ = generate_panel(skill=0.5)
    d = p.data.copy()
    offset = d["group"].astype(float) * 3.0
    d[LABEL] = d[LABEL] + offset
    d[PRED] = d[PRED] - offset
    res = decompose_by_group(Panel(data=d, train_end=p.train_end, label_name=p.label_name))
    assert res.level_effect < -MATERIAL_GAP
    assert res.passed is True
    assert "opposite direction" in res.verdict
    assert "close to pooled" not in res.verdict


def test_within_threshold_pass_names_the_fixed_threshold():
    p, _ = generate_panel(skill=0.5)
    res = decompose_by_group(p)
    assert abs(res.level_effect) < MATERIAL_GAP
    assert "close to pooled" in res.verdict
    assert "not a significance test" in res.verdict


def test_within_above_pooled_with_positive_between_ic_infers_no_mechanism():
    """Within > pooled does not mean ranking groups works against the prediction.

    The prediction carries a small positive share of the group offset, so it
    ranks groups the right way (between-group IC clearly positive) while pooled
    IC still falls below within-group IC. The verdict must state the figures and
    not claim the between-group component opposes the prediction.
    """
    p, _ = generate_panel(skill=0.5)
    d = p.data.copy()
    offset = d["group"].astype(float) * 3.0
    d[LABEL] = d[LABEL] + offset
    d[PRED] = d[PRED] + offset * 0.05
    res = decompose_by_group(Panel(data=d, train_end=p.train_end, label_name=p.label_name))
    assert res.between_ic > 0.5
    assert res.level_effect < -MATERIAL_GAP
    assert res.passed is True
    assert "opposite direction" in res.verdict
    assert f"between-group IC is {res.between_ic:+.4f}" in res.verdict
    assert "works against" not in res.verdict


@pytest.mark.parametrize("case", ["group_only", "genuine"])
def test_verdict_and_report_treat_the_three_ics_as_views_not_parts(case):
    """FAIL must not say the score comes from ranking groups; the within-threshold
    PASS must not claim positive within-group ability; the report must not
    present the difference as an additive effect."""
    if case == "group_only":
        res = decompose_by_group(_group_only_panel())
        assert res.passed is False
        assert "not additive parts" in res.verdict
    else:
        res = decompose_by_group(generate_panel(skill=0.5)[0])
        assert res.passed is True
        assert "does not by itself show positive within-group" in res.verdict
    for claim in ("Much of the score comes from", "group dummy", "survives inside groups"):
        assert claim not in res.verdict

    text = format_group_decomposition(res)
    assert "not additive components" in text
    assert "Level effect" not in text and "(ranking groups)" not in text
    for line in text.splitlines():
        assert len(line) <= WIDTH, line


def test_no_scorable_group_is_inconclusive_without_claiming_one_group():
    """With zero noise every group's prediction is constant, so no group has a
    within-group IC: the verdict must not speak of "one group" nor assert that
    pooled already equals within, and the report prints undefined, not nan."""
    res = decompose_by_group(_group_only_panel(noise=0.0))
    assert res.n_groups == 0
    assert res.passed is None
    assert "fewer than two groups" in res.verdict
    assert "(0 scorable)" in res.verdict
    assert "one group" not in res.verdict and "already a within-group" not in res.verdict
    text = format_group_decomposition(res)
    assert "undefined" in text and "nan" not in text


def test_undefined_between_group_ic_reads_undefined():
    """Two groups: within-group ICs exist but the between-group IC needs at least
    MIN_CROSS_SECTION groups per date, so it is undefined."""
    panel = _group_only_panel()
    d = panel.data.copy()
    d["group"] = d["group"] % 2
    d[LABEL] = d[LABEL] - d[PRED] + d["group"] * 3.0
    d[PRED] = d["group"] * 3.0 + np.random.default_rng(1).normal(0.0, 0.01, len(d))
    res = decompose_by_group(Panel(data=d, train_end=panel.train_end, label_name=panel.label_name))
    assert not np.isfinite(res.between_ic)
    assert res.passed is False
    assert "between-group IC is undefined" in res.verdict
    assert "nan" not in res.verdict
    text = format_group_decomposition(res)
    between_line = next(ln for ln in text.splitlines() if "Between-group IC" in ln)
    assert between_line.rstrip().endswith("undefined")
    assert "nan" not in text
