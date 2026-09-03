"""Semantic tests for translating audit numbers into human conclusions."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from audit import demo
from audit.audits.execution import audit_execution_timing
from audit.audits.selection import deflated_sharpe
from audit.metrics.ic import ICSeries
from audit.metrics.performance import (
    TRADING_DAYS,
    annualisation_label,
    annualise_sharpe,
)
from audit.report.text import (
    WIDTH,
    execution_lag_table,
    format_execution_timing,
    format_interpretation,
    format_selection,
)
from audit.synthetic import generate_return_panel


def _series(value: float | None, name: str = "ic") -> ICSeries:
    if value is None:
        values = pd.Series(dtype=float)
        n_obs = pd.Series(dtype=int)
        undefined = 1
    else:
        values = pd.Series([value], index=[pd.Timestamp("2024-01-02")])
        n_obs = pd.Series([20], index=values.index)
        undefined = 0
    return ICSeries(name, values, n_obs, 1, undefined, 0)


def _reading(demeaned: float | None) -> str:
    baselines = pd.DataFrame({"mean": [0.10]}, index=["persistence"])
    return format_interpretation(_series(0.20, "raw"), baselines, _series(demeaned))


@pytest.mark.parametrize("value", [-0.6880, -0.02])
def test_negative_demeaned_ic_is_never_described_as_positive_skill(value):
    reading = _reading(value).lower()
    assert "negative association" in reading
    assert "small but non-zero skill" not in reading
    assert "separate out-of-sample validation" in reading


@pytest.mark.parametrize("value", [-0.0199, 0.0, 0.0199])
def test_near_zero_demeaned_ic_is_described_as_no_information(value):
    reading = _reading(value).lower()
    assert "indistinguishable from zero" in reading
    assert "negative association" not in reading


def test_undefined_demeaned_ic_declines_to_interpret():
    reading = _reading(None).lower()
    assert "could not be computed" in reading
    assert "contains real information" not in reading


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.02, "small positive association"), (0.0999, "small positive association"),
     (0.10, "substantial positive association")],
)
def test_positive_interpretation_respects_declared_thresholds(value, expected):
    assert expected in _reading(value).lower()


def test_non_finite_mean_declines_to_interpret():
    reading = _reading(np.nan).lower()
    assert "could not be computed" in reading


def test_negative_best_baseline_is_not_described_as_free_positive_score():
    baselines = pd.DataFrame({"mean": [-0.0086, -0.0090]}, index=["persistence", "ewma"])
    reading = format_interpretation(_series(0.48, "raw"), baselines, _series(0.46)).lower()
    assert "no naive baseline achieved a positive ic" in reading
    assert "-2% of the headline" not in reading


# ----------------------------------------------------------------------
# Sharpe scale in the selection report
# ----------------------------------------------------------------------
def _selection_returns() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0007, 0.01, 600))


def test_selection_report_names_the_annualisation_factor_when_it_is_supplied():
    """Detection half: a supplied frequency is stated, not left to inference."""
    result = deflated_sharpe(_selection_returns(), n_trials=42, periods_per_year=252)
    text = format_selection(result)

    assert "Observed Sharpe (per period)" in text
    # Derived, not transcribed: a test that pins the label as a literal is the
    # same duplication it is meant to catch.
    assert f"Sharpe (annualised, {annualisation_label(252)})" in text
    assert "NOT AVAILABLE" not in text
    assert f"{result.observed_sharpe * np.sqrt(252):+.4f}" in text


def test_selection_report_emits_no_annualised_number_without_a_frequency():
    """Control half: the ambiguous label is gone and nothing replaces it.

    The report previously printed a bare "Observed Sharpe" while the surrounding
    narrative quoted the annualised figure for the same case, leaving the two
    numbers differing by sqrt(252) under one name. The fix must not swing the
    other way and invent an annualised figure the panel cannot support.
    """
    result = deflated_sharpe(_selection_returns(), n_trials=42)
    text = format_selection(result)

    assert "Observed Sharpe (per period)" in text
    assert "Sharpe (annualised)  NOT AVAILABLE -- observation frequency not supplied" in text
    assert "sqrt(" not in text

    # NOT AVAILABLE must not read as zero, and the annualised value must appear
    # nowhere in the rendered text under any rounding.
    assert result.observed_sharpe_annualised is None
    annualised = result.observed_sharpe * np.sqrt(252)
    for places in (1, 2, 3, 4):
        assert f"{annualised:.{places}f}" not in text


def test_selection_report_lines_fit_the_report_width():
    """Incident 11's second defect: a note wider than the report it sits in."""
    for periods in (None, 252):
        result = deflated_sharpe(_selection_returns(), n_trials=42, periods_per_year=periods)
        for line in format_selection(result).splitlines():
            assert len(line) <= WIDTH, line


def test_the_demo_and_the_report_render_one_execution_table():
    """The demo used to keep its own copy of these rows.

    It went on printing "ann. Sharpe" over a per-period figure after the heading
    in the report had been corrected, because the two renderings were separate.
    Pinning the shared rows inside the full report is what makes a divergence
    fail here rather than in the delivered output.
    """
    panel = generate_return_panel(lookahead=1.0)
    result = audit_execution_timing(panel)

    rows = execution_lag_table(result)
    rendered = format_execution_timing(result)

    assert "Sharpe (per period)" in rows[0]
    for row in rows:
        assert row in rendered


def _demo_surfaces(outdir) -> dict[str, str]:
    """Everything a demo run puts in front of a reader: stdout and every file."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        demo.main(["--outdir", str(outdir)])

    surfaces = {"<stdout>": buffer.getvalue()}
    for path in sorted(outdir.rglob("*")):
        if path.is_file():
            surfaces[path.name] = path.read_text(encoding="utf-8")
    return surfaces


def test_no_rendered_surface_claims_annualisation_without_naming_the_factor(tmp_path):
    """Scoped to the property, across every surface, not to one report.

    The previous version of this check asserted that "ann." appeared nowhere in
    the execution report. It passed while the demo's comparison table was still
    headed "raw (ann.)", because the assertion named a place rather than a
    property. That scoping is why the same defect survived two rounds of being
    fixed: each round corrected the rendering that had been noticed.

    Two properties are asserted, because the defect has two shapes.

    A line *claiming* annualisation must carry the factor string that
    ``annualisation_label`` produces -- not merely some number. Naming a factor
    is not the same as naming the right one: a sentence reading "annualised at
    252" passed the weaker form of this check while describing a multiplication
    by sqrt(252). Requiring the shared label is what makes the renderers agree,
    and the check is the enforcement arm of that sharing rather than its
    substitute. The alternatives are a declared unavailability (the honest case,
    where there is no factor to name) and a scale-suffixed identifier such as a
    JSON key that carries the scale in its name.

    A reported Sharpe must *declare* its scale at all. The first property alone
    lets a bare name pass by saying nothing -- which is how ``"sharpe": 11.85``
    survived in execution_report.json, correct and silent, with nothing telling
    a reader of that file which of the two scales it was on. Silence is not a
    third acceptable answer.

    Scope: demo surfaces only. ``cli.py`` renders the same ``to_text()`` output
    and is covered by proxy, with nothing pinning that -- a surface of its own
    would escape this test.

    Known limit. This checks that a rendered line carries the shared label and
    declares a scale. It cannot verify that a prose sentence describes the
    computation that produced the number beside it: a renderer has the string
    and the value, not the operation. A sentence quoting a wrong figure, or
    describing a factor it did not apply in words that avoid the label, is not
    reachable from here. What closes that gap is the sharing itself -- one
    string, taken from beside the operation -- not a stronger assertion.
    """
    surfaces = _demo_surfaces(tmp_path)
    label = annualisation_label(TRADING_DAYS)
    claims = ("ann.", "annualis", "annualiz")

    scales = ("per period", "per_period", "_annualised")

    named_factor = 0
    declared_unavailable = 0
    declared_per_period = 0
    for surface, text in surfaces.items():
        for line in text.splitlines():
            low = line.lower()

            # Property one: a claim of annualisation names what it used.
            if any(token in low for token in claims) and "_annualised" not in line:
                if label in line:
                    named_factor += 1
                elif "NOT AVAILABLE" in line:
                    declared_unavailable += 1
                else:
                    raise AssertionError(
                        f"{surface}: claims annualisation without carrying the "
                        f"shared factor label {label!r}, so nothing ties the "
                        f"words to the operation: {line!r}"
                    )

            # Property two: a Sharpe reported with a value declares its scale.
            # A line with no digits is prose about Sharpe, not a reported one.
            if "sharpe" not in low or not any(ch.isdigit() for ch in line):
                continue
            if any(token in low for token in scales):
                declared_per_period += 1
            elif "NOT AVAILABLE" in line:
                continue
            elif any(token in low for token in claims) and label in line:
                continue
            else:
                raise AssertionError(
                    f"{surface}: reports a Sharpe without declaring its scale. "
                    f"Per-period, annualised with a named factor, or unavailable "
                    f"with a reason -- silence is not an option: {line!r}"
                )

    # A scan that matched nothing would pass vacuously, so pin that the surfaces
    # were really collected and that the claim actually occurs in them. The demo
    # supplies a frequency throughout, so `declared_unavailable` is expected to
    # be zero here; the NOT AVAILABLE path is covered against `format_selection`
    # directly, where it can be produced.
    assert len(surfaces) > 5, sorted(surfaces)
    assert named_factor >= 3
    assert declared_per_period >= 5
    assert declared_unavailable == 0


def test_no_rendered_surface_keeps_the_headings_that_were_wrong(tmp_path):
    """The two specific regressions, pinned across every surface at once."""
    surfaces = _demo_surfaces(tmp_path)
    for surface, text in surfaces.items():
        assert "ann. Sharpe" not in text, surface
        assert "(ann.)" not in text, surface


def test_the_annualisation_label_describes_the_operation_actually_applied():
    """Ties the words to the arithmetic, which the surface scan cannot.

    The scan can see that a line carries the shared label. Only this can check
    that the label is true: the factor it names, applied to a per-period Sharpe,
    must reproduce what ``annualise_sharpe`` returns. A label reading "252"
    would fail here, which is what makes requiring it on every rendered line
    worth asserting.
    """
    label = annualisation_label(TRADING_DAYS)
    operation, periods = label.split()
    assert operation == "sqrt"
    assert int(periods) == TRADING_DAYS

    described = 0.1 * float(periods) ** 0.5
    assert annualise_sharpe(0.1, TRADING_DAYS) == pytest.approx(described)

    assert annualisation_label(None) == "not annualised"
