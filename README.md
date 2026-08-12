# backtest-audit

**A tool that tells you whether your backtest result is real.**

Give it predictions and realized labels. It reports how much of your headline
number is *free* — obtainable by a naive baseline with no model at all — and how
much is attributable to the model. It also checks whether the result depends on
correct time alignment at all, which is what a leak looks like from the outside.

---

## The problem

A high evaluation metric is not evidence of predictive skill. It can also be
produced by the structure of the target.

Decompose a persistent target into a stable per-entity level plus a deviation:

```
y(i, t) = level(i) + deviation(i, t)
```

For targets like realized volatility or log volume, the cross-sectional variance
of `level(i)` dwarfs that of `deviation(i, t)`. Any prediction that merely
recovers each entity's typical level — which a per-entity intercept does
automatically, without learning anything — scores a high cross-sectional IC while
containing **zero** information about dynamics.

The headline IC then measures something close to a tautology: entities that have
historically been volatile are still volatile. That is true, useless, and
indistinguishable from skill if you only look at one number.

This tool separates the two.

---

## Demo output

`make demo` builds two synthetic panels with **known** ground truth and audits
both. Case 1 has perfect level knowledge and *exactly zero* genuine skill. Case 2
has the same level knowledge plus real information about deviations.

```
case                raw IC   demeaned IC   increment  naive incr.
-----------------------------------------------------------------
level_only         +0.6295       +0.0006     -0.0022      +0.2194
genuine_skill      +0.8577       +0.6880     +0.4667      +0.5197
```

Raw IC rates the skill-free model at 0.63 — a number most backtest reports would
present as a strong result. Demeaned IC rates it at 0.0006.

The full report makes the decomposition explicit:

```
==============================================================================
BASELINE DECOMPOSITION
==============================================================================

  Raw IC (Pearson)                               +0.6295   [sd 0.049, hit 100%, n=104]
  Raw IC (Spearman)                              +0.6028   [sd 0.052, hit 100%, n=104]

  Free score available without any model:
    ├─ persistence                              +0.8416   [hit 100%]
    ├─ ewma                                     +0.7913   [hit 100%]
    ├─ per_entity_train_mean                    +0.6549   [hit 100%]
    └─ cross_sectional_mean                   undefined   [undefined (constant cross-section)]

  After removing the per-entity level (training-period mean):
    Demeaned IC                                  +0.0006   [sd 0.098, hit 48%, n=104]
      t-stat (naive)            0.06
      t-stat (Newey-West)       0.07   [maxlags=4, p=0.9445]

  Increment over strongest baseline (persistence), partial correlation:
    Incremental IC                               -0.0022   [sd 0.097, hit 45%, n=104]

==============================================================================
READING
==============================================================================

  Headline raw IC is +0.6295.
  A naive predictor (persistence) reaches +0.8416 with no model at all.
  The naive predictor BEATS the model outright: on this metric the model
  adds nothing over doing no modelling at all.

  Demeaned IC is indistinguishable from zero: once the stable per-entity
  level is removed, the prediction carries no information about deviations.
  The headline IC is measuring the level, not forecast skill.
```

---

## Quick start

The demo needs no database and no network:

```bash
pip install -e ".[dev]"
make demo
make test
```

To audit your own results:

```bash
backtest-audit predictions.csv --train-end 2024-06-30 --label-name realized_range
```

The CSV needs four columns: `entity_id, event_date, prediction, label`. Any
pipeline in any language can produce it.

For the Postgres-backed parts (point-in-time storage, SQL feature construction):

```bash
make up        # starts postgres in docker
make schema    # applies sql/001_schema.sql and sql/002_pit_views.sql
```

---

## What it checks

**Baseline decomposition.** Four naive predictors are scored through the exact
same code path as the model: last observed value (`persistence`), the entity's
training-period mean, an EWMA, and the previous cross-sectional mean. Whatever
they achieve is free. Each obeys the same information constraint as the model —
no baseline may use same-day or future labels — because a leaking baseline would
*understate* the model's increment, which is the wrong direction to err in.

**Demeaned IC.** Cross-sectional IC after subtracting each entity's
training-period mean from both prediction and label. This is the headline metric:
it measures correlation between predicted and realized *deviations*, which is
what a user of the forecast actually needs, since the level is knowable without
any model.

**Incremental IC.** Partial correlation of prediction and label controlling for
the strongest baseline — a bounded, interpretable increment (see design notes for
why the two obvious alternatives are wrong).

**Newey-West significance.** Daily IC series are autocorrelated, so the naive
`mean / (sd/√T)` t-statistic overstates significance. The report shows the naive
statistic, the HAC-corrected statistic, and the ratio between the two standard
errors, so the size of the correction is visible rather than implicit.

**Undefined vs zero.** A correlation on a constant cross-section is *undefined*,
not zero. The framework reports it as `undefined` with a reason and excludes the
date from averages. "Not measurable" and "measured, found to be nil" are
different claims, and collapsing them is how a summary statistic misleads.

---

## Design decisions

*This section is the point of the project. Each item is a place where the obvious
implementation is wrong.*

### The time convention is a single date, not two

A panel row is `(entity_id, event_date, prediction, label)`, where `event_date`
is when the target is **realized** and the prediction must have been computable
strictly before it. Indexing both columns by realization date makes the
correctness condition local and checkable, and removes any opportunity for an
off-by-one between "the day the features came from" and "the day the label came
from". Callers whose data is keyed by forecast date shift it once, at the
boundary. Two conventions coexisting inside a library is how alignment bugs are
born.

### Demeaning must use training-period statistics only

Subtracting a full-sample per-entity mean would remove a quantity computed partly
from the evaluation labels — leakage committed by the audit tool itself. So
`Panel.require_train_end()` **raises** rather than defaulting to the full sample.
A tool that quietly leaked while checking for leaks would not deserve to be
believed about anything else.

### Two obvious increment statistics are wrong, and the third needs a correction

*Difference of correlations*, `corr(x,y) − corr(b,y)`, is not a correlation and
has no bounded interpretation.

*Correlation of differences*, `corr(x−b, y−b)`, looks like "removing the baseline
from both sides", but the shared `−b` term is common to both arguments and
induces correlation of its own. When `Var[b]` is large it dominates, and the
statistic can **exceed** `corr(x,y)` — incoherent for something called an
increment. `tests/test_baselines.py` constructs a case where this reads > 0.5
while the true conditional association is zero.

*Partial correlation* is the right statistic — but applied naively it is still
biased upward, and this is the most interesting finding in the project. Every
baseline is a **noisy proxy** for the entity level: `persistence` is a single
observation, so it carries a whole period of transient noise alongside the level.
Controlling for a mismeasured proxy leaves residual level information in both
residuals, and the model gets credited for level knowledge it obtained for free.

Measured on the zero-skill synthetic panel, the naive partial correlation reads
**+0.2194** — entirely spurious.

The fix is to demean all three series by their own training means before taking
the partial correlation. Demeaning does not suffer the proxy problem, because
each series' level is estimated from that series itself rather than imported from
another. Composed this way the statistic reads **−0.0022** at zero skill and
rises monotonically with true skill. `demean=True` is the default; the biased
variant is retained only so the demo can show the size of the bias, and it
carries a warning in its metadata.

A consequence worth stating: when the control *is* the level estimate, demeaning
leaves it identically zero and the partial correlation is correctly reported as
**undefined** — you cannot control for the level twice.

### SQL window frames must exclude the current row

The default window frame in SQL **includes** the current row. A "trailing mean"
written without an explicit frame therefore contains same-day information — and
because the output still looks like a plausible moving average, the error
survives code review easily. Every frame in `sql/002_pit_views.sql` ends at
`1 PRECEDING`, and `tests/test_sql_windows.py` asserts the property against a
hand-computable fixture so a future edit that drops it fails the build.

### Corrections are stored as rows, not updates

`price_raw` is bitemporal: `event_date` (what the data describes) and
`knowledge_date` (when it became available). A revision is an **insert**, never
an update. Without this, a corrected value silently overwrites the original and
the historical record becomes the *current* view of the past rather than what was
knowable at the time — after which no amount of care in the modelling code can
prevent look-ahead. `price_asof(date)` reconstructs the actual historical view via
an as-of join; comparing a pipeline's results against `price_restated` is the
look-ahead test.

### Sanity checks belong in CI, not in a notebook

Shuffle and shift tests are `pytest` tests that run on every push. Data
credibility is a build-time constraint, not something to verify once by hand and
then trust indefinitely.

---

## Status

Implemented: panel contract, baseline decomposition, demeaned IC, partial-correlation
increment with the errors-in-variables correction, Newey-West inference, text and
JSON reports, offline demo, PIT schema and SQL feature views, 42 tests.

Next: alignment audit as CI tests (shuffle / shift / future-shift), the
leaky-vs-clean example pipelines on real public data, PIT-vs-restated comparison,
group decomposition, HTML report.

---

## Limitations

Stated plainly, because a tool that audits credibility should be candid about its
own:

- **The demo uses synthetic data.** That is deliberate — the correct answer must
  be known for the demonstration to demonstrate anything — but it means the demo
  shows the tool works, not that any particular real strategy is flawed.
- **Public daily data is rarely revised**, so the point-in-time module is
  exercised against a constructed revision scenario rather than a natural one.
  Real vendor data would test it harder.
- **Corporate actions are handled only to the extent that a ratio-based target is
  invariant to multiplicative adjustment.** Full corporate-action handling is out
  of scope.
- **The baselines are not exhaustive.** A model beating everything here is not
  thereby proven useful; it is only not obviously free. Domain-appropriate
  baselines (a fitted GARCH, an implied-volatility quote) would be stronger
  controls and are not implemented.
- **Linear methods throughout.** Partial correlation controls for *linear*
  dependence on the baseline; a non-linear relationship would leave residual
  association that this framework would misattribute to the model.

## License

MIT
