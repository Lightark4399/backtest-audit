# backtest-audit

**A tool that tells you whether your backtest result is real.**

Give it predictions and realized labels. It reports how much of your headline
number is *free* — obtainable by a naive baseline with no model at all — and how
much is attributable to the model. It also checks whether the result depends on
correct time alignment at all, which is what a leak looks like from the outside.

**Documentation:** [SPEC.md](SPEC.md) — the research question and what would
count as failure · [PLAN.md](PLAN.md) — module breakdown and verification ·
[AI_NOTES.md](AI_NOTES.md) — how it was built with AI assistance, and the nine
things that went wrong

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

## The same finding, as a backtest would report it

`make demo` ends with a strategy built on a prediction that has **exactly zero**
skill — it knows each entity's typical level and nothing else:

```
      annualised Sharpe             147.1
      hit rate                     100.0%
      maximum drawdown               0.0%
      periods                         104
```

Every day profitable, no drawdown, a Sharpe no real strategy reaches. It is worth
pausing on how convincing that table is, because none of it is earned: the book is
long the persistently-volatile names and short the persistently-quiet ones, and
the target barely moves.

The same positions, scored against demeaned labels — the part of the target that
actually varies:

```
      annualised Sharpe              -4.5
      hit rate                      37.5%
      maximum drawdown             186.8%
```

Nothing was left. The audit reaches the same verdict in IC units: raw IC +0.63,
demeaned IC +0.0006.

The Sharpe figures here are not achievable returns — no costs, no capacity, no
constraints. They are the same information as the IC in different units, and the
module says so. The reason to compute them anyway is that this is the form in
which the deception is normally encountered.

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

### The alignment audit is powerful in one direction and blind in another

Shuffling labels within a date collapses the IC to zero, as it must. But two
findings from testing it against panels of known construction shaped the module:

**The shift test runs on demeaned series, because the level blinds it.** The
per-entity level is constant in time, so shifting labels does not disturb it at
all. Measured on the synthetic panels, a one-day shift moves the raw IC of a
genuinely skilful model by 5% and that of a *zero-skill* model by 0.1% -- neither
is usable. After demeaning, the same shift moves the skilful model's IC by 14%.
The level does not merely inflate the headline number; it also blinds the check
meant to validate it.

**The backward shift is diagnostic, never a verdict.** A forecaster of a
persistent target is usually built from lagged labels, so its prediction
resembles the label it was *built from* more than the one it forecasts. On the
clean example pipeline -- honest by construction -- scoring against the previous
date raises the demeaned IC from 0.71 to 0.97. Gating on that would fail nearly
every autoregressive model, so the backward shift is reported with its
interpretation and the forward shift carries the verdict.

### What the alignment audit cannot catch, stated plainly

The example pipelines carry five switchable defects, and the alignment audit
detects one class of them. A contemporaneous feature -- a same-day measurement
whose timestamp was assumed available earlier than it is -- produces a prediction
that genuinely matches the label it is scored against. The *pairing* is sound;
what is wrong is that the prediction could not have been computed in time. That
is a question about data vintage, which the point-in-time module addresses.

The same reasoning explains a result worth seeing: an off-by-one in an
autoregressive pipeline turns *into* contemporaneous leakage, and passes the
alignment audit. The module detects misalignment when the prediction is not
itself built from lagged labels. Both cases are covered in the tests.

### Point-in-time: the only module that catches an unknowable feature

The alignment audit is blind to a prediction that matches its label for the wrong
reason. This module supplies the missing check by asking a different question:
*could this have been computed at the time?*

Observations are stored bitemporally -- ``event_date`` for the day described,
``knowledge_date`` for the day the value became available -- and corrections are
inserted as new rows, never updates. The same feature SQL is then run over two
relations: the restated view (current best value for every date) and an as-of
reconstruction (the value as it stood on each date). Scoring the same model both
ways puts the look-ahead advantage in the same units as the rest of the report.

On a constructed scenario where 30% of observations are first reported with error
and corrected five days later, the restated backtest scores a demeaned IC of
0.727 against the point-in-time 0.639 -- **an unearned +0.088**. The relationship
is dose-responsive: 10% / 35% / 60% revision rates give gaps of +0.045 / +0.101 /
+0.178.

Three details make that number attributable rather than merely suggestive:

* **Both vintages go through the identical SQL macro.** Two feature
  implementations would let a difference be an artefact of the code.
* **Both are scored on exactly the same evaluation dates.** The as-of
  reconstruction is sampled, since each date costs a pass over the history.
  Leaving the restated arm on the full date set made the panels differ in
  composition as well as vintage -- on the no-revision control that confound
  alone produced a gap of -0.013, larger than the threshold used to call a result
  material. With the dates aligned, the control gives a gap of exactly zero.
* **The restated arm does not move with the revision scenario.** It sees final
  values by definition, and a test asserts it is invariant to the revision rate,
  so the gap can only come from the as-of arm.

When nothing was ever revised the audit reports no gap -- and says explicitly
that this is *not* a clean bill of health, since a feature built from same-day
data is unknowable in time whether or not any value was corrected.

### Survivorship is the one bias with nothing to look at

Every other check examines rows that exist. Survivorship is about which rows
exist at all, and that makes it the hardest to notice: no anomalous value, no
correlation behaving oddly, nothing to flag. A universe backfilled from a current
constituent list simply never loads the entities that failed.

The audit reconstructs membership by listing and delisting dates and scores the
same model both ways. On a panel where a quarter of entities delist and those
entities are genuinely harder to forecast, restricting to survivors raises the
demeaned IC from 0.495 to 0.546 -- **+0.051 of unearned score**.

The control case is what makes it a measurement rather than an alarm: when
attrition is *uncoupled* from predictability, the same audit reports a gap of
-0.002. Entities leaving for reasons unrelated to the target cost sample size but
introduce no bias, and a check that flagged that would be flagging attrition
itself.

On a balanced panel the audit reports **not a pass but an unanswerable
question** -- a universe assembled without its delisted entities in the first
place looks exactly like a complete one, and the absence is invisible from inside
the data.

### Group decomposition: a score that only works across groups

If entities fall into groups with different levels -- sectors, exchanges, size
bands -- a prediction that merely identifies the group will rank the pooled
cross-section well while ranking nothing within any group. That distinction is
practical, not academic: a forecast is used inside a group far more often than
across one.

The decomposition reports pooled IC, size-weighted within-group IC, and
between-group IC. On a prediction constructed to encode only the group, pooled IC
reads **+0.935** while within-group IC is **+0.005** and between-group is
**+1.000**.

Two choices worth noting. Within-group ICs are averaged **weighted by group
size**, because an unweighted mean lets a four-entity group count as much as a
four-hundred-entity one, and small groups produce the noisiest estimates; the
unweighted figure is reported alongside, since a gap between the two is itself a
sign of heterogeneous group quality. And a group whose typical cross-section is
below three is reported as `undefined` rather than averaged in, for the same
reason degenerate correlations are excluded elsewhere.

### The validation protocol audit only fires when it should

`train_test_split(shuffle=True)` is the most common way to inflate a backtest and
the most innocuous-looking line in the pipeline. Random assignment scatters
observations across folds without regard to time, and for a persistent target
adjacent dates are near-duplicates -- so the model is effectively evaluated on
data it trained on.

The audit scores the same model under shuffled K-fold, expanding-window
walk-forward, and walk-forward with an embargo. On a panel where the
feature-to-label relationship drifts, random splitting reads **+0.695** against
**+0.602** for the purged walk-forward: **+0.094 of unearned score**, with the
embargo alone accounting for +0.015 of it.

The conditional half is what makes this a measurement rather than a maxim. With
a *stationary* relationship the same audit reports an inflation of **-0.0002**.
Testing revealed why, and the finding shaped the module: with a low-capacity
model and a stable relationship there is nothing for random assignment to
exploit -- training on a random subset and training on the past yield the same
coefficients. Random splitting leaks when the relationship *drifts*, because the
random folds hand the model rows drawn from the test period's regime. A module
that condemned shuffling unconditionally would be repeating a rule of thumb; this
one measures whether the rule applies to your data.

### Effective sample size

The HAC correction already tells you the standard error is understated. It does
not tell you in a form anyone acts on. The report now also states it as a count:

```
    t-stat (naive)            4.21
    t-stat (Newey-West)       2.35   [maxlags=4, p=0.0210]
    SE inflation              1.79x  [lag-1 autocorr +0.75]
    effective sample            15 of 104 days  [85% lost to serial dependence]
```

"104 days of evidence, worth about 15 independent observations" needs no
knowledge of what a HAC estimator is, and it is the phrasing that stops a reader
over-reading a t-statistic. Negative autocorrelation is capped at `n_eff = n`
rather than awarding a bonus, since claiming more information than observations
would be the same overstatement in the opposite direction.

### Postgres is the reference, DuckDB is what runs

``sql/001_schema.sql`` and ``sql/002_pit_views.sql`` are the reference design.
``sql/duckdb/001_schema.sql`` is the executable port, and it is what the
integration tests run against: DuckDB is embedded, so the point-in-time tests
need no service to provision and run in CI unchanged. The dialects agree on
everything load-bearing here -- ``DISTINCT ON``, window frames, CTEs, CHECK
constraints. Two differences are documented where they bite: Postgres
set-returning functions become table macros, and ``ASOF`` is a reserved word in
DuckDB (it has a native ``ASOF JOIN``), so the macro parameter is named
``cutoff``.

---

## Status

Implemented: panel contract, baseline decomposition, demeaned IC,
partial-correlation increment with the errors-in-variables correction,
Newey-West inference, alignment audit (shuffle / forward shift / backward
diagnostic), leaky-vs-clean example pipelines with five switchable defects,
bitemporal store with as-of reconstruction, point-in-time vs restated
comparison, survivorship audit via universe reconstruction, within/between group
decomposition, validation-protocol comparison (random vs walk-forward vs purged),
effective sample size, a thin PnL layer reporting raw and demeaned Sharpe, text
and JSON reports, offline demo, 125 tests.

Possible extensions are listed with their rationale and cost in
[PLAN.md](PLAN.md#roadmap). None of them blocks the framework being usable: the
six channels of inflation it enumerates are each implemented, tested and wired
into the report.

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
