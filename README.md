# backtest-credibility-audit

**A tool that tells you whether your backtest result is real.**

Give it predictions and realized labels. It reports how much of your headline
number is *free* (obtainable by a naive baseline with no model at all) and how
much is attributable to the model. It also checks whether the result depends on
correct time alignment, which is what a leak looks like from the outside.

---

## The problem: level plus deviation

A high evaluation metric is not evidence of predictive skill. It can also be
produced by the structure of the target.

Decompose a persistent target into a stable per-entity level plus a deviation:

```
y(i, t) = level(i) + deviation(i, t)
```

For targets like realized volatility or log volume, the cross-sectional variance
of `level(i)` dwarfs that of `deviation(i, t)`. Any prediction that merely
recovers each entity's typical level (which a per-entity intercept does
automatically, without learning anything) scores a high cross-sectional IC while
containing **zero** information about dynamics. Entities that have historically
been volatile are still volatile: true, useless, and indistinguishable from skill
if you only look at one number.

This tool separates the two.

---

## A zero-skill strategy with a Sharpe of 147

`make demo` builds a prediction that has **exactly zero** skill. It knows each
entity's typical level and nothing else. Scored the way a backtest would score
it:

```
      Sharpe (per period)         9.2651
      Sharpe (ann. sqrt 252)       147.1
      hit rate                    100.0%
      peak-to-trough (score)      0.0000
      periods                        104

  These are a unit-gross scoring device, not an invested book: no
  costs, no capacity, no constraints. The Sharpe figures are not
  achievable returns, and peak-to-trough is the decline of the score
  series in its own units, not a drawdown of capital.
```

Every day profitable, never declining. The book is long the persistently
volatile names and short the persistently quiet ones, and the target barely
moves. The same positions, scored against **demeaned** labels (the part of the
target that actually varies):

```
      Sharpe (per period)        -0.2809
      Sharpe (ann. sqrt 252)        -4.5
      hit rate                     37.5%
      peak-to-trough (score)      2.2536
```

Nothing was left. The audit reaches the same verdict in IC units, next to a
second synthetic panel that does carry real information about deviations:

```
case                raw IC   demeaned IC   increment  naive incr.
-----------------------------------------------------------------
level_only         +0.6295       +0.0006     -0.0022      +0.2194
genuine_skill      +0.8577       +0.6880     +0.4667      +0.5197
```

Raw IC rates the skill-free model at 0.63, a number most backtest reports would
present as a strong result. Demeaned IC rates it at 0.0006. Both panels are
synthetic with **known** ground truth, so the right answer is not a matter of
opinion. The Sharpe figures are not achievable returns; they are the IC in
different units, computed because this is the form in which the deception is
normally encountered.

---

## Quick start

The demo needs no database and no network:

```bash
pip install -e ".[dev]"
make demo
make test
```

Install from source: this project is not published to PyPI, and the similarly
named `backtest-audit` package there is an unrelated project.

To audit your own results:

```bash
btca predictions.csv --train-end 2024-06-30 --label-name realized_range
```

The CSV needs four columns: `entity_id, event_date, prediction, label`. Any
pipeline in any language can produce it. The Postgres-backed parts
(point-in-time storage, SQL feature construction) are optional: `make up` then
`make schema`.

---

## What it checks

One audit module for each of the nine channels of inflation the
[spec](docs/SPEC.md) enumerates. Every report carries a coverage manifest: a
module that did not run still appears, with its reason and the evidence it needs.

| Module | What it catches | Needs |
|---|---|---|
| Baseline decomposition | Score a naive predictor gets for free; demeaned IC and partial-correlation increment over the strongest baseline | `prediction`, `label` |
| Alignment | Whether the score depends on correct time alignment (shuffle, forward shift, backward diagnostic) | `event_date` |
| Point-in-time | Features computed from data revised after the fact | event and knowledge times, as-of evidence |
| Survivorship | A universe missing its failures, from panel-derived attrition | dated universe membership |
| Grouping | A score that works only across groups, not within them | a group column |
| Serial-dependence significance | Autocorrelation inflating t-statistics (Newey-West HAC, effective sample size) | a dated IC series |
| Validation protocol | A split that leaks: random vs walk-forward vs embargoed walk-forward | feature columns `f_*` |
| Execution timing | A trade at a price that was no longer available | forward returns |
| Selection bias | The best of N configurations (Deflated Sharpe, FDR screening) | every candidate's returns, trial count |

A correlation on a constant cross-section is reported as `undefined`, never as
zero, and a check that ran but could not reach a conclusion is `INCONCLUSIVE`,
distinct from one that was `SKIPPED`. Why each design choice was made is in
[docs/DESIGN.md](docs/DESIGN.md).

---

## Limitations

- **The demo uses synthetic data.** Deliberately, so the correct answer is known,
  but it shows the tool works, not that any particular real strategy is flawed.
- **Public daily data is rarely revised**, so the point-in-time module is
  exercised against a constructed revision scenario rather than a natural one.
- **Corporate actions** are handled only to the extent that a ratio-based target
  is invariant to multiplicative adjustment.
- **The baselines are not exhaustive.** Beating them means a result is not
  obviously free, not that it is useful. A fitted GARCH or an implied-volatility
  quote would be stronger controls and is not implemented.
- **Linear methods throughout.** Partial correlation controls for linear
  dependence on the baseline only.

Known blind spots of individual modules are discussed in
[docs/DESIGN.md](docs/DESIGN.md).

---

## Documentation

- [docs/SPEC.md](docs/SPEC.md): the research question and what would count as failure
- [docs/PLAN.md](docs/PLAN.md): module breakdown, verification, roadmap
- [docs/DESIGN.md](docs/DESIGN.md): design decisions and the evidence behind them
- [docs/STATUS.md](docs/STATUS.md): what is implemented
- [docs/MIGRATION.md](docs/MIGRATION.md): breaking changes and what to do
- [docs/CROSS_REPO_AUDIT.md](docs/CROSS_REPO_AUDIT.md): findings propagated to sibling repositories
- **How it was built:** [docs/AI_NOTES.md](docs/AI_NOTES.md), built with AI assistance, with the incident log and the constraint each failure added

## License

MIT
