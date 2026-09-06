# MIGRATION

## 0.1.2 → 0.2.0

A minor release, not a patch. The distribution and CLI are renamed, a reported
metric is renamed and changes units, annualisation now requires an explicit
observation frequency, and the selection module can no longer emit a strong PASS
from an approximate benchmark. Each of those changes something a caller reads or
calls.

Nothing in the panel contract changed. A CSV of
`(entity_id, event_date, prediction, label)` that worked with 0.1.2 works
unchanged, and the importable package is still `audit`.

---

### 1. The distribution and the CLI are renamed

| | 0.1.2 | 0.2.0 |
|---|---|---|
| Distribution | `backtest-audit` | `backtest-credibility-audit` |
| Console script | `backtest-audit` | `btca` |
| Import package | `audit` | `audit` (unchanged) |

**Why.** `backtest-audit` is taken on PyPI by an unrelated static-analysis tool
for look-ahead bias in backtesting code. The domains are adjacent enough that
`pip install backtest-audit` silently gets you someone else's project.

**What breaks.** Any install by the old distribution name; any shell script or CI
step invoking `backtest-audit`; any code calling
`importlib.metadata.version("backtest-audit")`.

**Migration.** Install from source — this project is not published to PyPI:

```bash
pip install -e ".[dev]"     # or: pip install <path-or-git-url>
btca predictions.csv --train-end 2024-06-30
```

Replace `backtest-audit` with `btca` in scripts, and
`version("backtest-audit")` with `version("backtest-credibility-audit")`. Note
that `importlib.metadata.version("backtest-audit")` will not raise if the
unrelated PyPI package is installed in the same environment — it will return
that project's version. Check the name you get back.

---

### 2. `max_drawdown` → `additive_peak_to_trough`, and the unit changes

**What it was.** `PerformanceStats.max_drawdown`, a fraction: the peak-to-trough
decline of the score series divided by its running peak, floored at 1.0.

**What it is.** `PerformanceStats.additive_peak_to_trough`, an absolute
difference in the units of the score series. JSON key
`additive_peak_to_trough`. The `compare_performance` column is
`additive_peak_to_trough_demeaned`.

**Why.** Positions in this layer are dollar-neutral and unit-gross and nothing
compounds, so the series is a cumulative *score*, not the net asset value of an
invested book. Expressing its decline as a fraction of a running peak described
it as a percentage of something that is not capital, and the result could exceed
100% — it read 186.8% on the demo's zero-skill panel, which no drawdown of
invested capital can do. A reader would reasonably have understood a
conventional equity-curve maximum drawdown, which it never was.

**What breaks.** The attribute and the JSON key are gone, so a reader of either
gets an `AttributeError` or a `KeyError` rather than a number on the wrong
scale — deliberately. **Do not simply rename your reference:** the value is on a
different scale. A stored 0.5 under the old name is 50% of a running peak; the
new field would report that same episode as an absolute decline in score units,
which is a different number. Comparisons against historical thresholds must be
re-derived, not renamed.

**Migration.** Rename the reference and re-derive any threshold. If you have real
capital and a compounded NAV, compute a conventional maximum drawdown from that
NAV yourself and report it under your own key; this layer does not, because it
has no capital to speak of.

---

### 3. `periods_per_year` has no default, and annualisation can be unavailable

**What changed.** `performance()`, `compare_performance()` and
`deflated_sharpe()` take `periods_per_year: int | None = None`. It was
previously `252`.

**Why.** The panel contract is four columns and carries no observation
frequency. A default of 252 asserted daily data about panels that might be
weekly or monthly, and the error is silent: every annualised figure is wrong by
`sqrt(actual / 252)` with nothing in the output saying so.

**What breaks.**

- `PerformanceStats.sharpe_annualised` is now `None` when no frequency was
  supplied. It is **not** `0.0` and **not** `NaN`. Arithmetic on it raises
  `TypeError` rather than silently producing a wrong number. `NaN` is still used
  for "computed and undefined"; `None` means "no frequency was supplied", and
  the two are distinct on purpose.
- `compare_performance()` columns are renamed. `sharpe_raw` and
  `sharpe_demeaned` became `sharpe_raw_per_period` and
  `sharpe_demeaned_per_period`, and they now hold **per-period** figures. The
  annualised columns `sharpe_raw_annualised` and `sharpe_demeaned_annualised`
  exist only when `periods_per_year` is supplied. The old names raise `KeyError`
  rather than returning a number on the wrong scale.
- The text report prints
  `Sharpe (annualised)  NOT AVAILABLE -- observation frequency not supplied`
  when no frequency was given. Parsers looking for a number there must handle
  this string. It does not mean zero and does not mean the computation failed.

**Migration.** Pass the frequency you actually have:

```python
performance(panel, periods_per_year=252)          # daily
performance(panel, periods_per_year=52)           # weekly
performance(panel)                                # per-period figures only
```

If you do not know the observation frequency, do not guess — the per-period
Sharpe is still reported and is the figure the rest of the framework compares
against.

**A caveat now stated in the code.** `sqrt(T)` scaling assumes i.i.d. returns.
Under autocorrelation the error's direction follows its sign: positive
autocorrelation inflates the annualised figure, negative deflates it (Lo 2002).
It is not an upper bound.

---

### 4. The selection module will not emit a strong PASS from an approximate benchmark

**The defect.** In Bailey and López de Prado, the Deflated Sharpe Ratio's
benchmark `SR*` is defined against `V[{SR_n}]` — the variance of the Sharpe
estimates **across the N trials**, a property of the search. Through 0.1.2 the
module passed the sampling variance of the *winning candidate's own* Sharpe
estimator instead. Every other part of the computation was correct, including
the z-statistic, whose denominator is that estimator variance.

The two coincide when trials are independent and identically distributed, and
diverge otherwise. Measured on synthetic candidate sets: the cross-trial
variance is **0.003×** the substituted one on a correlated parameter grid, and
**29.5×** on a heterogeneous candidate set. Since `SR*` scales with the square
root, the substituted benchmark is far too high in the first case and far too
low in the second. **The direction of the error is not determined a priori** and
cannot be recovered from the winning return stream.

**What changed.** `deflated_sharpe()` takes an optional `trial_sharpes`:

```python
# formal: the statistic the module is named after
deflated_sharpe(returns, n_trials=42, trial_sharpes=[...42 Sharpes...])

# proxy: the substitution, honestly labelled, PASS unreachable
deflated_sharpe(returns, n_trials=42)
```

`trial_sharpes` must have exactly `n_trials` entries or `ValueError` is raised —
a silent disagreement between the ledger and the declared count is the error the
ledger exists to prevent.

**New result fields**, all present in `to_dict()` and in the text report:

| Field | Values |
|---|---|
| `method` | `formal` \| `proxy` |
| `trial_variance_source` | `cross_trial` \| `winner_estimator` |
| `n_trials_kind` | `declared` \| `inferred_from_ledger` |
| `strong_pass_eligible` | bool |
| `inconclusive_reason` | `not_computable` \| `proxy_benchmark` \| `below_threshold` \| `null` |

**What breaks.**

- **Without `trial_sharpes`, `passed` can never be `True`.** A result that would
  previously have returned `passed=True` now returns `passed=None` with
  `inconclusive_reason="proxy_benchmark"`, even when `deflated_probability`
  exceeds 0.95. If you gate a pipeline on `passed is True`, that gate now fails
  closed until you supply a ledger. This is the intended narrowing: the old
  `True` asserted a statistic that had not been computed.
- **`INCONCLUSIVE` here means the check ran and the evidence was insufficient.**
  It is not `SKIPPED`. In the coverage manifest the entry is
  `run_state=COMPLETED` with `verdict=INCONCLUSIVE` and no skip reason. The
  remedy is a trial ledger, not a missing input.
- **The headline key is renamed by method.** `method=formal` emits
  `selection_adjusted_sharpe_per_period`; `method=proxy` emits
  `approximate_selection_adjusted_sharpe_per_period`. A parser must read the
  `method` field, or look for both keys.

**Two things that are deliberately not capped.**

`FAIL` remains reachable under a proxy benchmark. A substituted benchmark can be
too high as easily as too low, so a FAIL is not formally warranted either; it is
retained as a caution, says so in its own verdict text, and should be read that
way. The asymmetry is intentional: endorsing a result wrongly and doubting one
wrongly are different kinds of error.

A single trial (`n_trials=1`) is exempt from the ceiling. `expected_max_sharpe`
returns `0.0` for one trial whatever variance it is given, so the formal and
substituted benchmarks are identical rather than approximately equal, and there
is no selection to deflate.

**Migration.** If you search a parameter grid, record the Sharpe of every
configuration you examine — including the ones you abandoned — and pass them as
`trial_sharpes`. If you cannot, expect `INCONCLUSIVE` and read the result as an
approximation. Do not infer the trial count from the winning return stream; it
is not recoverable from it.

---

### 5. Renamed serialisation keys

Every reported Sharpe now declares its scale in its own name. Reading a bare
`sharpe` and inferring the scale from context was possible in 0.1.2 and is not
now, because the bare names are gone.

| Artefact | 0.1.2 key | 0.2.0 key |
|---|---|---|
| `execution_report.json` (each lag) | `sharpe` | `sharpe_per_period` |
| `PerformanceStats.to_dict()` | `sharpe` | `sharpe_per_period` |
| `PerformanceStats.to_dict()` | `max_drawdown` | `additive_peak_to_trough` |
| `screen_candidates()` column | `sharpe` | `sharpe_per_period` |
| `selection_report.json` | `observed_sharpe` | `observed_sharpe_per_period` |
| `selection_report.json` | `expected_max_sharpe` | `expected_max_sharpe_per_period` |
| `selection_report.json` | `threshold_sharpe` | `threshold_sharpe_per_period` |
| `selection_report.json` | `sharpe_estimator_variance` | `sharpe_per_period_estimator_variance` |

**A value change hides among these renames.** `execution_report.json`'s per-lag
Sharpe is not merely renamed: through 0.1.2 it was annualised with a hardcoded
252, and it is now per period. The demo's figures moved from
`+188.13 / +2.97 / -0.26` to `+11.8511 / +0.1868 / -0.0163` — the same numbers
divided by `sqrt(252)`. The execution audit receives no observation frequency,
and its lag comparison is a ratio that `sqrt(T)` scaling leaves unchanged, so
per period is the honest scale. If you stored those figures, they are not
comparable across the version boundary.

Python attribute names on the result objects are unchanged
(`LagResult.sharpe`, `PerformanceStats.sharpe`,
`DeflatedSharpeResult.expected_max_sharpe`). Only the serialised keys moved,
because the ambiguity was in the artefact a reader sees rather than in the code
that reads it.

---

### Checklist

- [ ] Install by `backtest-credibility-audit`; invoke `btca`.
- [ ] Replace `version("backtest-audit")` with the new distribution name.
- [ ] Rename `max_drawdown` → `additive_peak_to_trough` **and re-derive any
      threshold**; the unit changed.
- [ ] Pass `periods_per_year` wherever you need an annualised figure; handle
      `None` and the `NOT AVAILABLE` string.
- [ ] Update `compare_performance` column names to the `_per_period` /
      `_annualised` pair.
- [ ] Supply `trial_sharpes` to `deflated_sharpe` if you gate on `passed is
      True`; otherwise expect `INCONCLUSIVE`.
- [ ] Update parsers for the renamed serialisation keys, and re-derive anything
      compared against stored `execution_report` Sharpe figures.
