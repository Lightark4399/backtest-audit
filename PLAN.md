# PLAN

Module breakdown, the order things were built in, and how each was verified.

---

## Architecture

Three layers, deliberately separable so each can be tested without the others:

```
panel.py            data contract — one time convention, enforced
   │
   ├── metrics/     pure functions over a Panel; no I/O, no database
   ├── audits/      checks that compose metrics into verdicts
   ├── ingest/      bitemporal storage (DuckDB); only the PIT audit needs it
   └── report/      pure formatting; no computation
```

The metric layer depends on nothing but pandas and numpy. That is why the test
suite and the demo run in a bare container with no service to provision, and why
`make demo` works on a machine that has never seen a database.

**The data contract is the whole interface.** A panel row is
`(entity_id, event_date, prediction, label)` where `event_date` is when the
target is *realized*, and the prediction must have been computable strictly
before it. Any pipeline in any language can produce that CSV.

---

## Build order and verification

Each stage was verified before the next began. Where verification produced a
surprise, the surprise is recorded in `AI_NOTES.md` and the design changed.

### M1 — Baseline decomposition

**Modules.** `panel.py`, `metrics/ic.py`, `metrics/baselines.py`,
`metrics/partial.py`, `metrics/significance.py`, `report/text.py`, `run.py`,
`cli.py`, `demo.py`, `synthetic.py`.

**Verification.**
- Perfect prediction → IC exactly 1.0; independent prediction → IC ~0
- Rank IC invariant under monotone transform (catches a Spearman that forgot to rank)
- Zero-skill-with-level-knowledge → raw IC 0.63, demeaned IC 0.0006
- Demeaned IC monotone in the generative skill parameter
- Full-sample demeaning refused, not silently substituted
- HAC standard error exceeds naive under positive autocorrelation
- SQL window frames checked statically, so the check runs without a database

**Surprises that changed the design.** Two, both in `AI_NOTES.md`: the
increment statistic was biased twice before it was right, and the
cross-sectional-mean baseline turned out to be *undefined* rather than ~0.

### M2 — Alignment audit and example pipelines

**Modules.** `audits/alignment.py`, `examples/pipelines.py`.

**Verification.**
- Shuffle preserves every marginal (only the pairing changes)
- Shuffle is deterministic given a seed
- Shift takes the label from the intended date and never borrows across entities
- No baseline changes when a same-day or future label is perturbed
- A deliberately misaligned panel is caught; honest panels are not flagged

**Surprises.** Three, in `AI_NOTES.md`: the level blinds the shift test; the
backward shift cannot be a verdict; the persistence benchmark cannot be a gate.

### M4 — Validation protocol and effective sample size

**Modules.** `audits/protocol.py`, additions to `metrics/significance.py`.

**Verification.**
- Stationary relationship → inflation −0.0002 (the control)
- Drifting relationship → inflation +0.094, monotone in the drift parameter
- The embargo can only remove information, never add it
- Random K-fold scores every row exactly once
- `n_eff` falls as autocorrelation rises; negative `ρ` awards no bonus

**Surprise.** The first attempt found no inflation at all on a stationary panel
with a low-capacity model. That turned out to be correct behaviour, not a bug,
and it changed the module's claim from "random splitting inflates" to "random
splitting inflates when the relationship drifts" — a conditional the module now
measures rather than assumes.

### M5 — PnL layer

**Modules.** `metrics/performance.py`, a fourth demo section.

**Verification.**
- Positions dollar-neutral and unit-gross on every date
- Positions invariant to monotone rescaling of the prediction
- PnL arithmetic checked against a hand-computable four-entity case
- Demeaned Sharpe orders panels identically to demeaned IC
- Zero-skill panel: raw Sharpe > 20 with a ~100% hit rate, demeaned Sharpe < 10

**Surprise.** The first run produced annualised Sharpes of 147–223 with 100% hit
rates and zero drawdown across every panel, including one with no skill at all.
That turned out to be the level effect expressed in Sharpe units, and it changed
the module: instead of hiding it, the layer reports raw and demeaned Sharpe as a
pair, mirroring the raw/demeaned IC decomposition.

### M3 — Point-in-time, survivorship, grouping

**Modules.** `ingest/duckdb_store.py`, `src/audit/sql/duckdb/001_schema.sql`,
`audits/pit.py`, `audits/survivorship.py`, `audits/grouping.py`.

**Verification.**
- A revision is stored as a new row; the original belief survives
- As-of returns the pre-correction value before the correction lands
- `knowledge_date < event_date` is rejected by the schema
- No revisions → gap exactly 0 (not approximately)
- Gap monotone in the revision rate: +0.045 / +0.101 / +0.178
- The restated arm is invariant to the revision scenario
- Uncoupled attrition → survivorship gap ~0
- A prediction encoding only the group → pooled +0.935, within +0.005

**Surprises.** Two: the PIT control exposed a date-sampling confound, and the
first survivorship generator was too weak to demonstrate anything.

---

## Current status

Version 0.2.0 has a registry-backed audit coverage manifest. The panel contract,
baseline decomposition,
alignment, point-in-time vintage comparison, survivorship, grouping, validation
protocol, PnL, execution timing and selection-bias modules are implemented. The
runner emits one out-of-sample test-period credibility report; it no longer
advertises train/all scopes that some modules could not honour consistently.
Every one of the nine shipped audit channels now appears in text and JSON even
when it did not run, with a closed reason code and the evidence required to make
it runnable. The demo writes an index across independent known-truth cases rather
than pretending those cases share one audit scope.

The release boundary is verified as well as the source tree: CI builds a wheel,
installs it into a clean environment outside the repository, instantiates
`BitemporalStore`, runs the PIT schema smoke test, then runs the demo. Runtime SQL
is a package resource rather than an assumed repository-relative file.

### M6 — Execution timing and selection bias

**Modules.** `audits/execution.py`, `audits/selection.py`.

**Verification.**
- A look-ahead signal: IC 0.878 at lag 0, 0.013 at lag 1 — flagged
- An honest forecast: no edge at lag 0, edge at lag 1 — named and passed
- Decay ratio monotone in the degree of look-ahead
- Best of 42 pure-noise candidates fails; the same returns as a single test pass
- BH keeps genuinely strong candidates and rejects all of the noise
- The convenience p-value is named `iid_normal_pvalue`, marked exploratory, and
  cannot by itself produce a strong PASS

---

## Feature freeze

Version 0.2.0 is feature-frozen after correctness and coverage hardening.
It is a minor rather than a patch release: the distribution and CLI are
renamed, a reported metric is renamed and changes units, an observation
frequency parameter is required for annualisation, and the selection module
gains a verdict ceiling. `MIGRATION.md` states what breaks and what to do.

The items below are of two kinds, and the difference is not cosmetic. **Declined**
means the thing conflicts with what this repository is for and will not be built
here whoever asks. **Future issue** means it is compatible and unbuilt, waiting
on a stated condition. Filing a decline as a future issue leaves it looking
merely unfinished, which invites someone to finish it.

### Declined: Sortino, Omega, CVaR and other performance ratios

Not deferred. They will not be added.

This repository decomposes a score into the part any naive baseline gets for
free and the part attributable to the model. Every number it ships is either one
half of such a decomposition or a statement about what it cannot measure. The
PnL layer is the closest thing to an exception, and its own docstring says the
Sharpe register is not the better one — it exists to restate an existing finding
in the units a reader recognises, and it reports raw and demeaned as a pair
because the gap between them *is* the finding.

Sortino, Omega and CVaR are performance measures. They characterise a return
distribution; they do not separate free from earned, and there is no demeaned
counterpart of a CVaR that means anything. Adding them would grow the surface on
which incident 12 happened — a presentation layer reproducing the level effect
before exposing it — while adding no capacity to decompose anything. The result
would be a second-rate performance library bolted to a credibility auditor,
worse at the first job than the tools that specialise in it and muddier at the
second, which is the only job it has.

`SPEC.md` states the position this follows from: not a strategy, no alpha
claimed, sought or implied. A reader wanting downside-risk ratios is better
served by a library that does that and nothing else, run on the same returns.

### Future issue: true purge and CPCV

Add per-sample feature and label information intervals, remove overlapping
training rows, then generate combinatorial purged paths. The current module is
honestly named `embargoed_walk_forward`; a date gap is not a purge.

### Future issue: TrialLedger and PBO

**The ledger is a precondition for computing the Deflated Sharpe Ratio as
defined, not a matter of traceability.** Bailey and Lopez de Prado define the
benchmark `SR*` in terms of `V[{SR_n}]`, the variance of the Sharpe estimates
*across the N trials*. That is a property of the search, and it needs the Sharpe
of every configuration examined. Without it `deflated_sharpe` substitutes the
sampling variance of the winner's own estimator, which coincides with `V[{SR_n}]`
only for independent, identically distributed trials. Measured on synthetic
candidate sets the two differ by 0.003x on a correlated parameter grid and 29.5x
on a heterogeneous one, so the substitution is not a rounding matter and its
direction is not knowable in advance.

Until the ledger exists the module reports `method=proxy`, renames its headline
figure `approximate_selection_adjusted_sharpe_per_period`, and caps the verdict at
INCONCLUSIVE. Supplying `trial_sharpes` today already produces the formal
statistic; what is deferred is the contract that makes a caller record those
Sharpes as a matter of course.

Make every searched configuration an explicit input so winners can be traced to
the complete candidate set. Add CSCV/PBO only after the ledger contract exists;
never infer search size from the winning return stream.

### Future issue: engine adapters

**Deferred, not declined, and the distinction was examined rather than assumed.**
The instinct is to decline it under `SPEC.md`'s "not a backtesting engine", but
an adapter does the opposite of embedding an engine: it keeps the engine outside
and takes its output. Declining it on that ground would misread the non-goal.

The stronger argument for declining is that it adds no capability. The panel
contract is four columns, and `PLAN.md` already says any pipeline in any language
can produce that CSV — so the export contract exists, and an adapter only saves a
user the transformation. That makes it a convenience, and conveniences wait for
someone who actually wants one; it does not make it out of scope.

So it stays a future issue, with the condition stated: a user who holds VectorBT
or Backtrader output and cannot readily produce the four columns themselves.
Absent that, each adapter is a standing coupling to a third-party schema that
changes on someone else's timetable, for a transformation the user could write
once.

The design constraints, unchanged, and they are the reason doing it properly is
harder than it looks. Define a narrow export contract for CSV, VectorBT and
Backtrader results. Do not embed or reproduce those engines. Missing trial,
universe or timing evidence must yield an explicit evidence gap rather than a
clean verdict — an adapter that turned a partial export into a clean report
would be this framework certifying a pipeline it could not see, which is the
failure `SPEC.md` names first. That last requirement is newly representable: the
coverage registry now carries `SKIPPED` with `REQUIRES_EXTERNAL_EVIDENCE`, so an
adapter has somewhere honest to put what an engine did not export.

### Future issue: robust p-values for candidate screening

Accept HAC or block-bootstrap p-values and pass them to
`benjamini_hochberg`. Until then, `screen_candidates` remains an explicitly
exploratory iid-normal approximation.

### Future issue: a store-backed survivorship arm

`run_survivorship_audit` derives its point-in-time arm from panel presence: an
entity counts as delisted when it is absent from the panel's final dates. That
cannot see an entity the panel never contained, so a universe backfilled from a
survivor list reports no attrition — the case the audit most needs to catch. The
module states this in its own verdict and docstring.

`universe_asof` in `src/audit/sql/duckdb/001_schema.sql`, reachable through
`BitemporalStore.universe()`, already implements calendar-based membership and is
tested. Wiring it would give the audit a second arm without that blind spot.

**The condition that would make it worth doing:** a caller who holds a
bitemporal store *and* an entity table carrying listing and delisting dates that
did not come from the same survivor-filtered source as the panel. Absent that
second condition the store-backed arm inherits the same bias through a longer
path and buys nothing. The demo has no such store, so wiring it now would add a
code path no shipped case exercises — which is the criterion 5 failure this very
audit was raised to check for.

### Future issue: pipeline counterparts for two helper-layer properties

Both are silent-if-broken rather than broken; `CROSS_REPO_AUDIT.md` records the
check that found them.

- The `format_interpretation` semantic partition tests, written after incident
  15, assert nothing about the rendered report. `render_report` does call it and
  the READING section is in the delivered output, but every one of those tests
  would still pass if that call disappeared.
- `test_every_provenance_field_reaches_both_rendered_surfaces` asserts its
  generic property against results constructed inside the test. Its pipeline
  counterpart checks named fields instead, so the generic form is asserted only
  at the helper layer.

The remedy in both cases is the pairing `factor-zoo-audit` arrived at
independently in AS-03: a property over all entry points, plus an incident
regression pinning the specific call chain.

### Future issue: full point-in-time training replay

Rebuild every training set at its historical cutoff from the observations and
universe then knowable. The existing PIT comparison measures evaluation-vintage
impact and must not be described as full historical retraining.

---

## Scope discipline

`SPEC.md` sets the bar: a module ships only when it meets all six acceptance
criteria, including being wired into the report. The failure mode to avoid is a
repository with ten half-built modules instead of five finished ones — the second
is a project, the first is a graveyard.
