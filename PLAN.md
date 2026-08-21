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

**Modules.** `ingest/duckdb_store.py`, `sql/duckdb/001_schema.sql`,
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

95 tests. Six inflation channels, five with a dedicated module. Every module
wired into the text and JSON reports. CI runs lint, tests, and the demo end to
end on every push.

---

## Roadmap

Ordered by expected value, not by ease. Nothing here blocks the project being
complete — these are extensions, and the repository is usable without them.

### M6 — Execution timing and selection bias

**Modules.** `audits/execution.py`, `audits/selection.py`.

**Verification.**
- A look-ahead signal: IC 0.878 at lag 0, 0.013 at lag 1 — flagged
- An honest forecast: no edge at lag 0, edge at lag 1 — passes, with the profile
  named rather than treated as anomalous
- Decay ratio monotone in the degree of look-ahead
- Best of 42 pure-noise candidates fails; the same returns as a single test pass
- BH keeps genuinely strong candidates and rejects all of the noise

**Surprise.** The first verdict logic reported the healthiest profile — no edge
at lag 0, edge at lag 1 — as INCONCLUSIVE, because it only looked for decay. A
pure forecast has nothing to decay *from* at lag 0. Restructured to name that
profile explicitly.

### Next

**A thin PnL layer.** Signal → position → return → Sharpe, drawdown, equity
curve, sitting on top of the existing panel contract without changing it. Not
because Sharpe is a better metric than IC — for model-validation work it is not —
but because "Sharpe 3.2 with a 4% drawdown" is legible to a reader who would
skim past "demeaned IC 0.0006". The audit machinery underneath stays as it is.

### Later

### Issues, not roadmap

Ideas that would broaden the project without deepening its central claim:
parameter-neighbourhood continuity (peak vs plateau), stationarity testing
(ADF/KPSS), the spurious-regression Monte Carlo. Worth doing, not worth delaying
a release for.

---

## Scope discipline

`SPEC.md` sets the bar: a module ships only when it meets all six acceptance
criteria, including being wired into the report. The failure mode to avoid is a
repository with ten half-built modules instead of five finished ones — the second
is a project, the first is a graveyard.
