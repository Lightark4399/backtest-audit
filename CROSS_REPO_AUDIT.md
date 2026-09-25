# CROSS_REPO_AUDIT

Findings cross the repository boundary even though edits do not. When an
incident in any sibling repository reveals a reusable failure class, this file
records **this repository's check against that class** — nothing more.

The incident body lives where the defect occurred. This file carries only the
upstream identifier, a link, the date, and what the check here found. Two bodies
drift; one body and a pointer do not.

References here name a path and quote the phrase they rely on, never a line
number. Line numbers were tried and two of six went stale within days — both
moved by the very commits that corrected the claims they pointed at. A path and
a quotation locate a claim and survive the edit; a line number adds drift and
locates nothing a search would not.

## Release rule

**A pending item in this file blocks a release and blocks a merge to `main`.**
Pending means the check has not been run, or it found something that is still
open. There is deliberately no time-based rule: "within two weeks" is not
enforceable by anything in this repository, and a rule nobody can check is worse
than no rule, for the reason `SPEC.md` gives about checks that teach their users
to ignore them.

Closing an item requires the check result recorded below, not an intention.

---

## Item 1 — inbound: a mechanism that is never reached

| | |
|---|---|
| Origin | `factor-zoo-audit`, `ASSERTION_SCOPE_AUDIT.md`, finding **AS-04** |
| Finding title | universe eligibility must precede magnitude and cleaning |
| Link | https://github.com/Lightark4399/factor-zoo-audit/blob/main/ASSERTION_SCOPE_AUDIT.md |
| Date found upstream | 2026-09-05 |
| Checked here | 2026-09-05 |
| Resolved here | 2026-09-06 |
| Status | **CHECKED_CLEAN** — registry clean; one finding, resolved by correcting the claims; the wiring is a registered future issue |

**Failure class.** A mechanism that is registered, tested and documented, but
never actually reached in execution. Upstream, a point-in-time universe
reconstruction was implemented, tested at the storage layer and advertised in
the README status list, with zero call sites in the factor pipeline: the gate
the project claimed was not connected to the thing it protects.

The constraint that catches this is `SPEC.md` acceptance criterion 5, written in
*this* repository after the validation-protocol audit shipped with fifteen
passing tests and was then silently skipped in every demo case. The rule existed
here and was not propagated, which is why this file exists.

**The coincidence is the evidence.** Both repositories implemented a
calendar-based universe reconstruction, both tested it, both documented it, and
both forgot to reach it — independently, under the same name, found on the same
day by two separate audits. Neither team learned the omission from the other.
Two independent occurrences of one shape are the strongest available argument
that the failure class is structural rather than incidental: the mechanism is
easy to build and easy to test in isolation, and nothing about writing it
prompts anyone to check that the pipeline calls it. That is what makes criterion
5 a rule rather than an observation, and what justifies the cost of this file.

### Check: every stage in `AUDIT_REGISTRY`

Verified dynamically with a call trace over a full demo run, not by string
search: dispatch through the registry, the runner's conditional branches and the
externally attached results do not show up in a grep.

| Stage | Invoked in execution | In coverage manifest | Observable in demo / integration test |
|---|---|---|---|
| `baseline` | yes — `raw_ic`, `demeaned_ic` | yes | yes — every demo report |
| `alignment` | yes — `run_alignment_audit` | yes | yes — every demo report |
| `point_in_time` | yes — `run_pit_audit` (attached, not by the runner, by design) | yes | yes — `pit_report.*` |
| `survivorship` | yes — `run_survivorship_audit` | yes | yes — every demo report |
| `grouping` | yes — `decompose_by_group` | yes | yes — every demo report |
| `significance` | yes — `newey_west_tstat` | yes | yes — every demo report |
| `protocol` | yes — `compare_protocols` | yes | yes — `drifting_report.*` |
| `execution` | yes — `audit_execution_timing` | yes | yes — `execution_report.*` |
| `selection` | yes — `deflated_sharpe` (attached, not by the runner, by design) | yes | yes — `selection_report.*`, and the capped/ledger pair |

All nine chains are intact. Stages that did not run in a given case appear in
that case's manifest as `SKIPPED` with a reason code, and the demo index records
all nine as `COMPLETED` across the independent known-truth cases.

`point_in_time` and `selection` are not invoked by `run_baseline_audit`. That is
deliberate — both require evidence the panel contract does not carry, so the
runner registers `REQUIRES_EXTERNAL_EVIDENCE` and the demo attaches the results.
The chain is complete; it simply does not pass through the runner.

### Finding: `universe_asof` is implemented, tested, documented, and unreached

The registry is clean. A supporting mechanism is not.

- `universe_asof` is a SQL macro in `src/audit/sql/duckdb/001_schema.sql`,
  reached only through `BitemporalStore.universe()`
  (`BitemporalStore.universe` in `src/audit/ingest/duckdb_store.py`).
- `BitemporalStore.universe()` has **no call sites outside `tests/test_pit.py`**.
  The call trace over a full demo run confirms it is never executed.
- It is tested — `tests/test_pit.py` asserts the macro's membership by date, and
  `tests/test_sql_windows.py` asserts it consults `delisting_date`.
- It was documented as the mechanism behind the survivorship audit. At the time
  of the check, `src/audit/audits/survivorship.py` read "That is what
  ``universe_asof`` implements", and `SPEC.md` and `README.md` both said
  "survivorship via universe reconstruction". All three have since been
  corrected — see the decision below; they are quoted here as they stood when
  the finding was made, because that is what the finding was about.
- `run_survivorship_audit` does not call it. It derives the point-in-time arm
  from panel presence instead: `surviving_entities` treats an entity as delisted
  when it is absent from the panel's final dates.

This is the same shape as the upstream defect, and the difference matters. The
panel-derived reconstruction can only see entities that are already in the
panel. A universe backfilled from a survivor list contains no delisted entities
at all, so `delisted_entities` returns the empty set and the audit reports no
attrition — on precisely the data that is most survivorship-biased. The macro
consults an independent listing/delisting calendar and does not have that blind
spot.

**Severity: lower than upstream, and the reason is worth stating.** The module
does not certify. On a panel with no attrition it returns `passed=None` and says
so explicitly: "a panel assembled without delisted entities in the first place
would look exactly like this, and the absence would be invisible here. Check
that the source universe was built from listing and delisting dates." Criterion
6 is satisfied at the point of use. What was wrong is the wiring and the three
documentation claims that described a connection which does not exist.

### Decision: correct the claims, do not wire the mechanism

Taken 2026-09-06. The repository is feature-frozen, and wiring the store-backed
arm requires a bitemporal store the demo path does not have. Adding it now would
create a code path no shipped case exercises — which is the criterion 5 failure
this very item exists to check for, reintroduced by the attempt to close it.

So the three claims were corrected to describe what the module does:

- `src/audit/audits/survivorship.py` now has a section on how it reconstructs
  the universe and what that costs, in the module's own voice, stating that on a
  panel backfilled from a survivor list it reports no attrition — the data it
  would most need to catch. It names `universe_asof` as an unwired alternative
  available to callers holding a store, rather than as its own implementation.
- `SPEC.md`'s channel table reads "survivorship via panel-derived attrition".
- `README.md`'s status list says attrition is derived from panel presence and
  not from a listing calendar, and points at the docstring for the blind spot.

Wiring is registered in `PLAN.md` as a future issue with the condition that
would make it worth doing: a caller holding both a bitemporal store and an
entity table whose listing and delisting dates did **not** come from the same
survivor-filtered source as the panel. Without that second condition the
store-backed arm inherits the same bias by a longer route and buys nothing.

---

## Item 2 — return leg: property tests are not sufficient on their own

| | |
|---|---|
| Origin | this repository's incident 19 (assertion scope), raised to `factor-zoo-audit` |
| Their response | `ASSERTION_SCOPE_AUDIT.md`, findings **AS-03** and **AS-04** |
| Link | https://github.com/Lightark4399/factor-zoo-audit/blob/main/ASSERTION_SCOPE_AUDIT.md |
| Date found upstream | 2026-09-05 |
| Checked here | 2026-09-05 |
| Status | **CHECKED_CLEAN** — two gaps found, both silent-if-broken rather than broken, both registered as future issues |

**Their finding.** A property test is not sufficient by itself, because a
property can hold at the helper layer while the pipeline never calls the helper
— which is exactly the `universe_asof` shape. Their conclusion: both layers are
needed. Property tests over all entry points, *plus* incident regressions that
pin the specific call chain that was once broken.

That is correct, and it is the sharper statement of incident 19's own lesson. A
property scoped to a helper is scoped to a place after all — the place being a
function rather than a file.

### Check: do this repository's property tests have the same gap?

Two do. Neither is broken today; both would stay silent if they broke.

1. **`format_interpretation`'s semantic partition tests** (`tests/test_report.py`,
   written after incident 15) exercise the helper directly across negative,
   near-zero, positive, undefined and threshold-boundary inputs. Nothing asserts
   that a rendered report contains a `READING` section at all. `render_report`
   does call it, and the demo output does carry the section — verified — so the
   chain is intact. But if it stopped being called, every one of those semantic
   tests would still pass, and the interpretation that incident 15 exists to
   protect would vanish from the delivered report unnoticed.

2. **The generic provenance-surface property**
   (`test_every_provenance_field_reaches_both_rendered_surfaces`) asserts that
   every key in `SelectionProvenance.to_dict()` reaches the JSON and the text.
   It runs against results constructed inside the test and calls
   `format_selection` directly. Its pipeline counterpart,
   `test_the_demo_shows_the_cap_downgrading_a_verdict`, reads demo artefacts but
   checks named fields rather than the generic property — so the generic form is
   asserted only at the helper layer.

Where the two layers already exist, they were built for the reason their audit
gives: the annualisation property is asserted both at the helper
(`annualisation_label` against the arithmetic it describes) and over every
surface a demo run produces, and the cap is asserted both against
`format_selection` and against the artefacts the demo writes.

### Disposition: registered, not closed by fixing

Both gaps are silent-if-broken rather than broken. The call chains they fail to
pin are intact today, verified directly: `render_report` calls
`format_interpretation` and the READING section is present in delivered output,
and the demo does write selection reports carrying the provenance fields. So
neither is a defect in the shipped artefact; each is a test that would not notice
if one became a defect.

They are registered in `PLAN.md` as a future issue rather than treated as
outstanding work, so this item is CHECKED_CLEAN and does not block a release.
Recording an accepted gap is a different act from leaving one open: the check ran,
the result is written down, and the remedy has a place to live. What would make
this item OPEN again is a *new* class of gap found by a later audit, not the
continued existence of these two.

The remedy, when it is taken, is the pairing AS-03 states: a property over all
entry points, plus an incident regression pinning the specific call chain that
was once broken.

---

## Item 3 — outbound: verdict semantics that claim more than the check measures

| | |
|---|---|
| Origin | this repository, branch `claude/audit-semantics-closeout` (audit-semantics closeout) |
| Raised to | `factor-zoo-audit` — **not yet raised**; this repository does not edit or clone the sibling |
| Date found here | 2026-09-23 |
| Status | **PENDING** — checked here and corrected within the scope listed under "Check here"; points still open here are listed under "Open here" below; not yet raised with `factor-zoo-audit` |

**Failure classes.** Four, each reusable by any audit that gates on a scalar:

1. **A fixed cut-off described as a statistical bound.** Every `MATERIAL_GAP`
   comment read "indistinguishable from (estimation) noise". The value is a
   chosen materiality level on the difference of two point estimates, never
   calibrated to sampling error.
2. **A heuristic described as the estimator it approximates.** The
   `effective_n` docstring said it was "the same information the HAC correction
   uses". It is an AR(1) formula in the lag-1 autocorrelation; dependence at
   higher lags can leave it at `n` while the HAC standard error differs
   materially from the naive one.
3. **A non-finite statistic falling through to PASS.** The finite check guards
   only the FAIL branch (`np.isfinite(gap) and gap > MATERIAL_GAP`), and the
   final `else` assigns `passed = True`, so an undefined gap is reported as a
   pass whose verdict prints `nan`.
4. **A comparison arm named after what the caller was supposed to supply.** The
   survivorship report labelled the input panel "Point-in-time universe"; the
   module never checks or reconstructs membership.

### Check here

- Class 1: all four `MATERIAL_GAP` comments (`pit`, `protocol`, `survivorship`,
  `grouping`) and the `MIN_TESTABLE_IC` comment in `execution` now say "not a
  significance bound"; within-threshold PASS verdicts name the fixed threshold.
  The grouping verdicts, docstrings and report treat pooled, within-group and
  between-group IC as three views rather than additive parts: the FAIL verdict
  no longer says the score comes from ranking groups, the within-threshold PASS
  no longer claims positive within-group ability, the opposite-direction PASS
  states only the measured figures, and the report line "Level effect" is now
  "Pooled minus within-group" with a one-line caveat (the JSON key
  `level_effect` is kept). The `significance` module docstring no longer says
  positive autocorrelation always makes the naive standard error too small.
  Values and operators unchanged.
- Class 2: the `effective_n` docstring, the report line (now
  "effective n (AR(1))") and the note say AR(1) approximation. The note no
  longer infers from a positive lag-1 autocorrelation that the naive
  t-statistic is optimistic. `tests/test_significance.py` pins the lag-2 case
  and a positive-lag-1 series whose HAC standard error is below the naive one.
- Class 3: `protocol` and `grouping` return INCONCLUSIVE on a non-finite gap;
  `execution` and the alignment shift checks guard it before any verdict.
  `survivorship` and `pit` did not: both were reproduced through the public API
  (old verdicts: "moves the demeaned IC by only +nan", "agree to within nan").
  Fixed by a separate, approved behaviour change: a non-finite gap now gives
  `passed=None` and an INCONCLUSIVE verdict naming the undefined arm. The
  no-revisions and no-attrition paths keep precedence. Finite cases are
  unchanged. The regressions `test_undefined_gap_is_not_a_pass` and
  `test_undefined_gap_with_revisions_is_not_a_pass` also check the JSON, the
  text report and the coverage manifest. The text report's IC and gap lines in
  both blocks now print `undefined` instead of `+nan`, through the existing
  `_fmt` helper; finite values keep the same format and column, and the JSON
  still carries the non-finite values unchanged. Related, not the same defect, and
  deliberately left for a separate decision: the alignment shuffle check
  (`ok = np.isfinite(perturbed) and`) reports a non-finite result as FAIL
  rather than INCONCLUSIVE.
- Class 4: the report line and docstrings now say "As-supplied panel"; the
  JSON keys `pit_ic` and `pit_demeaned_ic` keep their names for compatibility.
  The module docstring describes the gap as the score's sensitivity to keeping
  only the entities seen in the tail window (the final `tail_dates` dates, the
  last date by default), to be read as survivorship bias only once the caller
  has established the input's membership basis. The verdicts and the report
  describe that window rather than "the final date" when `tail_dates > 1`, and
  state the measured tail-window-subset versus as-supplied gap: "Gap
  attributable to survivorship" became "Gap (tail-window subset minus
  as-supplied)", and the verdicts no longer infer
  that absent entities were delisted, were harder or easier to predict, or
  that the sample was selected on outcome. README's +0.051 is labelled as a
  synthetic example with a known generator. JSON keys (`survivors_*`, `pit_*`,
  `n_entities_delisted`, `gap`) are kept; the result docstring states what
  each holds.

### Open here

Recorded, not corrected, in this batch:

- The alignment shuffle check (`ok = np.isfinite(perturbed) and`) reports a
  non-finite result as FAIL rather than INCONCLUSIVE.
- `survivorship` scores both arms over the same nominal evaluation window, but
  computes each IC independently, so the dates each arm can actually score
  (enough entities, non-constant cross-section) need not coincide. The gap can
  therefore mix a composition difference with a difference in scorable dates.
  The module docstring states this boundary; the algorithm is unchanged.
- `survivorship` does not validate `tail_dates`. `surviving_entities` slices
  `dates[-tail_dates:]`, so `0` selects every date, a negative value `-k`
  selects all but the first `k` dates, and a value above the number of panel
  dates selects every date. The verdicts and report still describe the window
  as "the final N dates" (e.g. "the final 0 dates", "the final -1 dates"), so
  the text and the slice disagree. Pre-existing; input validation or a change
  to the slice is a separate behaviour decision, not made in this batch.

This list is what was found by the checks in this item, not the result of a
general audit of the repository; other open points may exist.

Closing this item requires the four classes raised with `factor-zoo-audit` by
the maintainer, and a decision on each point under "Open here".
