# CROSS_REPO_AUDIT

Findings cross the repository boundary even though edits do not. When an
incident in any sibling repository reveals a reusable failure class, this file
records **this repository's check against that class** — nothing more.

The incident body lives where the defect occurred. This file carries only the
upstream identifier, a link, the date, and what the check here found. Two bodies
drift; one body and a pointer do not.

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
  (`src/audit/ingest/duckdb_store.py:230`).
- `BitemporalStore.universe()` has **no call sites outside `tests/test_pit.py`**.
  The call trace over a full demo run confirms it is never executed.
- It is tested — `tests/test_pit.py:117-118`, and `tests/test_sql_windows.py:113`
  asserts the macro consults `delisting_date`.
- It was documented as the mechanism behind the survivorship audit. At the time
  of the check, `src/audit/audits/survivorship.py:19` read "That is what
  ``universe_asof`` implements", and `SPEC.md:49` and `README.md:569` both said
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
