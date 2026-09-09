# CLAUDE.md

Working rules for this repository. `SPEC.md` is the contract, `PLAN.md` the build
order, `AI_NOTES.md` the incidents these rules come from — read those first.

## Status: feature-frozen

Correctness, naming and documentation fixes only. `pyproject.toml` owns the version.

- Do not add modules, metrics, roadmap items or dependencies. An idea that does
  not meet every acceptance criterion belongs in an issue, not in `src/`.
- Declined: Sortino, Omega, CVaR — credibility decomposition, not performance.
- Deferred: engine adapters — the four-column contract already exports. See PLAN.

## Acceptance criteria, as checks to run

1. **Known-truth only.** Assert against synthetic panels whose generative
   decomposition is explicit, never against stored output from a previous run — a
   golden file freezes today's bugs into the expected answer.
2. **Paired control.** Every detection test ships with the honest case that must
   not fire. A check that flags correct data trains its users to ignore it.
3. **Monotone in severity.** Where a scenario has a dial, assert the metric moves
   with it. Responding to presence alone is not detection.
4. **Undefined is not zero.** Degenerate correlations return `None` with a count,
   never `0.0`. `SKIPPED` (never ran) and `INCONCLUSIVE` (ran, evidence could not
   support a conclusion) stay distinct into the report; their remedies differ.
5. **Wired, and exercised by the demo.** Importable and tested is not delivered.
   A registered module appears in text and JSON output even when it did not run,
   with its reason code and required evidence; registry and report keys match.
   A demo case must trigger it — the only check that catches a silent skip.
6. **Limits documented.** Each module states what it cannot catch.

## Test rules

- **Never weaken an assertion to make an implementation pass.** If a deliberate
  contract change makes an old assertion wrong, the test may change only with a
  written note: old behaviour, target behaviour, migration impact.
- Partition interpretation tests by semantics — negative / near-zero / positive,
  both sides of a threshold, finite / non-finite, defined / undefined, PASS /
  FAIL / INCONCLUSIVE — not by branch. Assert the number implies the sentence.
- Cover cross-module contracts (scope, evaluation dates, provenance) with
  orchestration tests; locally correct functions do not compose into truth.
- When a module reports nothing, check the fixture before concluding it is wrong.

## Correctness invariants

- Demean using training-period statistics only. `Panel.require_train_end` must
  raise, never fall back to the full sample.
- Compare two quantities only over an identical row subset.
- A pass/fail gate may use only a quantity estimable without bias from the data at
  hand; anything else is reported as context, not as a verdict.
- Let unexpected errors propagate. Never turn one into a skipped section or NaN.
- Every report carries the commit and config that produced it; every random
  operation is seeded. Write artefacts with explicit `encoding="utf-8"`.
- Verify release shape: build the wheel, then install and exercise it outside the
  repository, loading package data via `importlib.resources`, not a repo path.
- `make test`, `make lint` and `make demo` must keep running offline with no
  database and no network; a service dependency there is a regression.

## Repository boundary, and what still crosses it

This repository only. `factor-zoo-audit` is a sibling worked on in parallel by
someone else: do not edit it, clone it, or propose changes to it. If a task
appears to require it, stop and say so rather than working around it.

Findings cross that boundary even though edits do not. "Fixed once" is not a
cross-repository invariant: an incident here that reveals a reusable failure class
— packaging shape, provenance, non-finite input, a setting that controls nothing,
a mechanism registered but never reached — creates an explicit audit item for each
sibling repository, recorded here: you do not touch the sibling, and you do not
drop the propagation because you cannot.

## Workflow

- One task per commit, single-purpose. Finish it, then stop for review.
- Never `git add -A`; stage the files you actually changed.
- Never push. Review happens locally and the maintainer pushes.
- State acceptance as a working-tree-checkable result: "N tests pass, demo says X".
- Report what happened: show failing output, say plainly what was skipped.
