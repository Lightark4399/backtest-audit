# SPEC

What this project is for, what counts as done, and what would count as failure.

---

## The research question

> Given a backtest result, how much of it is real?

Not "is this strategy profitable" — a question that requires knowing the future.
The question here is narrower and answerable now: **of the score a backtest
reports, how much could have been obtained without any predictive insight at
all, and how much depended on information that was not available at the time?**

That decomposition is the entire product. Everything in the repository exists to
produce it or to defend it.

---

## Why the question is worth asking

A high evaluation metric is not evidence of skill. It can also be produced by the
structure of the target. For a persistent target — realized volatility, log
volume, intraday range — decompose it as

```
y(i, t) = level(i) + deviation(i, t)
```

For such targets the cross-sectional variance of `level(i)` dominates that of
`deviation(i, t)`. Any prediction recovering `level(i)` — which a per-entity
intercept does automatically, without learning anything — scores a high
cross-sectional IC while carrying **zero** information about dynamics.

The headline number is then close to a tautology: entities that were volatile
remain volatile. True, useless, and indistinguishable from skill if you look at
one number.

The same reasoning applies to five other channels, each with its own module. The
framework enumerates them rather than treating "leakage" as one undifferentiated
worry:

| Channel of inflation | Module |
|---|---|
| Free baseline / entity level | baseline decomposition, demeaned IC |
| Time-alignment error | shuffle / forward-shift audit |
| Data revised after the fact | point-in-time vs restated |
| Universe missing its failures | survivorship via universe reconstruction |
| Group-level differences | within/between group decomposition |
| Autocorrelation inflating significance | Newey-West HAC |

---

## Acceptance criteria

A module is done when all of the following hold. Code that satisfies some but
not all of these is not done, regardless of how much of it exists.

**1. It is correct on a case whose answer is known by construction.**
Assertions are against synthetic panels whose generative decomposition is
explicit — not against stored outputs from a previous run, which would lock in
whatever bugs exist today.

**2. It does not fire on a correct input.**
Every detection test is paired with a control. A check that flags honest data
teaches its users to ignore it, at which point it protects nothing. The controls
are as load-bearing as the detections:

- Shuffled labels → IC collapses to ~0
- Zero-skill-with-level-knowledge → demeaned IC ~0 despite raw IC 0.63
- No revisions → point-in-time gap exactly 0
- Attrition uncoupled from predictability → survivorship gap ~0
- Groups unrelated to the prediction → pooled ≈ within-group

**3. It responds to severity, not merely to presence.**
Where a scenario has a dial, the metric must be monotone in it. A module that
flagged every scenario equally would be responding to the scenario existing
rather than to how bad it is.

**4. It reports "undefined" and "inconclusive" as distinct from "zero" and
"pass."** "Not measurable" and "measured, found to be nil" are different claims.
Collapsing them is how a summary statistic misleads.

**5. It is wired into the report.**
A module whose output no one sees is not delivered. No exceptions — this is the
rule that keeps the repository from accumulating orphaned code.

**6. Its limits are documented.**
Each module states what it cannot catch. See the falsification standard below.

---

## What would count as failure

Stated in advance, so the project can be judged rather than merely admired.

**The framework fails if it certifies a leaky pipeline as clean.**
The example pipelines carry five switchable defects for exactly this test. The
current, honest position: the alignment audit catches misalignment; it does
**not** catch a contemporaneous feature, because the pairing there is genuinely
correct and only the vintage is wrong. That is documented, not hidden. If a
future change caused the framework to report "all passed" on the leaky pipeline
without qualification, the framework would have failed.

**The framework fails if it flags honest work.**
Tracked by the control tests above. A false-alarm rate high enough to be ignored
is functionally equivalent to no framework at all.

**The framework fails if its own metrics are inflated by the effects it detects.**
It would be incoherent for a tool that detects leakage to leak. Concretely:
demeaning must use training-period statistics only, and `Panel.require_train_end`
raises rather than defaulting to the full sample. This has already been violated
once and fixed — see `AI_NOTES.md`.

**The framework fails if a reported number cannot be re-derived.**
Every report carries the git commit and configuration that produced it. Every
random operation is seeded. A number a reviewer cannot reproduce is an assertion,
not a result.

---

## Explicit non-goals

- **Not a strategy.** No alpha is claimed, sought, or implied.
- **Not a backtesting engine.** It audits results produced elsewhere; the input
  contract is four columns.
- **Not a general leakage detector.** It enumerates specific channels and is
  candid that the list is not exhaustive.
- **Not a claim that these baselines are the right ones.** A model beating
  everything here is not thereby proven useful — only not obviously free. A
  fitted GARCH or an implied-volatility quote would be a stronger control and is
  not implemented.

---

## Scope discipline

The largest risk to this project is scope creep, so the constraint is written
down: **a module ships only when it satisfies all six acceptance criteria,
including being wired into the report.** Ideas that do not yet meet that bar
belong in issues, not in `src/`.
