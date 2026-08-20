# AI_NOTES

How this repository was built with AI assistance, what went wrong, and the
constraints added in response.

Every incident below actually occurred during development. The numbers are the
ones observed at the time, and each fix is visible in the commit history.

---

## Why keep this file

The project is a tool for detecting inflated backtest results. It was written
with AI assistance. During development the assistant produced **an inflated
metric inside the tool that detects inflated metrics** — twice, in two different
ways, before it was right.

That is not an embarrassment to be buried. It is the strongest available evidence
for the thing the project claims: that a plausible-looking number needs an
adversarial check, and that the check has to be run against a case whose answer
is already known. A framework asserting this while hiding its own near-misses
would be arguing against itself.

---

## Incident 1 — the increment metric was biased, twice

**What happened.** The framework reports how much a model adds over a naive
baseline. The first implementation used `corr(ŷ − b, y − b)`: subtract the
baseline from both sides and correlate what remains. It reads as "removing the
baseline", and it survived review — including mine.

It is wrong. The shared `−b` term is common to both arguments and induces
correlation of its own. On a real job's output it produced a **delta_ic of 0.664
against a raw IC of 0.640** — an increment larger than the total, which is
incoherent.

**How it was caught.** Not by inspection. By constructing a panel with *exactly
zero* genuine skill and checking what the metric said. It said 0.2194.

**The second bias.** Replacing it with the textbook partial correlation

```
r_xy·b = (r_xy − r_xb·r_yb) / sqrt((1 − r_xb²)(1 − r_yb²))
```

was still not enough. Every baseline is a **noisy proxy** for the entity level —
`persistence` is a single observation, carrying a whole period of transient noise
alongside the level. Controlling for a mismeasured proxy leaves residual level
information in both residuals, and the model gets credited for it. On the same
zero-skill panel the corrected formula still read **+0.2194** with a persistence
control and **+0.1045** with a training-mean control.

**The fix.** Demean all three series by their own training means *before* taking
the partial correlation. Each series' level is then estimated from itself rather
than imported from a mismeasured stand-in. The statistic reads **−0.0022** at
zero skill and rises monotonically with true skill.

**Constraint added.** Every metric must be evaluated on a synthetic panel whose
generative decomposition is explicit, including a zero-skill case. No metric
ships on the strength of its derivation alone. The biased variant is kept behind
a flag so the demo can display the size of the bias, and it carries a warning in
its output metadata.

---

## Incident 2 — the spec was wrong and the code was right

**What happened.** The written spec said the cross-sectional-mean baseline
"should give an IC close to 0." Implementing it revealed that the value is
constant within each date, so its correlation has a **zero denominator** — the IC
is *undefined*, not zero.

**Why it matters.** "Not measurable" and "measured, found to be nil" are
different claims about a model. Reporting 0.0 would assert the second when only
the first is true.

**Constraint added.** Degenerate correlations return `None` and are excluded from
averages with a count, never coerced to 0.0. The distinction propagates all the
way to the report, which prints `undefined` with a reason. The same rule later
produced the `inconclusive` verdict class in the alignment audit, and a test now
pins it down.

---

## Incident 3 — the check was blinded by the effect it was checking for

**What happened.** The shift test scores predictions against a neighbouring
date's labels; a correct result should degrade. Run on the raw IC, a one-day
shift moved a genuinely skilful model's score by **5.5%** and a *zero-skill*
model's by **0.1%**. Neither is a usable signal.

**Diagnosis.** The per-entity level is constant in time, so shifting labels does
not disturb it at all. The level does not merely inflate the headline number — it
also blinds the test meant to validate it.

**The fix.** Run the shift test on demeaned series. The same shift then moves the
skilful model's score by 14%.

**Constraint added.** When a check has no power, say so rather than reporting a
pass. Where the underlying signal is too weak to test, the module returns
`inconclusive` with the reason instead of a green tick.

---

## Incident 4 — a threshold that failed honest work

**What happened.** The shift test initially required the drop to exceed a
persistence-derived benchmark: shifted IC should be about `ρ · base`, where `ρ`
is the label's autocorrelation. On a panel honest by construction, the test
**failed**.

**Diagnosis.** Observation noise in the label attenuates the *measured*
autocorrelation, while the prediction correlates only with the *signal* part. So
`ρ · base` systematically understates what a correct model achieves, and gating
on it produces false alarms.

**The fix.** The persistence figure is reported as context for reading the
magnitude of the drop, but the verdict rests on an ordering that survives the
objection: a correctly aligned prediction must not score *better* against a
neighbouring date's labels than against its own.

**Constraint added.** A pass/fail gate may only use a quantity that can be
estimated without bias from the data at hand. Anything else is reported as
context.

---

## Incident 5 — a verdict that would have failed nearly every honest model

**What happened.** The backward shift (scoring against the *previous* date's
labels) was implemented as a pass/fail check. On the clean example pipeline —
honest by construction — it raised the demeaned IC from **0.7138 to 0.9663** and
returned FAIL.

**Diagnosis.** Any forecaster of a persistent target is built from lagged labels,
so its prediction resembles the label it was *built from* more than the one it
forecasts. High backward correlation is the norm, not a defect.

**The fix.** The backward shift is now diagnostic only. It never asserts a
verdict, is excluded from the summary counts, and is reported with the reading
that explains it.

**Constraint added.** Before a check becomes a gate, it must be run against a
pipeline that is correct by construction. Detection tests and false-alarm tests
are written together; neither ships alone.

---

## Incident 6 — the control exposed a confound in the comparison

**What happened.** The point-in-time audit compares a restated backtest against
one reconstructed as of each date. The control case — no revisions at all — must
show a gap of zero. It showed **−0.0133**, larger than the threshold the module
uses to call a result material.

**Diagnosis.** Not a vintage effect. The as-of reconstruction is *sampled*
(each date costs a pass over the history), while the restated arm used every
evaluation date. The two panels differed in composition as well as in vintage, so
the gap was partly a comparison of different samples.

**The fix.** Score both arms on exactly the same evaluation dates. The control
then gives a gap of **exactly zero**, because the two panels become identical. A
test also asserts the restated arm is invariant to the revision scenario, so the
gap can only come from the as-of arm.

**Constraint added.** Any comparison of two quantities must be computed over an
identical subset. This was already the rule inside the partial-correlation module
— all three pairwise correlations use the same rows — and the incident showed it
had not been applied consistently across modules.

---

## Incident 7 — a demonstration that proved nothing

**What happened.** The "leaky" example pipeline was built with a contemporaneous
feature set to the label itself. The pipeline scored an IC of **exactly 1.0000**.

**Why it is a problem.** A pipeline reporting a perfect score would be caught by
inspection in any real review. Detecting it proves nothing about detecting
realistic mistakes, and a demo built on a caricature is a misdirection.

**The fix.** The leak was made realistic: a same-day measurement with an
optimistic timestamp, modelled as a noisy proxy rather than a copy of the label.
The inflation is now large but plausible — demeaned IC 0.8153 against the clean
pipeline's 0.7138.

**Constraint added.** Defects in the example pipelines are written the way they
are actually shipped: full-sample scaling placed next to the data loading, alpha
chosen on the evaluation period, a universe of survivors. If a defect would be
obvious in review, it is not a useful test case.

---

## Incident 8 — the assistant was asked to overstate what a module catches

**What happened.** With five defects in the leaky pipeline, the alignment audit
caught one. Worse, a deliberate off-by-one in an autoregressive pipeline *also*
passed: shifting the predictions turned the defect into contemporaneous leakage,
where the pairing is genuinely correct and only the vintage is wrong.

There was an obvious temptation — tune the scenario until the demo showed
everything being caught.

**What was done instead.** The boundary was documented. `README.md` and the
pipeline module both state which module catches which defect, and that the
alignment audit detects misalignment only when the prediction is not itself built
from lagged labels.

**Constraint added.** A module's limits are part of its deliverable. An audit
tool that overstated its own coverage would fail its own standard, and a reader
who discovers an undocumented gap discounts everything else in the repository.

---

## Incident 9 — a generator too weak to demonstrate its own effect

**What happened.** The first survivorship scenario produced a gap of **+0.0019**,
indistinguishable from noise, and the module appeared not to work.

**Diagnosis.** The generator scattered delistings across the whole history, so
most doomed entities left *before* the evaluation period began and contributed
nothing to either arm's score. The module was fine; the test fixture was diluted.

**The fix.** Delistings now occur within the evaluation window. The gap became
**+0.0512** with the coupling on, and **−0.0019** with it off — the second number
being the one that matters, since it shows the module does not flag attrition
that is unrelated to predictability.

**Constraint added.** When a module reports nothing, check the fixture before
concluding the module is wrong. A test that cannot fail is not evidence.

---

## Incident 10 — the module found nothing, and that was the right answer

**What happened.** The validation-protocol audit was built on the premise that
random K-fold inflates a backtest on panel data. On the first stationary test
panel it reported an inflation of **+0.0015** — essentially nothing.

**The temptation.** Assume the module is broken and adjust until it produces the
expected number.

**Diagnosis.** The module was correct. With a low-capacity model and a stable
relationship, there is nothing for random assignment to exploit: training on a
random subset and training on the past yield the same coefficients. Random
splitting leaks when the relationship **drifts**, because the random folds hand
the model rows drawn from the test period's regime.

**The fix.** A generator with a drifting feature-to-label relationship, and a
claim narrowed to match what is actually true. Stationary → −0.0002. Drift 1.0 →
+0.070. Drift 2.0 → +0.093.

**Constraint added.** When a module reports nothing, establish whether the
scenario contains the effect before concluding the module is wrong. Incident 9
was the same lesson from the other direction, and the difference matters: there
the fixture was diluted, here the premise was too broad. Both were found by
asking what the data should contain rather than what the output should say.

---

## Incident 11 — a module that passed its tests and shipped invisible

**What happened.** The validation-protocol audit was merged with fifteen passing
tests and correct wiring into the report. In every demo case it produced
`"protocol": null`.

**Diagnosis.** The audit refits the model under different splitting schemes, so
it needs feature columns. Every demo panel carried finished predictions and no
features, so the module was silently skipped — which is the correct behaviour for
a missing input, and meant 350 lines of new code were invisible in the shipped
output.

**How it was caught.** Not by the test suite, which passed. The coding agent
noticed it while reviewing the artefacts a sync had changed, and reported it
without being asked. It also flagged a second cosmetic defect in the same pass: a
newly added note ran to 158 characters in a report formatted to 78.

**The fix.** A third demo case built on a panel that carries features, so the
module appears in the output. The panel also needed a real prediction rather than
a constant placeholder — with a constant, every other module's correlation was
undefined and the report came out mostly empty.

**Constraint added.** `SPEC.md` acceptance criterion 5 now requires a demo case
that triggers the module, not merely that it be wired in. Tests prove a module
works; only the demo proves anyone will see it.

---

## Incident 12 — the PnL layer reproduced the deception before it exposed it

**What happened.** A thin PnL layer was added so findings could be stated in
Sharpe units. Its first run reported annualised Sharpes of **147 to 223**, hit
rates of **100%**, and **zero drawdown** — on every panel, including one built
with exactly zero skill.

**The temptation.** Treat it as a bug in the position construction and adjust
until the numbers look plausible.

**Diagnosis.** The numbers were correct. On a sign-constant, persistent target, a
long-short book built from prediction ranks is long the high-level names and
short the low-level ones. Because the level barely moves, that book wins every
single day. The absurd Sharpe *is* the level effect, in different units — exactly
what raw IC reports.

**The fix.** Report raw and demeaned Sharpe as a pair, mirroring the raw/demeaned
IC decomposition. The zero-skill panel now shows Sharpe 147 raw against −4.5
demeaned, with the hit rate falling from 100% to 37.5%. A test asserts the two
metrics order panels identically, since they are meant to be the same information
in different units and disagreement would mean one is measuring something else.

**Constraint added.** When a new presentation layer makes results look better,
the first hypothesis is that it is reproducing a known bias rather than revealing
a new result. This one would have shipped a demo whose headline table was the
very deception the project exists to expose.

---

## Workflow constraints

The rules that emerged, applied to every subsequent session:

**Tests are not to be modified to make them pass.** Stated explicitly in each
task given to the coding agent. It has held: when a lint rule flagged an
over-broad `pytest.raises(Exception)`, the agent stopped and asked rather than
narrowing the assertion on its own initiative. Narrowing it to
`duckdb.ConstraintException` was the right call — the broad form would have
passed on a typo'd table name — but it was a decision about the test's meaning,
and the agent correctly declined to make it unilaterally.

**Every metric is verified against a known-truth case before it is trusted.**
Incidents 1, 3, 6 and 9 were all caught this way and by no other means. Reading
the code was not sufficient in any of them.

**Detection tests and false-alarm tests are written together.** Incidents 4 and
5 were both false alarms on honest input. A check that fires on correct data is
worse than no check, because it trains its users to ignore the output.

**Comparisons are computed over identical subsets.** From incident 6, applied
across all modules.

**Verification is by result, not by process.** During one session the machine
rebooted and the agent's terminal was lost mid-task. Recovery took two minutes,
because the acceptance criteria were "95 tests pass and the demo prints these two
numbers" — checkable directly from the working tree, with no dependence on the
agent's state.

**The agent's own quality observations are worth acting on.** Twice it reported
problems it had not been asked to look for: a sync that had reverted two
previously-fixed lint issues, with the correct diagnosis that the upstream source
needed the same fix or the regression would recur every round; and a newly merged
module that was skipped in every demo case despite passing its tests. Both were
right, and both fixes went upstream. It also declined to proceed when a source
directory did not exist, rather than guessing at the most recent similar path —
which would have silently reverted three documents and reintroduced a batch of
style regressions.

---

## What AI assistance was and was not good at

**Good at:** writing a module from a specification; producing thorough test
suites once the properties to test were named; mechanical work (dependency
synchronisation, lint fixes, formatting) with high reliability; noticing
inconsistencies across files that a human would miss.

**Not good at, unaided:** knowing when a plausible statistic is biased. Incidents
1, 3, 4 and 6 all involved a formula that was defensible on paper and wrong in
context. None was found by reasoning about the code; every one was found by
running it against a case whose answer was known in advance.

That asymmetry is the reason this project exists, and it is why the constraints
above are about verification rather than about prompting.
