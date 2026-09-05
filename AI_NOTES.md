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

## Incident 13 — a defect only visible in the delivered artefact

**What happened.** The demo writes its reports with `Path.write_text()`, which on
Windows uses the system encoding. The box-drawing characters in the decomposition
tree were written as cp1252 mojibake, and the files had been committed that way
for several rounds — publicly visible on GitHub the whole time.

**How it was caught.** Not by the test suite, which never reads those files, and
not by CI, which does not check their encoding. The coding agent noticed it while
comparing incoming and existing artefacts during a sync.

**The fix.** `encoding="utf-8"` on every `write_text` call in `demo.py` and
`cli.py`.

**Constraint added.** Tests establish that code runs; only inspecting the
delivered artefact establishes what a reader sees. This is the same lesson as
incident 11 from a different angle — there a module was invisible in the output,
here it was visible and wrong.

---

## Incident 14 — the check misread its own healthiest outcome

**What happened.** The execution-timing audit scores a signal as execution is
delayed, and flags a collapse between lag 0 and lag 1 as look-ahead. Run against
an honest forecast, it returned INCONCLUSIVE.

**Diagnosis.** The honest signal had an IC of −0.006 at lag 0 and +0.143 at lag 1,
and the verdict logic required a large lag-0 score before it would read the
profile at all. But a pure forecast *should* have no edge at lag 0: that return
had already happened when the signal was computed. The healthiest possible
profile was being treated as unreadable.

**The fix.** A third verdict class that names it — no edge on the contemporaneous
return, edge once execution moves forward — and explains why that is what
forecasting looks like.

**Constraint added.** When designing a check around a failure signature, enumerate
what success looks like too. The failure case was correct from the first attempt;
the success case had not been thought through, and would have flagged every
honest signal as unreadable.

---

## Incident 15 — the number was right and the sentence said the opposite

**What happened.** A demeaned IC of **−0.6880** was formatted as "small but
non-zero skill beyond the level." The numerical metric was correct; the report
translated a strongly negative association into positive-sounding evidence.
That is the same class of polished-but-wrong output the project exists to catch,
produced by the audit tool itself.

**Diagnosis.** The interpretation branch checked `abs(value) < 0.02` and then
`value < 0.10`. Every negative value outside the near-zero band therefore fell
into the small-positive branch. Existing tests exercised the metric and the
positive report path, but not the semantic input space. Code paths were covered;
sign, boundaries and undefined states were not. The positive happy path had been
mistaken for a complete behaviour contract.

**The fix.** The report now distinguishes material negative association,
near-zero association, small positive association, substantial positive
association and undefined input. Negative output asks the reader to check sign
conventions and says that treating a reversal as a signal requires separate
out-of-sample validation; it does not recommend an ex-post inversion.

**Constraint added.** Tests for interpretation functions are partitioned by
business semantics, not merely by executable branches: negative/zero/positive,
both sides of thresholds, finite/non-finite, defined/undefined and
PASS/FAIL/INCONCLUSIVE. They also assert the semantic invariant connecting the
number to the sentence. A correct statistic with an incorrect explanation is a
failed result.

---

## Incident 16 — a configuration value that never controlled the result

**What happened.** The CLI and `run_baseline_audit` accepted `scope=train` and
recorded it in the output config. Raw IC and several audits respected the value,
but demeaned IC always evaluated `test_slice()`. A report could therefore claim
to describe training data while silently mixing training- and test-period
numbers.

**Diagnosis.** Each metric's local tests were correct, but no orchestration test
asserted that every result in one report described the same evaluation period.
Because the command completed successfully, the mismatch was more dangerous
than a crash: it created plausible output with false provenance.

**The fix.** Credibility verdicts are now test-period only. The CLI and runner no
longer expose train/all options, every audit receiving a scope gets `test`, and
the report states the actual evaluation dates and `evaluation_scope=test`.
Lower-level metric functions retain explicit scopes for diagnostic use, outside
the combined verdict.

**Constraint added.** A setting that is recorded but does not control every
claimed consumer is worse than no setting. Orchestration tests must verify
cross-module contracts — scope, dates and provenance — rather than assuming
locally correct functions compose into a truthful report.

---

## Incident 17 — source tests passed while the wheel could not run PIT

**What happened.** All source tests and the demo passed, and a wheel could be
built, but the wheel did not contain the DuckDB schema. From an installed wheel,
constructing `BitemporalStore()` raised `FileNotFoundError` because runtime code
looked for `sql/duckdb/001_schema.sql` at the repository root.

**Diagnosis.** Editable installs made the repository file visible, so every
development check exercised a shape users would not receive. Building an
artefact was treated as sufficient without installing and executing that exact
artefact from outside the source tree.

**The fix.** The schema moved under `src/audit/sql/duckdb`, is declared as
package data and is loaded with `importlib.resources`. CI now builds the wheel,
installs it into a clean virtual environment, changes out of the repository and
instantiates the PIT store.

**Constraint added.** Release-shape behaviour is a separate test surface. A
package is not verified until the built artefact is inspected, installed and
used from a location where repository files cannot rescue it.

---

## Incident 18 — an omitted audit looked like an audit with nothing to report

**What happened.** The runner silently omitted modules whose inputs were absent.
A prediction-only panel therefore produced a polished report with no protocol,
execution, point-in-time or selection section, but no statement that those
questions had never been examined. A reader could not reliably distinguish that
partial report from one whose full audit surface had run.

**Diagnosis.** Incident 2 established that undefined cannot masquerade as zero:
"not measurable" and "measured, found to be nil" are different assertions. The
metric layer preserved that distinction, but the orchestration layer violated
the same principle at chapter scale. ``format_pit`` existed and
``render_report`` accepted a PIT result, satisfying the old acceptance criterion
literally, while ``run_baseline_audit`` never supplied the result and never
declared its absence. Tests checked present sections, not the complement of what
the report failed to contain.

**The fix.** A closed registry now names all nine shipped audit channels. Every
text and JSON report carries a coverage manifest. Execution state is separate
from verdict: ``SKIPPED`` means the computation did not run, while
``INCONCLUSIVE`` means it ran but the evidence could not support a conclusion.
Skip reasons use a three-value enum rather than free text, and PIT and selection
are first-class attachable result slots. The demo writes an index across its
independent known-truth cases so it proves every registered audit is visible
without pretending those cases share one scope.

**Constraint added.** A report is complete only when it describes both the
checks it ran and the checks it did not run. The registry and report key sets
must be identical. Missing evidence, explicit disabling and externally-required
evidence remain machine-distinguishable, and unexpected implementation errors
must propagate rather than being converted into a skipped section or ``NaN``.

---

## Incident 19 — the numbers were right and both labels were wrong

**What happened.** Two quantities in the PnL layer were correctly computed and
incorrectly named.

The first was "maximum drawdown". Positions are dollar-neutral and unit-gross,
and the layer accumulates an additive score series; nothing is invested and
nothing compounds. Dividing that series' peak-to-trough decline by its running
peak produced a figure presented as a percentage, and on the demo's level-only
panel it read **186.8%**. A drawdown of invested capital cannot exceed 100%, so
the number was either impossible or not what its label said. It was the second.

The second was "Observed Sharpe" in the selection report, printed as **+0.0741**
while the README narrative for the same case said "annualised Sharpe of 1.18".
Both were correct: the Deflated Sharpe must be computed on per-period returns,
and 0.0741 x sqrt(252) = 1.18. The label named neither scale, so the two numbers
differed by a factor of 15.9 under one heading.

**Why it is the harder case.** Incident 15 was a correct statistic with an
incorrect sentence, and this is the same class one level down: a correct
statistic with an incorrect *name*. It is harder to notice, because a wrong
number can be checked against a known-truth case and a wrong name cannot. Every
test passed. The arithmetic was right in both instances, and would have stayed
right through any amount of numerical verification. What failed was the claim
the report made about what it had computed — which is precisely the failure this
project exists to detect in other people's backtests.

**The fix.** The drawdown is renamed `additive_peak_to_trough` and reported in
the units of the score series, with the report stating what it is and what it is
not. The division by a running peak is gone, along with the floor of 1.0 that
existed only to stop the ratio exploding — a guard against a symptom of the
wrong formulation.

For the Sharpe, `periods_per_year` lost its default of 252. The panel contract
is four columns and carries no observation frequency, so a default asserted
daily data about panels that might be weekly. Without a frequency the report
prints the per-period figure and states `NOT AVAILABLE -- observation frequency
not supplied`; with one it prints both and names the factor. The docstring
records that sqrt-T scaling assumes i.i.d. returns and that the error's
direction follows the sign of the autocorrelation — positive inflates, negative
deflates (Lo 2002). It is not an upper bound; that would hold for one sign only.

**The part that took three rounds: the test named a place, not a property.**
Renaming the data showed how many places rendered it. The lag table had two
renderings and the comparison table a third, and each round of fixing corrected
only the one that had been noticed — the report formatter, then the demo's copy
of the lag table still headed "ann. Sharpe", then the demo's comparison table
headed "raw (ann.)". The test written after the first round asserted that "ann."
appeared nowhere in the execution report. It passed while a second surface was
wrong, because the assertion had been scoped to where the defect was last seen.

A fourth shape survived even that. ``execution_report.json`` carried
``"sharpe": 11.85`` — correct, per period, and silent about which scale it was
on. It passed a check that only constrained lines *claiming* annualisation,
since a bare name claims nothing. The assertion now covers both shapes across
every surface a demo run produces, stdout and all seventeen files: a claim of
annualisation must name its factor, and a reported Sharpe must declare its
scale. Both halves are mutation-tested, because a check of this kind that cannot
fail is the thing it is supposed to prevent.

A fifth shape escaped even the widened check. The narrative sentence beside the
selection report read "1.18 annualised at 252" while the tables it sat next to
had been corrected to say sqrt(252). It passed because the assertion required a
claim to name *a* factor, and this one named a number — the period count, as
though it were the multiplier. Checking that a scale was declared is not
checking that the declared scale is the one applied.

The fix is not a stricter pattern. Every renderer now takes its factor string
from ``annualisation_label``, which lives beside ``annualise_sharpe``, so the
words and the operation cannot drift apart without being edited together; the
assertion requires that shared label rather than any number, and a separate test
checks the label against the arithmetic it describes. The same de-duplication
fixed the lag table two shapes earlier. One test was itself pinning the label as
a literal string, and was rewritten to derive it.

**A limit of the check, recorded rather than papered over.** A renderer holds a
string and a value, not the computation that produced them. It cannot verify
that a prose sentence describes the operation behind the number beside it: a
sentence quoting the wrong figure, or describing in its own words a factor it
did not apply, remains unreachable from a rendering test. Sharing the string is
what closes that gap, as far as it can be closed. A pattern broad enough to look
like it closed the rest would only give false confidence.

**Constraint added.** Scope an assertion to the property, not to the place the
property was last violated. A test named after a location passes as soon as the
defect moves, and a rename moves it. Where one quantity is rendered more than
once, the renderings are shared rather than checked for agreement.

**Audit item for sibling repositories.** This is a reusable failure class, so by
the rule in ``CLAUDE.md`` it creates an item for ``factor-zoo-audit`` rather than
staying here: any assertion phrased against a named file or report, where the
property it defends applies to every rendered surface. That repository is not
edited from here; the item is recorded so it is not lost.

---

**A decision recorded rather than defaulted.** Removing the hardcoded factor
raised a question the parameter did not settle: whether a synthetic panel built
to be impossible should print an annualised Sharpe of 147 at all. It should, and
no threshold suppresses it. Incident 12 is that number, and the resolution there
was that it is correct and belongs in the output; a magnitude clamp would be the
same adjustment the incident rejected. "Implausibly large" is also not estimable
without bias from the data at hand, which by incident 4's constraint makes it
context and never a gate — and the gate would apply to real panels, where a
timestamp error producing an absurd Sharpe is a finding rather than an
embarrassment. It would additionally give the annualised field a fourth state,
after the three this incident just separated.

What the figure did need was not suppression but an unconditional statement of
what it is. The demo's caveat covered the first of its two performance blocks
and not the second, and the block that carried one explained only the
peak-to-trough row, leaving the Sharpe figures under a heading that reads like a
conventional performance statistic. The caveat now belongs to the block rather
than to the surrounding prose, so both carry it, and it is printed regardless of
magnitude: a Sharpe of 147 announces itself, while an ordinary-looking figure
from the same costless, capacity-free scoring device is the one a reader would
mistake for an achievable return.

**Constraint added.** A reported quantity's name is part of its correctness. A
number whose label implies a different construction than the one that produced
it is a wrong result, even when the arithmetic is right, and no known-truth test
will catch it — the check is to state the units and the scale next to the
figure, and to ask what a reader would assume the name meant. Where a scale
depends on an input the contract does not carry, the absence of that input is
reported as not applicable, distinct from zero and from a failed computation.

---

## Incident 20 — the right formula fed the wrong quantity

**What happened.** `deflated_sharpe` computed the Deflated Sharpe Ratio's
benchmark from the wrong variance. Bailey and Lopez de Prado define `SR*` using
`V[{SR_n}]`, the variance of the Sharpe estimates across the N trials — a
property of the *search*. The module passed `sharpe_estimator_variance(...)`
instead: the sampling variance of the winning candidate's own Sharpe estimator.

Every step around it was right. The extreme-value approximation was implemented
correctly, the skew and kurtosis correction was the published one, and the
z-statistic that follows the benchmark uses the estimator variance, which is
what BLdP specify *there*. One input to one term was a different quantity
wearing a compatible name, and the module reported a verdict as though the
formal statistic had been computed.

**Why it survived, and what that says about the method.** Nothing about it
looks wrong. Both quantities are "the variance of a Sharpe", both are positive,
both scale the benchmark sensibly, and the substitution is exactly right when
trials are independent and identically distributed — which is the case a test
built from independent noise candidates constructs. The repository's own demo
grid is 42 independent draws, so the formal and substituted benchmarks there are
0.0921 against 0.0806. A known-truth test built on that fixture confirms the
module works, because on that fixture it does.

That is a blind spot in the known-truth method itself, not a fact about this
module. The method's premise is that a synthetic panel with an explicit
generative decomposition tells you the answer in advance. It does — for the
panel you generated. A substituted input that coincides with the correct one
under the fixture's own assumptions is invisible to every assertion built on
that fixture, however many of them there are, and the fixture's assumptions are
usually the convenient ones: independent draws, identical distributions, no
correlation between candidates. The generator and the bug agreed, so the test
could only confirm the agreement.

What catches this is not a better assertion but a second fixture chosen to
violate the coincidence: here, a correlated grid and a heterogeneous candidate
set, where the two variances differ by 0.003x and 29.5x. So the constraint the
earlier incidents produced — verify every metric against a case whose answer is
known — needs a companion: **construct the known-truth case so that the
quantities you are substituting between are known to differ in it.** A fixture
built from independent draws cannot distinguish a per-trial variance from a
cross-trial one, and no amount of testing on it will.

**How far apart they get.** Measured, rather than asserted as a caveat:

| candidate set | `V[{SR_n}]` | winner's `Var[SR]` | ratio |
|---|---|---|---|
| correlated parameter grid | 0.000004 | 0.001325 | 0.003x |
| heterogeneous candidates | 0.040525 | 0.001372 | 29.5x |

Since `SR*` scales with the square root, a correlated grid — which is what a
real parameter scan produces — makes the substituted benchmark far too high, and
a heterogeneous candidate set makes it far too low. The direction of the error
is not determined a priori and cannot be recovered from the winning return
stream, which is the only thing the module receives. So it is not conservative,
and claiming it was would have been the more comfortable and less true position.

**The fix, and what it deliberately is not.** The approximation is kept, because
the correct variance requires the Sharpe of every candidate examined and this
repository has deferred that ledger. What is removed is the claim. With
`trial_sharpes` the result is the formal statistic; without it the result is
marked `method=proxy`, `trial_variance_source=winner_estimator`, the headline
figure is renamed `approximate_selection_adjusted_sharpe_per_period`, and the verdict is
capped at INCONCLUSIVE. `PLAN.md` now records the ledger as the precondition for
computing `SR*` as defined rather than as a governance nicety.

FAIL stays reachable, asymmetrically and on purpose. A substituted benchmark can
be too high as easily as too low, so a FAIL is not formally warranted either; it
is kept as a caution and says so in its own text, because the failure this
framework exists to prevent is endorsing a result rather than doubting one.

One trial is exempt from the cap. `expected_max_sharpe` returns 0.0 for a single
trial whatever variance it is handed, so there the two benchmarks are identical
rather than close, and capping would fail an honest single test for a
substitution that could not have touched it — incidents 4 and 5 again.

**Constraint added.** Where a known-truth fixture could make a substituted
input coincide with the correct one, the fixture is not evidence about the
substitution; a second case must be constructed in which the two are known to
differ. And a module must not report a named statistic when one of its
inputs is a stand-in for the quantity the definition names. Mechanically correct
is not the same as correct: the check is to state, for each input, which defined
quantity it is, and where they differ to name the substitution, the condition
under which the two coincide, and the direction of the error — or to record that
the direction is undetermined, which is a stronger statement than a vague
caveat and requires measuring it. A statistic whose definition has been quietly
relaxed cannot carry a strong verdict.

**Audit item for sibling repositories.** A reusable class, so by the rule in
`CLAUDE.md` it raises an item for `factor-zoo-audit` rather than staying here:
any published statistic implemented from a paper where an input is substituted
for the one the definition names. That repository is not edited from here.

---

## Workflow constraints

The rules that emerged, applied to every subsequent session:

**Test assertions must not be weakened merely to make an implementation pass.**
When an intentionally changed public contract makes an old assertion incorrect,
the test may change only with an explanation of the old behaviour, target
behaviour and migration impact. Tests protect the correct contract, not a known
bug. This distinction matters for the removal of the false scope option and the
rename from purged to embargoed walk-forward.

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
