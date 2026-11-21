## Status

Implemented: panel contract, baseline decomposition, demeaned IC,
partial-correlation increment with the errors-in-variables correction,
Newey-West inference, alignment audit (shuffle / forward shift / backward
diagnostic), leaky-vs-clean example pipelines with five switchable defects,
bitemporal store with as-of reconstruction, point-in-time vs restated
comparison, survivorship audit deriving attrition from panel presence (not from
a listing calendar -- see its module docstring for the blind spot that leaves),
within/between group
decomposition, validation-protocol comparison (random vs walk-forward vs embargoed),
effective sample size, a thin PnL layer reporting raw and demeaned Sharpe, text
execution-timing decay, Deflated Sharpe and FDR screening, text and JSON
reports, offline demo, and a test suite built around known-truth controls.

Possible extensions are listed with their rationale and cost in
[PLAN.md](PLAN.md#roadmap). None of them blocks the framework being usable: the
nine channels of inflation it enumerates are each implemented and tested. Every
report now carries an audit coverage manifest: a channel that did not run remains
visible with the missing evidence or configuration decision that prevented it.
