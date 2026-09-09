"""A claim one document makes about another is checked like any other claim.

``CLAUDE.md`` asserted for six commits and a tagged release that four items were
declined and that the decline was recorded in ``PLAN.md``. It was not: engine
adapters were still filed as a future issue pending demand, and Sortino, Omega
and CVaR appeared nowhere. Nothing noticed, because every check in this
repository inspects code, reports or artefacts, and this was one document lying
about another.

The comfortable conclusion was that nothing could have caught it. That was
wrong, and incident 21 records why: a cross-reference between documents drifts
exactly like two renderings of a number, and is checkable the same way.

What is checked here
--------------------
* Every item ``CLAUDE.md`` declares **declined** appears in ``PLAN.md`` under a
  ``### Declined:`` heading, with a reason under it.
* Every item ``CLAUDE.md`` declares **deferred** appears under a
  ``### Future issue:`` heading, with a reason under it.
* No item is declared both.
* Every ``### Declined:`` section in ``PLAN.md`` is accounted for by some item
  ``CLAUDE.md`` declares declined.

The asymmetry in that last pair is deliberate. Declined is checked in both
directions, because a decline recorded in only one of the two files is exactly
the defect this test exists for. Deferred is checked one way only: ``PLAN.md``
carries seven future issues and ``CLAUDE.md`` names one, because it names only
what a brief put out of scope. Requiring the reverse would force every future
issue into the working rules, which is not what that file is for.

The conventions parsed
----------------------
Both are structures the documents already use, not markers invented here.
``PLAN.md`` headings are ``### Declined: ...`` and ``### Future issue: ...``.
``CLAUDE.md`` bullets are ``- Declined: <items>`` and ``- Deferred: <items>``,
where the items are a comma-separated list terminated by the first dash,
semicolon or full stop, after which prose resumes.

That second convention did need a wording change: the bullet previously read
"Sortino, Omega and CVaR are declined (...); engine adapters are deferred",
which cannot be parsed without a pattern fitted to that sentence -- the failure
diagnosed twice already in this repository's history. One bullet per state, with
the state first, is the minimum that makes the claim machine-readable.
"""

from __future__ import annotations

import re

from .test_project_metadata import ROOT

CLAUDE = "CLAUDE.md"
PLAN = "PLAN.md"

# `- Declined: a, b, c — prose` / `- Deferred: x — prose`
_CLAIM = re.compile(r"^\s*-\s+(Declined|Deferred):\s*(.+)$")
# Items end where prose begins.
_ITEMS_END = re.compile(r"[—.;:]|--")

_PLAN_HEADING = re.compile(r"^###\s+(Declined|Future issue):\s*(.+)$")

# A heading with nothing under it records a decision without its reason.
MINIMUM_REASON = 120


def _claimed() -> list[tuple[str, str, int]]:
    """(state, item, line number) for every item CLAUDE.md places in a state."""
    claims = []
    lines = (ROOT / CLAUDE).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        match = _CLAIM.match(line)
        if match is None:
            continue
        state = "declined" if match.group(1) == "Declined" else "deferred"
        listed = _ITEMS_END.split(match.group(2), maxsplit=1)[0]
        for item in listed.split(","):
            if item.strip():
                claims.append((state, item.strip(), number))
    return claims


def _plan_sections() -> list[tuple[str, str, str]]:
    """(state, heading text, body) for every roadmap section in PLAN.md."""
    lines = (ROOT / PLAN).read_text(encoding="utf-8").splitlines()
    sections = []
    index = 0
    while index < len(lines):
        match = _PLAN_HEADING.match(lines[index])
        if match is None:
            index += 1
            continue
        state = "declined" if match.group(1) == "Declined" else "deferred"
        heading = match.group(2)
        body, index = [], index + 1
        while index < len(lines) and not lines[index].startswith(("###", "## ")):
            body.append(lines[index])
            index += 1
        sections.append((state, heading, "\n".join(body)))
    return sections


def test_the_parse_finds_claims_on_both_sides_rather_than_passing_vacuously():
    """A parser that matched nothing would pass every assertion below silently."""
    claims = _claimed()
    sections = _plan_sections()

    assert claims, f"no `- Declined:`/`- Deferred:` bullets found in {CLAUDE}"
    assert {state for state, _, _ in claims} == {"declined", "deferred"}
    assert [s for s in sections if s[0] == "declined"], f"no declines found in {PLAN}"
    assert [s for s in sections if s[0] == "deferred"], f"no future issues in {PLAN}"


def test_no_item_is_declared_both_declined_and_deferred():
    """The two states are exclusive; a document saying both says nothing."""
    declined = {item.lower() for state, item, _ in _claimed() if state == "declined"}
    deferred = {item.lower() for state, item, _ in _claimed() if state == "deferred"}
    assert not declined & deferred, sorted(declined & deferred)


def test_every_item_claude_places_in_a_state_is_recorded_in_plan():
    """The direction that failed: CLAUDE.md asserted a record PLAN.md did not hold."""
    sections = _plan_sections()

    for state, item, line in _claimed():
        matching = [
            (heading, body)
            for section_state, heading, body in sections
            if section_state == state and item.lower() in heading.lower()
        ]
        expected = "### Declined:" if state == "declined" else "### Future issue:"
        assert matching, (
            f"{CLAUDE}:{line} declares {item!r} {state}, but {PLAN} has no "
            f"{expected} section naming it. The decision is asserted in one "
            f"document and recorded in neither -- record it in {PLAN}, or stop "
            f"claiming it in {CLAUDE}."
        )
        for heading, body in matching:
            assert len(body.strip()) >= MINIMUM_REASON, (
                f"{PLAN} section {heading!r} records {item!r} as {state} without "
                f"a reason. {CLAUDE}:{line} says the reason is there; a heading "
                f"alone is a decision nobody can re-examine."
            )


def test_every_decline_in_plan_is_accounted_for_by_claude():
    """The reverse direction, for declines only.

    A decline appearing in one file and not the other is the defect either way
    round. Future issues are not checked in reverse: PLAN.md carries several that
    CLAUDE.md has no reason to name.
    """
    declined_items = [item for state, item, _ in _claimed() if state == "declined"]

    for state, heading, _ in _plan_sections():
        if state != "declined":
            continue
        assert any(item.lower() in heading.lower() for item in declined_items), (
            f"{PLAN} declines {heading!r}, which {CLAUDE} does not list as "
            f"declined. Either add it to the `- Declined:` bullet or explain in "
            f"{PLAN} why the working rules need not carry it."
        )
