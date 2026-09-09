"""A claim a document makes about the rest of the repository is checked.

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

What is not checked, having been counted
----------------------------------------
The inventory behind this file found roughly ninety prose cross-references
between the seven documents, and citations running the other way from module
docstrings back into them. None is checked. They are left deliberately, not
overlooked: a prose sentence asserting that another document "sets the bar" or
"records the reason" has no structure to parse, and a pattern fitted to today's
sentences is the failure this repository has diagnosed three times. Incident 21
carries the sizes so the next reader can tell a counted gap from an unexamined
one.

The conventions parsed
----------------------
Both are structures the documents already use, not markers invented here.
``PLAN.md`` headings are ``### Declined: ...`` and ``### Future issue: ...``.
``CLAUDE.md`` bullets are ``- Declined: <items>`` and ``- Deferred: <items>``,
where the items are a comma-separated list terminated by the first dash,
semicolon or full stop, after which prose resumes. ``MIGRATION.md`` marks
removed names by the column they sit in: its tables are headed with the version
each column describes, so a cell under ``0.1.2`` is a name this release removed
and a cell under ``0.2.0`` is one it must still answer to. An identifier in both
columns is unchanged, which is how ``audit`` appears on both sides.

That second convention did need a wording change: the bullet previously read
"Sortino, Omega and CVaR are declined (...); engine adapters are deferred",
which cannot be parsed without a pattern fitted to that sentence -- the failure
diagnosed twice already in this repository's history. One bullet per state, with
the state first, is the minimum that makes the claim machine-readable.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import re

import tomllib

import audit
from audit.coverage import AUDIT_REGISTRY

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


# ----------------------------------------------------------------------
# MIGRATION.md against the package it documents
# ----------------------------------------------------------------------
MIGRATION = "MIGRATION.md"
SPEC = "SPEC.md"

_BACKTICKED = re.compile(r"`([^`]+)`")
_DOTTED = re.compile(r"`([A-Z][A-Za-z0-9_]*)\.([a-z_][a-z0-9_]*)(?:\(\))?`")


def _package_source() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "src").rglob("*.py")
    )


def _declared_names() -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = project["project"]
    return {project["name"], *project.get("scripts", {})}


def _classes() -> dict[str, type]:
    found: dict[str, type] = {}
    for module in pkgutil.walk_packages(audit.__path__, "audit."):
        try:
            imported = importlib.import_module(module.name)
        except Exception:  # pragma: no cover - a module that cannot import is
            continue  # a failure other tests report far more clearly
        for name in dir(imported):
            value = getattr(imported, name)
            if isinstance(value, type):
                found.setdefault(name, value)
    return found


def _has_member(owner: type, attribute: str) -> bool:
    """Class attribute or dataclass field.

    A dataclass field declared without a default is not a class attribute, so
    ``hasattr`` alone misses exactly the fields these results are made of.
    """
    if hasattr(owner, attribute):
        return True
    if dataclasses.is_dataclass(owner):
        return attribute in {f.name for f in dataclasses.fields(owner)}
    return False


def _resolves(name: str, source: str, declared: set[str]) -> bool:
    """A name the package answers to: a literal it emits, or a name it declares.

    Serialisation keys, enum values and CLI flags all appear as string literals
    in the source; the distribution and console-script names come from
    ``pyproject.toml``. Between them these cover every kind of identifier
    ``MIGRATION.md`` promises a reader.
    """
    return name in declared or f'"{name}"' in source or f"'{name}'" in source


def _migration_columns() -> tuple[set[str], set[str]]:
    """(current, removed) identifiers, taken from the version-headed columns."""
    current: set[str] = set()
    removed: set[str] = set()
    header: list[str] | None = None
    for line in (ROOT / MIGRATION).read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        for column, cell in zip(header, cells, strict=False):
            for identifier in _BACKTICKED.findall(cell):
                if "0.2.0" in column or column == "Field":
                    current.add(identifier)
                elif "0.1.2" in column:
                    removed.add(identifier)
    return current, removed


def test_migration_names_an_interface_the_package_answers_to():
    """Every current name in MIGRATION.md resolves.

    This file is the entire interface documentation a downstream project gets,
    and a wrong name sends someone to a KeyError with no way to tell whether the
    interface or the note is at fault.
    """
    current, _ = _migration_columns()
    assert len(current) > 10, f"only {len(current)} current identifiers parsed"

    source, declared = _package_source(), _declared_names()
    missing = sorted(i for i in current if not _resolves(i, source, declared))
    assert not missing, (
        f"{MIGRATION} documents {missing} as the 0.2.0 interface, but the package "
        f"answers to no such name. Either the note is wrong or the rename is "
        f"incomplete, and a reader cannot tell which."
    )


def test_migration_removed_names_are_actually_removed():
    """The other direction: a name documented as gone must be gone.

    Identifiers listed in both version columns are unchanged -- the import
    package is the same on both sides -- and are exempt.
    """
    current, removed = _migration_columns()
    assert removed, f"no removed identifiers parsed from {MIGRATION}"

    source, declared = _package_source(), _declared_names()
    survivors = sorted(
        i for i in removed - current if _resolves(i, source, declared)
    )
    assert not survivors, (
        f"{MIGRATION} tells a reader {survivors} was removed in 0.2.0, but the "
        f"package still answers to it. A migration note that overstates what "
        f"broke costs its reader work for nothing."
    )


def test_migration_attribute_paths_exist_or_are_documented_as_removed():
    """``Class.attribute`` in prose, where the file makes its sharpest claims."""
    _, removed = _migration_columns()
    classes = _classes()
    text = (ROOT / MIGRATION).read_text(encoding="utf-8")

    pairs = set(_DOTTED.findall(text))
    assert len(pairs) > 4, f"only {len(pairs)} dotted paths parsed from {MIGRATION}"

    for class_name, attribute in sorted(pairs):
        assert class_name in classes, (
            f"{MIGRATION} refers to `{class_name}.{attribute}`, but the package "
            f"defines no class named {class_name!r}."
        )
        if _has_member(classes[class_name], attribute):
            continue
        assert attribute in removed, (
            f"{MIGRATION} refers to `{class_name}.{attribute}`, which does not "
            f"exist and is not listed in a 0.1.2 column as removed. A reader "
            f"cannot tell a name this release deleted from one it mistyped."
        )


# ----------------------------------------------------------------------
# SPEC.md against the registry it enumerates
# ----------------------------------------------------------------------
def test_spec_channel_table_matches_the_audit_registry():
    """SPEC promises one module per channel of inflation; the registry ships them.

    The table and the registry are two statements of the same count. Nothing
    connected them, so the documents could have claimed nine channels while the
    code shipped eight -- which is criterion 5's failure told from the outside.
    """
    lines = (ROOT / SPEC).read_text(encoding="utf-8").splitlines()
    rows = [
        line
        for line in lines
        if line.startswith("|")
        and "Module" not in line
        and set(line) - set("|- :")
    ]
    assert len(rows) == len(AUDIT_REGISTRY), (
        f"{SPEC} lists {len(rows)} channels of inflation and AUDIT_REGISTRY "
        f"ships {len(AUDIT_REGISTRY)} audits. One of them is wrong."
    )
