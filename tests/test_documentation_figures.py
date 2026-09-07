"""Figures quoted in the documentation must come from the shipped output.

Four of the five defects found while hardening this release were regressions
introduced by the fixes themselves, and every one was visible in delivered
output the moment it existed and invisible everywhere else. The last of them --
a README table still showing the execution audit's pre-rename annualised
figures, +188.13 against the +11.8511 the audit now reports -- escaped the
surface-scanning test written for exactly that shape, because that test's corpus
is the demo's own surfaces and documentation is not one of them.

This closes that gap. Every untagged fenced block in the documentation is
treated as a transcript of shipped output, and every line in it that carries a
number must appear verbatim in output regenerated from the current tree.

Two scoping choices, both deliberate:

*Untagged fences only.* A fence with a language tag declares code, and code
examples legitimately contain numbers that were never output -- a
``periods_per_year=252`` argument, a date in a command line. A bare fence in
these documents is quoting the tool.

*Regenerated, not stored.* The corpus is produced by running the demo now, not
read from ``examples/outputs``. Asserting documentation against committed
artefacts would let one stale file vouch for another, and asserting anything
against stored output from a previous run is what SPEC criterion 1 forbids. The
claim here is narrower and safe: the documentation quotes what this tree
currently produces.
"""

from __future__ import annotations

import contextlib
import io
import re

import pytest

from audit import demo

from .test_project_metadata import ROOT

DOCS = (
    "README.md",
    "PLAN.md",
    "SPEC.md",
    "AI_NOTES.md",
    "CROSS_REPO_AUDIT.md",
    "MIGRATION.md",
)

# The marker already used for the effective-sample-size block in README: the
# prose introducing the fence says the figures are illustrative. One convention,
# read by machine as well as by a person, rather than a second one invented here.
ILLUSTRATIVE = "illustrative"
ILLUSTRATIVE_LOOKBACK = 6

_DIGIT = re.compile(r"\d")


@pytest.fixture(scope="module")
def shipped_output(tmp_path_factory) -> str:
    """Everything one demo run puts in front of a reader: stdout and every file."""
    outdir = tmp_path_factory.mktemp("doc_figures")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        demo.main(["--outdir", str(outdir)])

    corpus = [buffer.getvalue()]
    for path in sorted(outdir.rglob("*")):
        if path.is_file():
            corpus.append(path.read_text(encoding="utf-8"))
    return "\n".join(corpus)


def _quoted_blocks(text: str) -> list[tuple[int, list[str]]]:
    """Untagged fenced blocks that are not marked illustrative, with line numbers."""
    lines = text.splitlines()
    blocks: list[tuple[int, list[str]]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("```"):
            index += 1
            continue

        tagged = line[3:].strip() != ""
        preamble = lines[max(0, index - ILLUSTRATIVE_LOOKBACK) : index]
        marked = any(ILLUSTRATIVE in p.lower() for p in preamble)

        body: list[tuple[int, str]] = []
        index += 1
        while index < len(lines) and not lines[index].startswith("```"):
            body.append((index + 1, lines[index]))
            index += 1
        index += 1

        if tagged or marked:
            continue
        for number, content in body:
            if _DIGIT.search(content):
                blocks.append((number, [content]))
    return blocks


def _common_prefix_length(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _what_the_output_says(doc_line: str, corpus: str) -> str:
    """The output line the quoted one was probably copied from, so the failure
    names the real value rather than only the stale one.

    Longest shared prefix rather than a parsed label: report lines are
    label-then-number, so the stale and current versions of the same row agree
    for as far as the label and any unchanged columns run, and diverge exactly
    at the figure that moved. A label parser has to guess where the label ends,
    and guessed wrong on rows like ``lag 0`` whose label contains a digit.
    """
    best, score = None, 0
    for line in corpus.splitlines():
        shared = _common_prefix_length(doc_line.rstrip(), line.rstrip())
        if shared > score:
            best, score = line, shared
    if best is None or score < 12:
        return "no comparable line found in the regenerated output"
    return repr(best.rstrip())


@pytest.mark.parametrize("doc", DOCS)
def test_quoted_figures_match_the_regenerated_output(doc, shipped_output):
    """A number quoted in the docs must be one this tree actually produces."""
    text = (ROOT / doc).read_text(encoding="utf-8")

    for line_number, (content,) in ((n, b) for n, b in _quoted_blocks(text)):
        if content.rstrip() in shipped_output or content.strip() in shipped_output:
            continue
        raise AssertionError(
            f"{doc}:{line_number} quotes a figure the current tree does not "
            f"produce.\n"
            f"  documentation says: {content.rstrip()!r}\n"
            f"  output says:        {_what_the_output_says(content, shipped_output)}\n"
            f"  Regenerate examples/outputs and update the prose, or mark the "
            f"block {ILLUSTRATIVE!r} in the sentence introducing it if the "
            f"figures are not a transcript."
        )


def test_the_check_reads_some_figures_rather_than_passing_vacuously():
    """A block scanner that matched nothing would pass every document silently."""
    quoted = sum(
        len(_quoted_blocks((ROOT / doc).read_text(encoding="utf-8"))) for doc in DOCS
    )
    assert quoted > 30, f"only {quoted} quoted figure lines found across {len(DOCS)} docs"


def test_the_illustrative_marker_exempts_a_block_and_is_actually_in_use():
    """The exemption must work, and must not be a facility nobody uses.

    README's effective-sample-size block shows strong positive autocorrelation to
    make the point readable; the demo panels are close to independent and report
    an SE inflation near 1.0. It is an illustration, not a transcript, and it is
    the reason this convention exists rather than a hypothetical.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert ILLUSTRATIVE in readme

    quoted = {content for _, (content,) in _quoted_blocks(readme)}
    # 4.21 appears only in the illustrative block. "t-stat (naive)" would not do:
    # it also appears in README's genuine full-report excerpt, so asserting on it
    # would test a string with two homes rather than the exemption.
    assert not any("4.21" in line for line in quoted), (
        "the illustrative block is being scanned; the marker is not taking effect"
    )
    assert any("t-stat (naive)" in line for line in quoted), (
        "the genuine transcript quoting a t-statistic is no longer being scanned"
    )
