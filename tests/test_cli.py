"""Command-line boundary tests."""

from __future__ import annotations

import pytest

from audit.cli import build_parser


def test_cli_does_not_offer_an_in_sample_scope_option():
    parser = build_parser()
    assert "--scope" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["panel.csv", "--train-end", "2024-01-01", "--scope", "train"])
