"""Column date detection matches Onyx exactly.

Whether a spreadsheet column reads as dates changes the descriptor chunks built
for that sheet, which changes what retrieval can find. Onyx's behavior is the
behavior we measured, so it is pinned here rather than improved on by accident.

The loose case is the interesting one. `"10-20"` in a spreadsheet is more often
a range than a date, and reading it as October 20th is arguably wrong. It is
still what Onyx does, so it is what brain does. Changing it is a deliberate
decision that starts by editing this file.
"""

from __future__ import annotations

from datetime import date

import pytest

from brain.chunking.tabular.analysis import _try_date


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-01-02", date(2026, 1, 2)),
        ("2026-01-02T10:00:00", date(2026, 1, 2)),
        # Month-first, matching US spreadsheet exports.
        ("01/02/2026", date(2026, 1, 2)),
        ("2026/01/02", date(2026, 1, 2)),
    ],
)
def test_unambiguous_dates_parse(value: str, expected: date) -> None:
    assert _try_date(value) == expected


@pytest.mark.parametrize("value", ["not a date", "", "abc", "12", "---"])
def test_non_dates_are_rejected(value: str) -> None:
    assert _try_date(value) is None


def test_the_cheap_guard_rejects_before_parsing() -> None:
    """A value with no date separator never reaches the parser.

    This runs on every cell of every sheet, so the guard is what keeps a
    million-row import from spending all its time in date parsing.
    """
    assert _try_date("123456") is None
    assert _try_date("abc") is None


def test_loose_parsing_is_intentional() -> None:
    """dateutil's guess for a bare day-month, kept for Onyx parity.

    If this test fails because someone tightened the parser, that is a real
    product decision and not a bug: the column typing it feeds is part of what
    was benchmarked. Update the test and say why.
    """
    parsed = _try_date("10-20")
    assert parsed is not None
    assert (parsed.month, parsed.day) == (10, 20)
