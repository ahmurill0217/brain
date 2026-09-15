# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from tests/unit/onyx/document_index/opensearch/test_time_cutoff_filter.py.
"""How time ranges become range clauses, and when undated documents survive.

`_get_search_filters` is a pure builder, so none of this needs OpenSearch. The
asymmetry between the two date fields is the interesting part: created_at always
keeps undated documents, last_updated only does so for an old open-ended window.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from brain.config import BrainSettings
from brain.index.queries import DocumentQuery
from brain.index.schema import CREATED_AT_FIELD_NAME, LAST_UPDATED_FIELD_NAME
from brain.models.search import TimeRange


def _build_filters(
    settings: BrainSettings,
    created_at_range: TimeRange | None = None,
    updated_at_range: TimeRange | None = None,
) -> list[dict[str, Any]]:
    """Filters with only the time ranges set. The ACL and hidden clauses are
    suppressed so whatever comes back is purely the time filtering."""
    return DocumentQuery._get_search_filters(
        settings=settings,
        include_hidden=True,
        access_control_list=None,
        source_types=[],
        tags=[],
        document_sets=[],
        attached_document_ids=None,
        created_at_range=created_at_range,
        updated_at_range=updated_at_range,
        min_chunk_index=None,
        max_chunk_index=None,
    )


def _clause_for_field(
    filter_clauses: list[dict[str, Any]], field_name: str
) -> dict[str, Any] | None:
    """Find a date clause: a bool/should whose first element ranges over the
    given field."""
    for clause in filter_clauses:
        should = clause.get("bool", {}).get("should")
        if should and "range" in should[0] and field_name in should[0]["range"]:
            return clause
    return None


def _range_bounds(clause: dict[str, Any], field_name: str) -> dict[str, int]:
    return clause["bool"]["should"][0]["range"][field_name]


def _includes_undated(clause: dict[str, Any]) -> bool:
    return any(
        isinstance(sub.get("bool"), dict) and "must_not" in sub["bool"]
        for sub in clause["bool"]["should"]
    )


def test_no_time_filter_produces_no_clause(settings: BrainSettings) -> None:
    assert _clause_for_field(_build_filters(settings), LAST_UPDATED_FIELD_NAME) is None
    assert _clause_for_field(_build_filters(settings), CREATED_AT_FIELD_NAME) is None


def test_created_in_window_bounds_created_at_on_both_ends(settings: BrainSettings) -> None:
    """A created-between intent is one created_at range with both bounds, and
    undated documents are kept: a document that exists was created sometime."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 3, 31, tzinfo=UTC)
    clauses = _build_filters(settings, created_at_range=TimeRange(start=start, end=end))

    created = _clause_for_field(clauses, CREATED_AT_FIELD_NAME)
    assert created is not None
    assert _range_bounds(created, CREATED_AT_FIELD_NAME) == {
        "gte": int(start.timestamp()),
        "lte": int(end.timestamp()),
    }
    assert _includes_undated(created)
    # A created-intent query must not touch last_updated.
    assert _clause_for_field(clauses, LAST_UPDATED_FIELD_NAME) is None


def test_updated_in_past_window_uses_overlap_not_strict_last_updated(
    settings: BrainSettings,
) -> None:
    """An updated-in-[S, E] intent is an overlap (last_updated >= S AND
    created_at <= E), not a strict last_updated window. Only the latest edit is
    stored, so a strict upper bound would drop a document edited again after E
    whose earlier in-window edit is the one the caller meant."""
    start = datetime.now(tz=UTC) - timedelta(days=7 * 30)
    end = datetime.now(tz=UTC) - timedelta(days=4 * 30)
    clauses = _build_filters(
        settings,
        created_at_range=TimeRange(end=end),
        updated_at_range=TimeRange(start=start),
    )

    updated = _clause_for_field(clauses, LAST_UPDATED_FIELD_NAME)
    assert updated is not None
    assert _range_bounds(updated, LAST_UPDATED_FIELD_NAME) == {"gte": int(start.timestamp())}
    assert "lte" not in _range_bounds(updated, LAST_UPDATED_FIELD_NAME)

    created = _clause_for_field(clauses, CREATED_AT_FIELD_NAME)
    assert created is not None
    assert _range_bounds(created, CREATED_AT_FIELD_NAME) == {"lte": int(end.timestamp())}

    # A document created 8 months ago and last edited 2 months ago satisfies
    # both clauses, which is the point of splitting the window across fields.
    created_8mo = int((datetime.now(tz=UTC) - timedelta(days=8 * 30)).timestamp())
    updated_2mo = int((datetime.now(tz=UTC) - timedelta(days=2 * 30)).timestamp())
    assert updated_2mo >= _range_bounds(updated, LAST_UPDATED_FIELD_NAME)["gte"]
    assert created_8mo <= _range_bounds(created, CREATED_AT_FIELD_NAME)["lte"]


def test_updated_since_recent_excludes_undated(settings: BrainSettings) -> None:
    """A recent open-ended lower bound drops undated documents, or "changed
    since yesterday" would return the whole undated corpus."""
    start = datetime.now(tz=UTC) - timedelta(days=1)
    clauses = _build_filters(settings, updated_at_range=TimeRange(start=start))

    updated = _clause_for_field(clauses, LAST_UPDATED_FIELD_NAME)
    assert updated is not None
    assert _range_bounds(updated, LAST_UPDATED_FIELD_NAME) == {"gte": int(start.timestamp())}
    assert not _includes_undated(updated)


def test_updated_since_old_open_bound_includes_undated(settings: BrainSettings) -> None:
    """Once the lower bound is older than the assumed document age, an undated
    document is assumed to fall inside it."""
    start = datetime.now(tz=UTC) - timedelta(days=settings.assumed_document_age_days + 10)
    clauses = _build_filters(settings, updated_at_range=TimeRange(start=start))

    updated = _clause_for_field(clauses, LAST_UPDATED_FIELD_NAME)
    assert updated is not None
    assert _includes_undated(updated)


def test_assumed_document_age_days_moves_the_threshold() -> None:
    """The cutoff is the setting, not a hard-coded 90 days."""
    start = datetime.now(tz=UTC) - timedelta(days=30)
    strict = BrainSettings(_env_file=None, assumed_document_age_days=90)
    lenient = BrainSettings(_env_file=None, assumed_document_age_days=1)

    strict_clause = _clause_for_field(
        _build_filters(strict, updated_at_range=TimeRange(start=start)),
        LAST_UPDATED_FIELD_NAME,
    )
    lenient_clause = _clause_for_field(
        _build_filters(lenient, updated_at_range=TimeRange(start=start)),
        LAST_UPDATED_FIELD_NAME,
    )
    assert strict_clause is not None and not _includes_undated(strict_clause)
    assert lenient_clause is not None and _includes_undated(lenient_clause)


def test_active_in_window_splits_across_fields(settings: BrainSettings) -> None:
    """An active-in-[S, E] intent becomes last_updated >= S AND created_at <= E,
    approximating activity from a document's [created_at, last_updated] span."""
    start = datetime(2025, 7, 3, tzinfo=UTC)
    end = datetime(2025, 7, 4, tzinfo=UTC)
    clauses = _build_filters(
        settings,
        created_at_range=TimeRange(end=end),
        updated_at_range=TimeRange(start=start),
    )

    updated = _clause_for_field(clauses, LAST_UPDATED_FIELD_NAME)
    assert updated is not None
    assert _range_bounds(updated, LAST_UPDATED_FIELD_NAME) == {"gte": int(start.timestamp())}

    created = _clause_for_field(clauses, CREATED_AT_FIELD_NAME)
    assert created is not None
    assert _range_bounds(created, CREATED_AT_FIELD_NAME) == {"lte": int(end.timestamp())}
    assert _includes_undated(created)


def test_naive_bounds_are_treated_as_utc(settings: BrainSettings) -> None:
    """Naive bounds coerce to UTC when the range is built, so the comparison
    against `now` inside the undated check cannot raise TypeError."""
    time_range = TimeRange(start=datetime(2024, 1, 1), end=datetime(2024, 3, 31))
    assert time_range.start is not None and time_range.start.tzinfo == UTC
    assert time_range.end is not None and time_range.end.tzinfo == UTC

    clauses = _build_filters(settings, created_at_range=time_range)
    created = _clause_for_field(clauses, CREATED_AT_FIELD_NAME)
    assert created is not None
    assert _range_bounds(created, CREATED_AT_FIELD_NAME)["gte"] == int(
        datetime(2024, 1, 1, tzinfo=UTC).timestamp()
    )


def test_empty_range_is_skipped(settings: BrainSettings) -> None:
    """A range with neither bound contributes nothing, rather than an empty
    range clause that would match nothing at all."""
    clauses = _build_filters(settings, created_at_range=TimeRange())
    assert _clause_for_field(clauses, CREATED_AT_FIELD_NAME) is None
