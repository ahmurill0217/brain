"""ACL parity.

The write side and the read side must produce strings that match exactly, or
search silently returns the wrong documents: too few (annoying) or too many
(a leak). These tests pin the string format and the visibility matrix.
"""

from __future__ import annotations

import pytest

from brain.models.acl import (
    AccessScope,
    ExternalAccess,
    acl_filter_for_scope,
    acl_for_document,
    prefix_external_group,
    prefix_user_email,
)


def test_prefixes_match_index_format() -> None:
    # These exact strings are what existing indexes contain. Changing them
    # would make those indexes unreadable.
    assert prefix_user_email("a@x.com") == "user_email:a@x.com"
    assert prefix_external_group("g1") == "external_group:g1"


def test_to_acl_omits_the_public_marker() -> None:
    # Public is stored as a boolean field on the chunk, not as an ACL entry:
    # a term filter on a boolean is much cheaper than one on a keyword list.
    access = ExternalAccess(external_user_emails={"a@x.com"}, is_public=True)
    assert access.to_acl() == {"user_email:a@x.com"}


def test_to_acl_covers_emails_and_groups() -> None:
    access = ExternalAccess(
        external_user_emails={"a@x.com", "b@x.com"},
        external_user_group_ids={"eng", "sales"},
    )
    assert access.to_acl() == {
        "user_email:a@x.com",
        "user_email:b@x.com",
        "external_group:eng",
        "external_group:sales",
    }


def test_empty_strings_are_dropped() -> None:
    # An empty entry would match an empty filter value and quietly widen access.
    access = ExternalAccess(external_user_emails={""}, external_user_group_ids={""})
    assert access.to_acl() == set()


def test_bypass_disables_filtering_entirely() -> None:
    # None and [] are different: None removes the filter, [] restricts to public.
    assert acl_filter_for_scope(AccessScope(bypass=True)) is None


def test_anonymous_scope_sees_only_public() -> None:
    assert acl_filter_for_scope(AccessScope()) == []


def test_scope_filter_lists_identity_and_groups() -> None:
    scope = AccessScope(user_email="a@x.com", external_group_ids=["eng", "sales"])
    assert acl_filter_for_scope(scope) == [
        "user_email:a@x.com",
        "external_group:eng",
        "external_group:sales",
    ]


@pytest.mark.parametrize("default_public", [True, False])
def test_missing_access_follows_the_default(default_public: bool) -> None:
    # "No permission info" is treated as public by default, but it is a
    # setting so a deployment can fail closed instead.
    is_public, acl = acl_for_document(None, default_public=default_public)
    assert is_public is default_public
    assert acl == []


def test_explicit_access_ignores_the_default() -> None:
    access = ExternalAccess(external_user_emails={"a@x.com"})
    is_public, acl = acl_for_document(access, default_public=True)
    assert is_public is False
    assert acl == ["user_email:a@x.com"]


def test_acl_output_is_sorted() -> None:
    # Sorted output keeps the indexed document byte-identical across runs, so
    # re-indexing unchanged content produces no diff.
    access = ExternalAccess(
        external_user_emails={"z@x.com", "a@x.com"},
        external_user_group_ids={"zeta", "alpha"},
    )
    _, acl = acl_for_document(access, default_public=False)
    assert acl == sorted(acl)


@pytest.mark.parametrize(
    ("doc_access", "scope", "should_match"),
    [
        (ExternalAccess.public(), AccessScope(), True),
        (ExternalAccess.public(), AccessScope(user_email="a@x.com"), True),
        (ExternalAccess(external_user_emails={"a@x.com"}), AccessScope(), False),
        (
            ExternalAccess(external_user_emails={"a@x.com"}),
            AccessScope(user_email="a@x.com"),
            True,
        ),
        (
            ExternalAccess(external_user_emails={"a@x.com"}),
            AccessScope(user_email="b@x.com"),
            False,
        ),
        (
            ExternalAccess(external_user_group_ids={"eng"}),
            AccessScope(user_email="b@x.com", external_group_ids=["eng"]),
            True,
        ),
        (
            ExternalAccess(external_user_group_ids={"eng"}),
            AccessScope(user_email="b@x.com", external_group_ids=["sales"]),
            False,
        ),
        (ExternalAccess.private(), AccessScope(bypass=True), True),
    ],
)
def test_visibility_matrix(
    doc_access: ExternalAccess, scope: AccessScope, should_match: bool
) -> None:
    """Mirrors what the index does: public OR (acl list intersects filter)."""
    is_public, doc_acl = acl_for_document(doc_access, default_public=False)
    scope_acl = acl_filter_for_scope(scope)

    if scope_acl is None:  # bypass sees everything
        visible = True
    else:
        visible = is_public or bool(set(doc_acl) & set(scope_acl))

    assert visible is should_match


def test_private_helper_is_visible_to_nobody() -> None:
    is_public, acl = acl_for_document(ExternalAccess.private(), default_public=True)
    assert is_public is False
    assert acl == []
