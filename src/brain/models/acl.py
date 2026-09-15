# Derived from onyx/access/models.py and onyx/access/utils.py.
"""Access control.

Two directions, and they must agree or search silently returns the wrong rows:

  write: a document's `ExternalAccess` becomes (is_public, [acl strings])
  read:  a viewer's `AccessScope` becomes the acl filter list

The index stores `public` as a boolean and everything else as prefixed strings
in a keyword field. A query matches when the document is public OR its acl list
intersects the viewer's filter list.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from brain.constants import EXTERNAL_GROUP_PREFIX, USER_EMAIL_PREFIX

# Guard rail from Onyx: a permission set larger than this is almost always a bug
# in the caller's permission sync, and it blows up the index document size.
MAX_ACL_ENTRIES = 5000


def prefix_user_email(user_email: str) -> str:
    """Prefix a user email so it cannot collide with a group name."""
    return f"{USER_EMAIL_PREFIX}{user_email}"


def prefix_external_group(group_id: str) -> str:
    """Prefix a group id so it cannot collide with a user email."""
    return f"{EXTERNAL_GROUP_PREFIX}{group_id}"


class ExternalAccess(BaseModel):
    """Who can see a document, as supplied by the caller at ingest time.

    `is_public` wins: a public document is visible to everyone regardless of the
    email and group sets.
    """

    external_user_emails: set[str] = Field(default_factory=set)
    external_user_group_ids: set[str] = Field(default_factory=set)
    is_public: bool = False

    model_config = {"frozen": True}

    @classmethod
    def public(cls) -> ExternalAccess:
        return cls(is_public=True)

    @classmethod
    def private(cls) -> ExternalAccess:
        """Visible to nobody.

        The right fallback when a permission lookup fails: failing closed keeps
        a document out of results rather than leaking it.
        """
        return cls(is_public=False)

    @property
    def num_entries(self) -> int:
        return len(self.external_user_emails) + len(self.external_user_group_ids)

    def to_acl(self) -> set[str]:
        """The prefixed strings for this document, excluding the public marker.

        The public flag is carried separately because the index stores it as a
        boolean field, which is far cheaper to filter on than a terms match.
        """
        acl: set[str] = {prefix_user_email(e) for e in self.external_user_emails if e}
        acl |= {prefix_external_group(g) for g in self.external_user_group_ids if g}
        return acl

    def __str__(self) -> str:
        def truncate(s: set[str], max_len: int = 100) -> str:
            text = str(s)
            return f"{text[:max_len]}... ({len(s)} items)" if len(text) > max_len else text

        return (
            f"ExternalAccess(emails={truncate(self.external_user_emails)}, "
            f"groups={truncate(self.external_user_group_ids)}, is_public={self.is_public})"
        )


class AccessScope(BaseModel):
    """Who is asking. Replaces Onyx's ORM User in every query path."""

    user_email: str | None = None
    external_group_ids: list[str] = Field(default_factory=list)
    # Admin / system reads: skip the ACL filter entirely and see every document.
    # Never set this from an end user's request.
    bypass: bool = False

    @classmethod
    def anonymous(cls) -> AccessScope:
        """No identity: public documents only."""
        return cls()

    @classmethod
    def admin(cls) -> AccessScope:
        return cls(bypass=True)


def acl_filter_for_scope(scope: AccessScope) -> list[str] | None:
    """The acl strings to match against, or None to disable ACL filtering.

    An empty list is meaningful and different from None: it restricts results to
    public documents. None removes the filter altogether.
    """
    if scope.bypass:
        return None
    acl: list[str] = []
    if scope.user_email:
        acl.append(prefix_user_email(scope.user_email))
    acl.extend(prefix_external_group(g) for g in scope.external_group_ids if g)
    return acl


def acl_for_document(
    external_access: ExternalAccess | None,
    *,
    default_public: bool,
) -> tuple[bool, list[str]]:
    """Split a document's access into (is_public, sorted acl strings).

    `external_access=None` means the caller did not supply permissions. Onyx
    treats that as public; `default_public` makes that choice explicit so a
    deployment can fail closed instead.
    """
    if external_access is None:
        return default_public, []
    return external_access.is_public, sorted(external_access.to_acl())
