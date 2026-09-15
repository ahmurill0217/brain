# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/configs/constants.py and onyx/prompts/constants.py.
"""Literal constants that are part of the on-disk / on-index format.

Nothing here is configurable: changing any of these values changes the meaning
of data already written to OpenSearch, so they are deliberately not settings.
"""

# Separates a metadata key from its value in the indexed `metadata_list` field,
# e.g. "author===jane". The query-time tag filter must use the same separator.
INDEX_SEPARATOR = "==="

# Joins sections when rendering a document as one blob. Also stripped out of
# titles before they are embedded.
RETURN_SEPARATOR = "\n\r\n"
SECTION_SEPARATOR = "\n\n"

# ACL sentinel for "anyone can see this". Never written to the ACL field itself:
# the index stores it as the boolean `public` column instead.
PUBLIC_ACL_PAT = "PUBLIC"

# ACL string prefixes. Prefixing keeps user emails from colliding with group
# names in a single keyword field.
USER_EMAIL_PREFIX = "user_email:"
EXTERNAL_GROUP_PREFIX = "external_group:"

# Citation / markdown parsing.
TRIPLE_BACKTICK = "```"
CODE_BLOCK_PAT = "```\n{}\n```"

# Metadata key that marks a document as excluded from question answering.
IGNORE_FOR_QA = "ignore_for_qa"

# Chunk id component. Part of the OpenSearch document id, so it is format, not
# configuration: `{document_id}__{max_chunk_size}__{chunk_index}`.
DEFAULT_MAX_CHUNK_SIZE = 512
