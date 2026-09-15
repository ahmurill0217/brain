# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/utils/text_processing.py.
"""String cleaning used on both the index and query sides.

These have to agree: text is cleaned before embedding and the same cleaning is
applied to queries, so a change here is effectively a reindex.
"""

from __future__ import annotations

import re
from urllib.parse import quote

CURLY_TO_STRAIGHT_QUOTES: dict[str, str] = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
}

# Characters that add embedding noise without adding meaning: emoji, dingbats,
# arrows, zero-width and bidi format controls.
_INITIAL_FILTER = re.compile(
    "["
    "\U0000fff0-\U0000ffff"  # specials
    "\U0001f000-\U0001f9ff"  # emoticons
    "\U0000200b-\U0000200f\U0000202a-\U0000202e\U00002060-\U0000206f"  # format controls
    "\U00002190-\U000021ff"  # arrows
    "\U00002700-\U000027bf"  # dingbats
    "]+",
    flags=re.UNICODE,
)

# Characters that make a UTF-8 encode raise. Unpaired surrogates in particular
# come out of some PDF and Office extractors and will fail the index write.
_INVALID_UNICODE_CHARS_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff﷐-﷯￾￿]"
)


def clean_text(text: str) -> str:
    """Strip noise characters and control codes, keeping newlines and tabs."""
    cleaned = _INITIAL_FILTER.sub("", text)
    return "".join(ch for ch in cleaned if ch >= " " or ch in "\n\t")


def remove_invalid_unicode_chars(text: str) -> str:
    """Remove characters that cannot be encoded as UTF-8.

    Must run on every string before it is sent to OpenSearch, which rejects the
    whole bulk request if one document contains an unpaired surrogate.
    """
    return _INVALID_UNICODE_CHARS_RE.sub("", text)


def shared_precompare_cleanup(text: str) -> str:
    """Aggressively normalize text for equality comparison only.

    Models reflow whitespace and swap punctuation for things that look more like
    their training data, which breaks exact quote matching. Lossy by design;
    never store the result.
    """
    text = text.lower()
    return re.sub(r'\s|\*|\\"|[.,:`"#-]', "", text)


def normalize_curly_quotes(text: str) -> str:
    for curly, straight in CURLY_TO_STRAIGHT_QUOTES.items():
        text = text.replace(curly, straight)
    return text


def make_url_compatible(s: str) -> str:
    return quote(s.replace(" ", "_"), safe="")


def replace_whitespaces_w_space(s: str) -> str:
    return re.sub(r"\s", " ", s)


def escape_newlines(s: str) -> str:
    return re.sub(r"(?<!\\)\n", "\\\\n", s)
