# Derived from onyx/indexing/chunking/tabular_section_chunker/util.py.
"""Shared helpers for rendering a sheet as retrievable text."""

from __future__ import annotations

from brain.text.tokenizer import BaseTokenizer, count_tokens


def label(name: str) -> str:
    """Render a column name with a space-substituted alias in parens.

    A query says "MTTR hours"; the header says `MTTR_hours`. Emitting both
    surface forms means either one matches on the keyword side.
    """
    return f"{name} ({name.replace('_', ' ')})" if "_" in name else name


def pack_lines(
    lines: list[str],
    prefix: str,
    tokenizer: BaseTokenizer,
    max_tokens: int,
) -> list[str]:
    """Greedily pack `lines` into chunks of at most `max_tokens`.

    `prefix` is prepended verbatim to every emitted chunk, so a descriptor that
    has to split still carries its sheet heading into each piece. A line that
    does not fit the post-prefix budget on its own is dropped rather than
    truncated: half a descriptor line is worse than none.
    """
    prefix_tokens = count_tokens(prefix, tokenizer) + 1 if prefix else 0
    budget = max_tokens - prefix_tokens

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        line_tokens = count_tokens(line, tokenizer)
        if line_tokens > budget:
            continue
        sep = 1 if current else 0
        if current_tokens + sep + line_tokens > budget:
            chunks.append(_join_with_prefix(current, prefix))
            current = [line]
            current_tokens = line_tokens
        else:
            current.append(line)
            current_tokens += sep + line_tokens
    if current:
        chunks.append(_join_with_prefix(current, prefix))
    return chunks


def _join_with_prefix(lines: list[str], prefix: str) -> str:
    body = "\n".join(lines)
    return f"{prefix}\n{body}" if prefix else body
