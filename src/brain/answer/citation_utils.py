# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/chat/citation_utils.py.
"""Citation bookkeeping around the processor.

An answer assembled from several search rounds ends up citing [17] and [104]
because each round numbered its own results. `collapse_citations` renumbers that
text down to 1, 2, 3 while keeping numbers the reader has already seen stable,
which is why it takes the existing mapping as well as the new one.
"""

from __future__ import annotations

import re

from brain.answer.citation_processor import CitationMapping
from brain.models.search import SearchDoc

# Same shape as DynamicCitationProcessor.citation_pattern, but with the numbers
# captured: group 2 for the double-bracket form, group 4 for the single.
_CITATION_ORDER_PATTERN = re.compile(r"([\[【［]{2}(\d+)[\]】］]{2})|([\[【［]([\d]+(?: *, *\d+)*)[\]】］])")  # noqa: RUF001

# Matches whole citations without capturing the numbers; used for substitution.
_CITATION_PATTERN = re.compile(r"([\[【［]{2}\d+[\]】］]{2})|([\[【［]\d+(?:, ?\d+)*[\]】］])")  # noqa: RUF001


def citation_mapping_from_search_result(
    citation_mapping: dict[int, str],
    search_docs: list[SearchDoc],
) -> CitationMapping:
    """Turn a search result's number -> document_id mapping into number -> SearchDoc.

    A search hands back the citation numbers it showed the model plus the
    documents themselves; the processor needs the two joined so it can render a
    link. Numbers whose document is missing from `search_docs` are dropped
    rather than guessed at.
    """
    docs_by_id = {doc.document_id: doc for doc in search_docs}
    return {
        citation_num: docs_by_id[doc_id]
        for citation_num, doc_id in citation_mapping.items()
        if doc_id in docs_by_id
    }


def extract_citation_order_from_text(text: str) -> list[int]:
    """Citation numbers in order of first appearance, without duplicates.

    Parses [1], [1, 2], [[1]], 【1】 and the other bracket variants.
    """
    seen: set[int] = set()
    order: list[int] = []

    for match in _CITATION_ORDER_PATTERN.finditer(text):
        if match.group(2):
            nums_str = match.group(2)
        elif match.group(4):
            nums_str = match.group(4)
        else:
            continue

        for raw_num_str in nums_str.split(","):
            num_str = raw_num_str.strip()
            if not num_str:
                continue
            try:
                num = int(num_str)
            except ValueError:
                continue
            if num not in seen:
                seen.add(num)
                order.append(num)

    return order


def collapse_citations(
    answer_text: str,
    existing_citation_mapping: CitationMapping,
    new_citation_mapping: CitationMapping,
) -> tuple[str, CitationMapping]:
    """Renumber the citations in `answer_text` to the smallest free numbers.

    Numbering continues after the highest key in `existing_citation_mapping`,
    whose entries are passed through untouched. A new citation that points at a
    document already in the existing mapping reuses that document's number
    instead of taking a fresh one, so the same source never appears twice in the
    reader's source list.

    Args:
        answer_text: Text containing citations to collapse, e.g. "See [25]".
        existing_citation_mapping: Citations already shown to the reader.
        new_citation_mapping: Citations in this text, keyed by the numbers as
            they appear in `answer_text`.

    Returns:
        (rewritten text, existing mapping plus the renumbered new entries).
    """
    doc_id_to_existing_citation: dict[str, int] = {
        doc.document_id: citation_num for citation_num, doc in existing_citation_mapping.items()
    }

    if existing_citation_mapping:
        next_citation_num = max(existing_citation_mapping.keys()) + 1
    else:
        next_citation_num = 1

    old_to_new: dict[int, int] = {}
    additional_mappings: CitationMapping = {}

    for old_num, search_doc in new_citation_mapping.items():
        doc_id = search_doc.document_id

        if doc_id in doc_id_to_existing_citation:
            old_to_new[old_num] = doc_id_to_existing_citation[doc_id]
            continue

        # The same document can arrive under two different old numbers; both
        # must collapse onto the one new number.
        existing_new_num = None
        for mapped_old, mapped_new in old_to_new.items():
            if (
                mapped_old in new_citation_mapping
                and new_citation_mapping[mapped_old].document_id == doc_id
            ):
                existing_new_num = mapped_new
                break

        if existing_new_num is not None:
            old_to_new[old_num] = existing_new_num
        else:
            old_to_new[old_num] = next_citation_num
            additional_mappings[next_citation_num] = search_doc
            next_citation_num += 1

    def replace_citation(match: re.Match) -> str:
        citation_str = match.group()

        if citation_str.startswith(("[[", "【【", "［［")):  # noqa: RUF001
            open_bracket = citation_str[:2]
            close_bracket = citation_str[-2:]
            content = citation_str[2:-2]
        else:
            open_bracket = citation_str[0]
            close_bracket = citation_str[-1]
            content = citation_str[1:-1]

        new_nums = []
        for raw_num_str in content.split(","):
            num_str = raw_num_str.strip()
            if not num_str:
                continue
            try:
                num = int(num_str)
            except ValueError:
                new_nums.append(num_str)
                continue
            # A number with no mapping is left alone: it belongs to some other
            # numbering space and renumbering it would point at the wrong doc.
            new_nums.append(str(old_to_new[num]) if num in old_to_new else num_str)

        new_content = ", ".join(new_nums)
        return f"{open_bracket}{new_content}{close_bracket}"

    updated_text = _CITATION_PATTERN.sub(replace_citation, answer_text)

    combined_mapping: CitationMapping = dict(existing_citation_mapping)
    combined_mapping.update(additional_mappings)

    return updated_text, combined_mapping
