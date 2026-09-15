# Derived from onyx/chat/citation_processor.py.
"""Resolving citation markers inside a token stream.

The hard part is that a citation arrives in pieces. `[`, `1`, `]` may be three
tokens, and until the closing bracket shows up there is no way to know whether
`[` starts a citation or a markdown link. The processor therefore holds back any
suffix that could still become a citation and releases it the moment the
question is settled, so the caller can stream everything else immediately.

Three output modes exist because the same answer text is consumed in different
places: a UI wants clickable `[[1]](url)` links, an intermediate research pass
wants the raw `[1]` markers so it can renumber them later, and a channel that
cannot show the sources at all wants them gone. All three track what was cited.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from enum import Enum

from brain.constants import TRIPLE_BACKTICK
from brain.models.search import CitationInfo, SearchDoc

logger = logging.getLogger(__name__)


class CitationMode(Enum):
    """How citations should be handled in the output.

    REMOVE: Citations are completely removed from output text. No CitationInfo
            objects are emitted. For surfaces where the sources are not shared
            with the reader.

    KEEP_MARKERS: Original citation markers like [1], [2] are preserved
            unchanged. No CitationInfo objects are emitted. For text that will
            be renumbered later with `collapse_citations`.

    HYPERLINK: Citations are replaced with markdown links like [[1]](url), and
            CitationInfo objects are emitted for UI tracking. For final answers
            shown to a user.
    """

    REMOVE = "remove"
    KEEP_MARKERS = "keep_markers"
    HYPERLINK = "hyperlink"


type CitationMapping = dict[int, SearchDoc]


def in_code_block(llm_text: str) -> bool:
    """Whether the text so far ends inside a fenced code block.

    Parity of the fence count is the whole test: an odd number of ``` means the
    last one opened a block that has not closed.
    """
    count = llm_text.count(TRIPLE_BACKTICK)
    return count % 2 != 0


class DynamicCitationProcessor:
    """Streams LLM tokens out, turning citation markers into the configured form.

    "Dynamic" because the number-to-document mapping arrives during the stream
    rather than up front: a search tool can run mid-answer, and its documents
    have to become citable without restarting the processor.

    Feed it one token at a time and it yields whatever is safe to display. In
    HYPERLINK mode each CitationInfo is yielded *before* the text containing its
    marker, so a frontend has the metadata in hand by the time the link arrives.
    """

    def __init__(
        self,
        citation_mode: CitationMode = CitationMode.HYPERLINK,
        stop_stream: str | None = None,
    ) -> None:
        """
        Args:
            citation_mode: How to render citations in the output. All three modes
                track what was cited via `get_seen_citations()`.
            stop_stream: Optional pattern that halts processing when it appears in
                the stream. Onyx reads this from a global config; brain takes it
                as an argument so the caller can pass `settings.stop_stream_pat`.
        """
        self.citation_to_doc: CitationMapping = {}
        self.seen_citations: CitationMapping = {}

        # Token processing state.
        self.llm_out = ""  # everything received so far
        self.curr_segment = ""  # held back pending citation resolution
        self.hold = ""  # held back pending stop-token resolution
        self.stop_stream = stop_stream
        self.citation_mode = citation_mode

        # Citation tracking.
        self.cited_documents_in_order: list[SearchDoc] = []
        self.cited_document_ids: set[str] = set()
        self.recent_cited_documents: set[str] = set()
        self.non_citation_count = 0

        # Matches potential incomplete citations: '[', '[[', '[1', '[[1', '[1,',
        # '[1, ', etc. Also matches the lenticular (【】) and fullwidth bracket
        # variants some models emit instead of ASCII brackets.
        #
        # NOTE: the inner group is written as `\d+(?:, ?\d+)*` (comma-separated runs of
        # digits) rather than `(?:\d+,? ?)*`. The latter nests an unbounded quantifier
        # (`\d+`) inside another unbounded quantifier (`(?:...)*`) with optional
        # separators, which makes a solid run of digits ambiguous to parse. When the
        # match ultimately fails the trailing `$` anchor, the engine backtracks through
        # exponentially many ways of splitting the digits (O(2^n)), pinning a CPU core.
        # The comma-separated form has exactly one way to parse a digit run, so it stays
        # linear. This must mirror `citation_pattern` below, which already requires
        # commas between numbers, so no real (closeable) citation is missed.
        self.possible_citation_pattern = re.compile(r"([\[【［]+(?:\d+(?:, ?\d+)*(?:, ?)?)?$)")  # noqa: RUF001

        # Matches complete citations:
        # group 1: '[[1]]', '[[2]]', etc. (also the doubled and single lenticular
        #          and fullwidth forms)
        # group 2: '[1]', '[1, 2]', '[1,2,16]', etc. (also the unicode variants)
        self.citation_pattern = re.compile(r"([\[【［]{2}\d+[\]】］]{2})|([\[【［]\d+(?:, ?\d+)*[\]】］])")  # noqa: RUF001

    def update_citation_mapping(
        self,
        citation_mapping: CitationMapping,
        update_duplicate_keys: bool = False,
    ) -> None:
        """Merge more citation number -> SearchDoc entries into the mapping.

        Args:
            citation_mapping: Citation numbers (1, 2, 3, ...) to SearchDocs.
            update_duplicate_keys: If True, overwrite entries whose keys already
                exist. If False (the default), keys already present are left
                alone. The default matters when two tools hand out overlapping
                numbers: the first registration is the one the model was shown,
                so it is the one that must stay.
        """
        if update_duplicate_keys:
            self.citation_to_doc.update(citation_mapping)
        else:
            duplicate_keys = set(citation_mapping.keys()) & set(self.citation_to_doc.keys())
            non_duplicate_mapping = {
                k: v for k, v in citation_mapping.items() if k not in duplicate_keys
            }
            self.citation_to_doc.update(non_duplicate_mapping)

    def process_token(self, token: str | None) -> Generator[str | CitationInfo]:
        """Consume one token and yield whatever is now safe to emit.

        Args:
            token: The next token from the LLM stream, or None to signal the end
                of the stream and flush whatever is still buffered.

        Yields:
            str: Text to display, with citations rendered per `citation_mode`.
            CitationInfo: Citation metadata, HYPERLINK mode only, yielded before
                the text that contains the corresponding marker.
        """
        # None -> end of stream, flush remaining segment.
        if token is None:
            if self.curr_segment:
                yield self.curr_segment
            return

        # Handle stop stream token.
        if self.stop_stream:
            next_hold = self.hold + token
            if self.stop_stream in next_hold:
                stop_pos = next_hold.find(self.stop_stream)
                text_before_stop = next_hold[:stop_pos]
                if text_before_stop:
                    # Emit what came before the stop pattern, then fall through
                    # to normal processing for it.
                    self.hold = ""
                    token = text_before_stop
                else:
                    # Stop pattern at the beginning, nothing to yield.
                    return
            elif next_hold == self.stop_stream[: len(next_hold)]:
                # Could still grow into the stop pattern; hold it back.
                self.hold = next_hold
                return
            else:
                token = next_hold
                self.hold = ""

        self.curr_segment += token
        self.llm_out += token

        # A bare ``` fence renders as an unstyled block in most markdown
        # viewers, so label it explicitly.
        if "`" in self.curr_segment:
            if self.curr_segment.endswith("`"):
                pass
            elif "```" in self.curr_segment:
                parts = self.curr_segment.split("```")
                if len(parts) > 1 and len(parts[1]) > 0:
                    piece_that_comes_after = parts[1][0]
                    if piece_that_comes_after == "\n" and in_code_block(self.llm_out):
                        # Label only this first bare fence; other fences in the
                        # buffered segment must stay untouched.
                        self.curr_segment = parts[0] + "```plaintext" + "```".join(parts[1:])

        citation_matches = list(self.citation_pattern.finditer(self.curr_segment))
        possible_citation_found = bool(
            re.search(self.possible_citation_pattern, self.curr_segment)
        )

        result = ""
        if citation_matches and not in_code_block(self.llm_out):
            match_idx = 0
            for match in citation_matches:
                match_span = match.span()

                intermatch_str = self.curr_segment[match_idx : match_span[0]]
                self.non_citation_count += len(intermatch_str)
                match_idx = match_span[1]

                if intermatch_str:
                    has_leading_space = intermatch_str[-1].isspace()
                elif match_idx > 0:
                    # Consecutive citations: no space between them.
                    has_leading_space = True
                else:
                    # Citation at the start of the segment: the space, if any,
                    # is in text already emitted. Unreachable as written, since
                    # match_idx was just advanced past this match and every
                    # match is at least three characters wide, so the branch
                    # above always wins. Kept because removing it would be a
                    # behavior change the moment the ordering above is fixed.
                    segment_start_idx = len(self.llm_out) - len(self.curr_segment)
                    if segment_start_idx > 0:
                        has_leading_space = self.llm_out[segment_start_idx - 1].isspace()
                    else:
                        has_leading_space = False

                # Enough plain text has gone by that a repeat citation is a new
                # claim rather than the same sentence citing twice.
                if self.non_citation_count > 5:
                    self.recent_cited_documents.clear()

                # Always tracks seen citations, whatever the mode.
                citation_text, citation_info_list = self._process_citation(
                    match, has_leading_space
                )

                if self.citation_mode == CitationMode.HYPERLINK:
                    # Text before the citation goes first, to preserve order.
                    if intermatch_str:
                        yield intermatch_str
                    # CitationInfo before the citation text, so a frontend has
                    # the metadata before the token carrying [[n]](link) and can
                    # render it immediately.
                    yield from citation_info_list
                    if citation_text:
                        yield citation_text

                elif self.citation_mode == CitationMode.KEEP_MARKERS:
                    if intermatch_str:
                        yield intermatch_str
                    yield match.group()

                else:  # CitationMode.REMOVE
                    # Deleting the marker can leave "text  more" or "text ." so
                    # the space in front of it goes too when what follows is a
                    # space or closing punctuation.
                    if intermatch_str:
                        remaining_text = self.curr_segment[match_span[1] :]
                        if intermatch_str[-1].isspace() and remaining_text:
                            first_char = remaining_text[0]
                            if first_char.isspace() or first_char in ".,;:!?)]}":
                                intermatch_str = intermatch_str.rstrip()
                        if intermatch_str:
                            yield intermatch_str

                self.non_citation_count = 0

            # Whatever trails the last citation could start the next one.
            self.curr_segment = self.curr_segment[match_idx:]
            self.non_citation_count = len(self.curr_segment)

        # Hold the segment back if it could still grow into a citation.
        if not possible_citation_found:
            result += self.curr_segment
            self.non_citation_count += len(self.curr_segment)
            self.curr_segment = ""

        if result:
            yield result

    def _process_citation(
        self, match: re.Match, has_leading_space: bool
    ) -> tuple[str, list[CitationInfo]]:
        """Resolve one matched marker into rendered text and CitationInfos.

        The match may be '[1]', '[1, 13, 6]', '[[4]]', '【1】', or a fullwidth-bracket
        equivalent.
        Numbers with no entry in the mapping are dropped: the model hallucinated
        a source, and pointing at a document that was never retrieved is worse
        than silence.

        Returns:
            (rendered text, new CitationInfos). Both are empty outside HYPERLINK
            mode, where the caller renders the marker itself.
        """
        citation_str: str = match.group()
        # lastindex 1 means the double-bracket group matched: already '[[1]]'.
        formatted = match.lastindex == 1

        citation_info_list: list[CitationInfo] = []
        formatted_citation_parts: list[str] = []

        # The regex guarantees matched brackets, so slicing them off is safe.
        citation_content = citation_str[2:-2] if formatted else citation_str[1:-1]

        for raw_num_str in citation_content.split(","):
            num_str = raw_num_str.strip()
            if not num_str:
                continue

            try:
                num = int(num_str)
            except ValueError:
                logger.warning("Invalid citation number format: %s", num_str)
                continue

            if num not in self.citation_to_doc:
                logger.warning(
                    "Citation number %s not found in mapping. Available: %s",
                    num,
                    list(self.citation_to_doc.keys()),
                )
                continue

            search_doc = self.citation_to_doc[num]
            doc_id = search_doc.document_id
            link = search_doc.link or ""

            # Tracked in every mode.
            self.seen_citations[num] = search_doc

            if self.citation_mode != CitationMode.HYPERLINK:
                continue

            formatted_citation_parts.append(f"[[{num}]]({link})")

            # The same document cited again a few characters later is the same
            # claim; emitting a second CitationInfo would double it in the UI.
            if doc_id in self.recent_cited_documents:
                continue
            self.recent_cited_documents.add(doc_id)

            if doc_id not in self.cited_document_ids:
                self.cited_document_ids.add(doc_id)
                self.cited_documents_in_order.append(search_doc)
                citation_info_list.append(
                    CitationInfo(citation_number=num, document_id=doc_id)
                )

        formatted_citation_text = " ".join(formatted_citation_parts)

        if formatted_citation_text and not has_leading_space:
            formatted_citation_text = " " + formatted_citation_text

        return formatted_citation_text, citation_info_list

    def get_cited_documents(self) -> list[SearchDoc]:
        """The cited SearchDocs, in the order they were first cited.

        Only populated in HYPERLINK mode; use `get_seen_citations()` otherwise.
        """
        return self.cited_documents_in_order

    def get_cited_document_ids(self) -> list[str]:
        """The cited document ids, in the order they were first cited.

        Only populated in HYPERLINK mode; use `get_seen_citations()` otherwise.
        """
        return [doc.document_id for doc in self.cited_documents_in_order]

    def get_seen_citations(self) -> CitationMapping:
        """Every citation number encountered in the text, in any mode.

        Keyed by the number as it appeared in the text, so this is what tells a
        REMOVE- or KEEP_MARKERS-mode caller which documents the answer leaned on.
        """
        return self.seen_citations

    @property
    def num_cited_documents(self) -> int:
        """Count of unique documents cited. Always 0 outside HYPERLINK mode."""
        return len(self.cited_document_ids)

    def reset_recent_citations(self) -> None:
        """Clear the dedupe window so a recent document can emit CitationInfo again.

        The window also clears itself once more than 5 non-citation characters
        have gone by.
        """
        self.recent_cited_documents.clear()

    def get_next_citation_number(self) -> int:
        """The next free citation number, for documents added mid-stream.

        Returns 1 when the mapping is empty, otherwise max(keys) + 1, so numbers
        already shown to the model are never reused for a different document.
        """
        if not self.citation_to_doc:
            return 1
        return max(self.citation_to_doc.keys()) + 1
