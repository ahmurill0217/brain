# Derived from onyx/context/search/retrieval/search_runner.py and
# onyx/context/search/pipeline.py.
"""One query against the index.

Two jobs, both small, both easy to get wrong in ways that are invisible:

`build_index_filters` resolves who is asking into the ACL list the index
matches against. The None-versus-empty-list distinction is the whole security
boundary: None disables ACL filtering, `[]` restricts to public documents. A
bug that turns one into the other either leaks every document or returns none.

`search_chunks` picks the retrieval mode. A keyword-only search skips the
embedding call entirely rather than computing a vector and weighting it to
zero, which is the difference between a cheap query and a model round trip.
"""

from __future__ import annotations

from brain.config import BrainSettings
from brain.embedding.protocol import Embedder, EmbedTextType
from brain.index.interface import DocumentIndex
from brain.models.acl import AccessScope, acl_filter_for_scope
from brain.models.search import ChunkIndexRequest, IndexFilters, InferenceChunk, SearchFilters
from brain.text.stopwords import strip_stopwords

# The only hybrid_alpha the index treats specially: it selects pure BM25.
KEYWORD_ONLY_ALPHA = 0.0


def build_index_filters(
    access: AccessScope,
    filters: SearchFilters | None = None,
) -> IndexFilters:
    """Combine the caller's narrowing with the viewer's access.

    Access is resolved here and nowhere else, so there is one place to read when
    asking whether a search can see a document.
    """
    base = filters.model_dump() if filters is not None else {}
    return IndexFilters(**base, access_control_list=acl_filter_for_scope(access))


def search_chunks(
    index: DocumentIndex,
    embedder: Embedder,
    request: ChunkIndexRequest,
    settings: BrainSettings,
) -> list[InferenceChunk]:
    """Run one retrieval, hybrid or keyword-only.

    The keyword path never touches the embedder. The hybrid path embeds the
    whole query — an embedding of stopword-stripped keywords would sit somewhere
    else in the space — while BM25 gets the stripped form, since "what is the"
    matches everything and ranks nothing.
    """
    num_to_retrieve = request.limit or settings.num_returned_hits

    if request.hybrid_alpha == KEYWORD_ONLY_ALPHA:
        return index.keyword_retrieval(request.query, request.filters, num_to_retrieve)

    query_embedding = embedder.embed([request.query], EmbedTextType.QUERY)[0]
    final_keywords = request.query_keywords or strip_stopwords(request.query)
    return index.hybrid_retrieval(
        request.query,
        query_embedding,
        final_keywords,
        request.filters,
        num_to_retrieve,
    )
