"""Documents to indexed chunks.

`IngestPipeline` is the entry point. The stages are exported alongside it
because they are worth testing and reusing on their own — `get_docs_to_update`
in particular, which decides whether a document is worth re-indexing at all.
"""

from brain.ingest.contextual_rag import (
    add_chunk_summaries,
    add_contextual_summaries,
    add_document_summaries,
)
from brain.ingest.image_sections import process_image_sections
from brain.ingest.pipeline import (
    IngestPipeline,
    filter_documents,
    get_docs_to_update,
)
from brain.ingest.vector_write import write_chunks_to_vector_db_with_backoff

__all__ = [
    "IngestPipeline",
    "add_chunk_summaries",
    "add_contextual_summaries",
    "add_document_summaries",
    "filter_documents",
    "get_docs_to_update",
    "process_image_sections",
    "write_chunks_to_vector_db_with_backoff",
]
