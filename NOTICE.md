# Attribution

brain is MIT licensed. Substantial portions are derived from
[Onyx](https://github.com/onyx-dot-app/onyx-foss), copyright DanswerAI, Inc.,
also MIT licensed. Reference commit: `67355bae264628f3323c5b491b46ba5ebf0fcfd6`.

Each derived module carries a one-line `# Derived from onyx/...` comment naming
its source. That comment is a pointer for anyone comparing behaviour against
upstream, not a licensing requirement: `LICENSE` at the repository root is
what satisfies MIT, and this file is the complete map. Modules not listed are
original work.

| brain module | Onyx source |
|---|---|
| `constants.py` | `onyx/configs/constants.py`, `onyx/prompts/constants.py` |
| `config.py` | `shared_configs/configs.py`, `onyx/configs/{app,chat,model}_configs.py`, `onyx/document_index/opensearch/constants.py`, `onyx/tools/tool_implementations/search/constants.py` |
| `models/acl.py` | `onyx/access/models.py`, `onyx/access/utils.py` |
| `models/document.py` | `onyx/connectors/models.py`, `onyx/connectors/cross_connector_utils/miscellaneous_utils.py` |
| `models/chunks.py` | `onyx/indexing/models.py` |
| `models/search.py` | `onyx/context/search/models.py`, `onyx/context/search/utils.py`, `onyx/server/query_and_chat/streaming_models.py` |
| `models/llm.py` | `onyx/llm/models.py`, `onyx/llm/model_response.py` |
| `models/results.py` | `IndexingPipelineResult` from `onyx/indexing/indexing_pipeline.py`, `ToolResponse` from `onyx/tools/tool_implementations/search/search_tool.py` |
| `text/processing.py` | `onyx/utils/text_processing.py` |
| `text/tokenizer.py` | `onyx/natural_language_processing/utils.py` |
| `text/stopwords.py` | `onyx/natural_language_processing/english_stopwords.py` |
| `text/csv_utils.py` | `onyx/utils/csv_utils.py` |
| `text/enrichment.py` | `onyx/document_index/chunk_content_enrichment.py` |
| `text/parallel.py` | `onyx/utils/threadpool_concurrency.py` |
| `extraction/file_types.py` | `onyx/file_processing/file_types.py` |
| `extraction/extract.py` | `onyx/file_processing/extract_file_text.py` |
| `extraction/html.py` | `onyx/file_processing/html_utils.py`, `onyx/file_processing/enums.py` |
| `extraction/pdf_images.py` | `onyx/file_processing/pdf_image_utils.py` |
| `extraction/images.py` | `onyx/file_processing/image_utils.py`, `onyx/utils/b64.py` |
| `extraction/image_summarization.py` | `onyx/file_processing/image_summarization.py`, `onyx/prompts/image_analysis.py` |
| `extraction/password.py` | `onyx/file_processing/password_validation.py` |
| `extraction/isolation.py` | `onyx/utils/process_isolation.py` |
| `extraction/isolated_runner.py` | `onyx/utils/isolated_runner.py` |
| `extraction/documents.py` | section assembly from `onyx/file_processing/extract_file_text.py` (`stage_xlsx_sheets`) and `onyx/file_processing/image_utils.py` |
| `chunking/chunker.py` | `onyx/indexing/chunker.py` |
| `chunking/document_chunker.py` | `onyx/indexing/chunking/document_chunker.py` |
| `chunking/section_chunker.py` | `onyx/indexing/chunking/section_chunker.py` |
| `chunking/text_section.py` | `onyx/indexing/chunking/text_section_chunker.py` |
| `chunking/image_section.py` | `onyx/indexing/chunking/image_section_chunker.py` |
| `chunking/tabular/chunker.py` | `onyx/indexing/chunking/tabular_section_chunker/tabular_section_chunker.py` |
| `chunking/tabular/analysis.py` | `onyx/indexing/chunking/tabular_section_chunker/analysis.py` |
| `chunking/tabular/sheet_descriptor.py` | `onyx/indexing/chunking/tabular_section_chunker/sheet_descriptor.py` |
| `chunking/tabular/total_descriptor.py` | `onyx/indexing/chunking/tabular_section_chunker/total_descriptor.py` |
| `chunking/tabular/util.py` | `onyx/indexing/chunking/tabular_section_chunker/util.py` |
| `llm/protocol.py` | shape informed by `onyx/llm/interfaces.py` |
| `llm/litellm_adapter.py` | `onyx/llm/multi_llm.py`, `onyx/llm/model_response.py` |
| `embedding/protocol.py` | `onyx/natural_language_processing/search_nlp_models.py`, `shared_configs/model_server_models.py` |
| `embedding/model_server_client.py` | `onyx/natural_language_processing/search_nlp_models.py` (local path only) |
| `embedding/chunk_embedder.py` | `onyx/indexing/embedder.py` |
| `embedding/batch_store.py` | `onyx/indexing/chunk_batch_store.py` |
| `store/protocol.py` | semantics from `onyx/indexing/indexing_pipeline.py`, `onyx/indexing/adapters/document_indexing_adapter.py` |
| `store/memory.py` | semantics from `onyx/indexing/indexing_pipeline.py`, `onyx/indexing/adapters/document_indexing_adapter.py` |
| `store/sqlite.py` | semantics from `onyx/indexing/indexing_pipeline.py`, `onyx/db/document.py`, `onyx/indexing/adapters/document_indexing_adapter.py` |
| `index/interface.py` | `onyx/document_index/interfaces_new.py` |
| `index/schema.py` | `onyx/document_index/opensearch/schema.py`, `onyx/document_index/opensearch/string_filtering.py`, `onyx/utils/datetime.py` |
| `index/queries.py` | `onyx/document_index/opensearch/search.py` |
| `index/client.py` | `onyx/document_index/opensearch/client.py`, `onyx/document_index/opensearch/cluster_settings.py` |
| `index/opensearch_index.py` | `onyx/document_index/opensearch/opensearch_document_index.py` |
| `ingest/pipeline.py` | `onyx/indexing/indexing_pipeline.py` (`index_doc_batch`, `index_doc_batch_prepare`, `filter_documents`, `get_docs_to_update`, `_verify_indexing_completeness`) |
| `ingest/image_sections.py` | `onyx/indexing/indexing_pipeline.py` (`_process_image_sections`, `_convert_documents_without_image_summaries`) |
| `ingest/contextual_rag.py` | `onyx/indexing/indexing_pipeline.py` (`add_contextual_summaries`, `add_document_summaries`, `add_chunk_summaries`) |
| `ingest/vector_write.py` | `onyx/indexing/vector_db_insertion.py` |
| `ingest/prompts.py` | `onyx/prompts/contextual_retrieval.py` |
| `retrieval/fusion.py` | `onyx/tools/tool_implementations/search/search_utils.py`, `onyx/context/search/pipeline.py`, `onyx/context/search/retrieval/search_runner.py`, `onyx/tools/tool_implementations/search/search_tool.py` |
| `retrieval/context.py` | `onyx/tools/tool_implementations/utils.py` |
| `retrieval/selection.py` | `onyx/secondary_llm_flows/document_filter.py`, `onyx/tools/tool_implementations/search/search_tool.py`, `onyx/tools/tool_implementations/search/search_utils.py` |
| `retrieval/search.py` | `onyx/context/search/retrieval/search_runner.py`, `onyx/context/search/pipeline.py` |
| `retrieval/query_expansion.py` | `onyx/secondary_llm_flows/query_expansion.py` |
| `retrieval/searcher.py` | `onyx/tools/tool_implementations/search/search_tool.py` (`run`), `onyx/context/search/pipeline.py` |
| `retrieval/prompts.py` | `onyx/prompts/search_prompts.py` |
| `answer/citation_processor.py` | `onyx/chat/citation_processor.py` |
| `answer/citation_utils.py` | `onyx/chat/citation_utils.py` |
| `answer/system_prompt.py` | `onyx/chat/prompt_utils.py`, `onyx/prompts/prompt_utils.py` |
| `answer/prompts/constants.py` | `onyx/prompts/constants.py` |
| `answer/prompts/chat_prompts.py` | `onyx/prompts/chat_prompts.py` |
| `answer/prompts/tool_prompts.py` | `onyx/prompts/tool_prompts.py` |
| `answer/prompts/search_prompts.py` | re-exports `retrieval/prompts.py` |
| `answer/prompts/contextual_retrieval.py` | re-exports `ingest/prompts.py` |

The PDF fixtures in `tests/fixtures/`, their generator, and the extraction test
modules in `tests/unit/extraction/` are derived from Onyx's
`backend/tests/unit/onyx/file_processing/`.

`tests/unit/answer/test_citation_processor.py` and `test_citation_utils.py` are
ported from Onyx's `backend/tests/unit/onyx/chat/`, and
`tests/unit/retrieval/test_fusion.py` from its
`backend/tests/unit/onyx/tools/test_search_utils.py`.

`tests/unit/index/test_get_doc_chunk_id.py`,
`test_document_chunk_serialization.py` and `test_time_cutoff_filter.py` are
ported from Onyx's `backend/tests/unit/onyx/document_index/opensearch/`.

The model server is a separate distribution under `model_server/`, derived from
Onyx's own `backend/model_server/`.

| brain-model-server module | Onyx source |
|---|---|
| `main.py` | `model_server/main.py` |
| `encoders.py` | `model_server/encoders.py` (bi-encoder route only) |
| `management.py` | `model_server/management_endpoints.py` |
| `utils.py` | `model_server/utils.py`, `model_server/constants.py` |
| `schemas.py` | `shared_configs/model_server_models.py` |
| `settings.py` | `shared_configs/configs.py` |
| `__main__.py` | `model_server/__main__.py` |
| `Dockerfile` | `backend/Dockerfile.model_server` |
| `pyproject.toml` | pins from `backend/requirements/model_server.txt` |
