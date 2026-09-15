# Attribution

brain is MIT licensed. Substantial portions are derived from
[Onyx](https://github.com/onyx-dot-app/onyx-foss), copyright DanswerAI, Inc.,
also MIT licensed. Reference commit: `67355bae264628f3323c5b491b46ba5ebf0fcfd6`.

Each derived module carries a header naming its Onyx source. This file is the
index. Modules not listed are original work.

| brain module | Onyx source |
|---|---|
| `constants.py` | `onyx/configs/constants.py`, `onyx/prompts/constants.py` |
| `config.py` | `shared_configs/configs.py`, `onyx/configs/{app,chat,model}_configs.py`, `onyx/document_index/opensearch/constants.py`, `onyx/tools/tool_implementations/search/constants.py` |
| `models/acl.py` | `onyx/access/models.py`, `onyx/access/utils.py` |
| `models/document.py` | `onyx/connectors/models.py`, `onyx/connectors/cross_connector_utils/miscellaneous_utils.py` |
| `models/chunks.py` | `onyx/indexing/models.py` |
| `models/search.py` | `onyx/context/search/models.py`, `onyx/context/search/utils.py`, `onyx/server/query_and_chat/streaming_models.py` |
| `models/llm.py` | `onyx/llm/models.py`, `onyx/llm/model_response.py` |
| `text/processing.py` | `onyx/utils/text_processing.py` |
| `text/tokenizer.py` | `onyx/natural_language_processing/utils.py` |
| `text/stopwords.py` | `onyx/natural_language_processing/english_stopwords.py` |
| `text/csv_utils.py` | `onyx/utils/csv_utils.py` |
| `text/enrichment.py` | `onyx/document_index/chunk_content_enrichment.py` |
| `text/parallel.py` | `onyx/utils/threadpool_concurrency.py` |
| `llm/protocol.py` | shape informed by `onyx/llm/interfaces.py` |
| `embedding/protocol.py` | `onyx/natural_language_processing/search_nlp_models.py`, `shared_configs/model_server_models.py` |
| `store/protocol.py` | semantics from `onyx/indexing/indexing_pipeline.py`, `onyx/indexing/adapters/document_indexing_adapter.py` |
| `index/interface.py` | `onyx/document_index/interfaces_new.py` |
