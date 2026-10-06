"""All tunables, in one place.

The chunking and retrieval defaults are the ones retrieval quality was
measured with. That measurement predates the move to gemini-embedding-001, so
treat them as a starting point until they are re-checked on a real corpus.

Reading os.environ at import time across many modules makes settings
untestable and import order significant. brain reads the environment exactly
once, here, and every other module takes `settings` as an argument.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HybridNormalization(str, Enum):
    MIN_MAX = "min_max"
    ZSCORE = "z_score"


class HybridSubqueryConfig(int, Enum):
    """Which subqueries make up the hybrid search, and their fusion weights.

    The weights must line up positionally with the subquery list, so the two are
    chosen together rather than configured separately.
    """

    # title vector + content vector + combined keyword -> [0.1, 0.45, 0.45]
    TITLE_AND_CONTENT_VECTOR = 1
    # content vector + combined keyword -> [0.5, 0.5]  (default)
    CONTENT_VECTOR_ONLY = 2


class BrainSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BRAIN_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    # ---------------------------------------------------------------- OpenSearch
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_username: str = "admin"
    opensearch_password: str = "StrongPassword123!"
    opensearch_use_ssl: bool = True
    # Self-signed by default in the bundled compose stack. Turn on with a real cert.
    opensearch_verify_certs: bool = False
    opensearch_ca_certs: str | None = None
    opensearch_client_timeout_s: int = 60
    opensearch_query_timeout_s: int = 50
    opensearch_index_name: str = "brain_chunks"
    opensearch_num_shards: int = 1
    opensearch_num_replicas: int = 1
    # Applied to title and content at both index and search time. Changing it
    # requires a full reindex.
    opensearch_text_analyzer: str = "english"
    opensearch_match_highlights_enabled: bool = False
    # Sets action.auto_create_index=false so a typo'd index name fails loudly
    # instead of silently creating an empty index.
    opensearch_set_cluster_settings: bool = True

    hybrid_normalization: HybridNormalization = HybridNormalization.MIN_MAX
    hybrid_subquery_config: HybridSubqueryConfig = HybridSubqueryConfig.CONTENT_VECTOR_ONLY
    # Doubles as knn `k`, `pagination_depth`, and the index's `ef_search`.
    hybrid_subquery_candidates: int = 500
    knn_ef_construction: int = 256
    knn_m: int = 32
    # A document with no timestamp is only assumed to fall inside an open-ended
    # "updated since" window when that window starts further back than this.
    assumed_document_age_days: int = 90
    max_result_window: int = 10_000

    # ----------------------------------------------------------------- Vertex AI
    # Credentials are Application Default Credentials: locally
    # `gcloud auth application-default login`, in a deployment a service
    # account. None takes the project those credentials belong to.
    vertex_project: str | None = None
    vertex_location: str = "us-central1"
    vertex_timeout_s: float = 180.0

    # ----------------------------------------------------------------- Embedding
    embedding_model_name: str = "gemini-embedding-001"
    # gemini-embedding-001 produces up to 3072 dimensions and truncates to this.
    # Changing it changes the index mapping, so it requires a full reindex.
    embedding_dim: int = 768
    embedding_normalize: bool = True
    # Chunk size in tokens. The model accepts 2048; 512 is what retrieval was
    # tuned with, and smaller chunks make sharper citations.
    embedding_context_size: int = 512
    # Texts per embedding request (Vertex allows 250), and the estimated tokens
    # per request. Vertex rejects a request over 20,000 tokens, and the estimate
    # below is not Gemini's own count, so the budget leaves headroom.
    embedding_batch_size: int = 100
    embedding_request_token_budget: int = 15_000
    embedding_num_threads: int = 8
    # tiktoken encoding that measures chunks. Gemini's tokenizer is not
    # published, so token counts are estimates; see brain.text.tokenizer.
    tokenizer_encoding: str = "cl100k_base"

    # ------------------------------------------------------------------ Chunking
    blurb_size: int = 128
    mini_chunk_size: int = 150
    large_chunk_ratio: int = 4
    chunk_overlap: int = 0
    # A chunk may not spend more than this fraction of its budget on metadata.
    max_metadata_percentage: float = 0.25
    chunk_min_content: int = 256
    strict_chunk_token_limit: bool = False
    skip_metadata_in_chunk: bool = False
    enable_multipass_indexing: bool = False
    enable_large_chunks: bool = False
    max_document_chars: int = 536_870_912
    max_chunks_per_doc_batch: int = 1000

    # -------------------------------------------------------------- Contextual RAG
    # Off by default: it costs one LLM call per chunk at index time.
    enable_contextual_rag: bool = False
    use_document_summary: bool = True
    use_chunk_summary: bool = True
    average_summary_embeddings: bool = False
    contextual_rag_max_context_tokens: int = 100
    max_tokens_for_full_inclusion: int = 4096
    contextual_rag_llm_timeout_s: int = 180
    contextual_rag_max_workers: int = 128

    # ------------------------------------------------------- Extraction / images
    image_extraction_enabled: bool = True
    # Only takes effect when an LLM is configured.
    image_summarization_enabled: bool = True
    image_summarization_timeout_s: int = 300
    # None means use the module defaults. Override to steer
    # summaries toward your own corpus (schematics, screenshots, slides).
    image_summarization_system_prompt: str | None = None
    image_summarization_user_prompt: str | None = None
    max_image_workers: int = 16
    max_embedded_images_per_file: int = 500
    min_embedded_image_dimension_px: int = 16
    max_xlsx_cells_per_sheet: int = 10_000_000
    # PDF text extraction runs in a subprocess; this bounds a hung parse.
    pdf_text_extraction_timeout_s: float = 120.0
    parse_with_trafilatura: bool = False
    html_ignored_classes: list[str] = Field(default_factory=lambda: ["sidebar", "footer"])
    html_ignored_elements: list[str] = Field(
        default_factory=lambda: [
            "nav",
            "footer",
            "meta",
            "script",
            "style",
            "symbol",
            "aside",
        ]
    )
    html_link_strategy: str = "strip"

    # -------------------------------------------------------------------- Access
    # A document ingested without permissions is world-readable.
    # Set false to fail closed instead.
    default_document_public: bool = True

    # ----------------------------------------------------------------- Retrieval
    num_returned_hits: int = 50
    max_chunks_fed_to_chat: int = 25
    keyword_query_hybrid_alpha: float = 0.2
    # Query-variant weights for reciprocal rank fusion.
    llm_semantic_query_weight: float = 1.3
    llm_keyword_query_weight: float = 1.0
    llm_non_custom_query_weight: float = 0.7
    original_query_weight: float = 0.5
    rrf_k: int = 50
    selection_token_budget_multiplier: int = 2
    max_chunks_for_relevance: int = 3
    full_doc_num_chunks_around: int = 5
    metadata_token_estimate: int = 75
    max_selected_sections: int = 10
    secondary_llm_flow_timeout_s: int = 60
    query_expansion_enabled: bool = True
    section_selection_enabled: bool = True
    # Off by default: a second LLM round trip per selected section.
    section_expansion_enabled: bool = False

    # -------------------------------------------------------------------- Answer
    # Three is enough for search-then-answer with one retry and bounds the cost
    # of a runaway loop.
    max_llm_cycles: int = 3
    stop_stream_pat: str | None = None

    # A Gemini model id, e.g. gemini-2.5-pro. Unset, answering is unavailable
    # and the LLM-assisted retrieval steps switch off (unless llm_fast_model
    # is set).
    llm_model: str | None = None
    # The model for everything except the answer itself: query expansion,
    # section selection, image summaries, contextual RAG. Mechanical work that
    # a cheaper, faster model does well, and that runs before the user sees
    # anything. Unset, those use llm_model.
    llm_fast_model: str | None = None
    # Some Gemini models are served only from "global". None uses vertex_location.
    llm_location: str | None = None
    llm_temperature: float = 0.0
    llm_max_input_tokens: int = 128_000

    # ------------------------------------------------------------- Store and API
    document_store_url: str = "sqlite:///brain.db"
    api_host: str = "0.0.0.0"
    api_port: int = 8100
    api_key: str | None = None

    def hybrid_fusion_weights(self) -> list[float]:
        """Fusion weights, positionally matched to the subquery list.

        Kept next to the subquery choice so the two cannot drift apart. The
        OpenSearch normalization processor rejects weights that do not sum to 1.
        """
        if self.hybrid_subquery_config is HybridSubqueryConfig.TITLE_AND_CONTENT_VECTOR:
            # Title is already included in the content text, so its own vector
            # gets only a small share.
            return [0.1, 0.45, 0.45]
        return [0.5, 0.5]


_settings: BrainSettings | None = None


def get_settings() -> BrainSettings:
    """Process-wide settings, read from the environment once."""
    global _settings
    if _settings is None:
        _settings = BrainSettings()
    return _settings


def set_settings(settings: BrainSettings) -> None:
    """Override the cached settings. For tests and for embedding hosts."""
    global _settings
    _settings = settings
