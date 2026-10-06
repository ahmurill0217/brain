"""Shared wiring for the retrieval benchmarks.

Brain is driven in process rather than over HTTP, for three reasons: each
dataset gets its own OpenSearch index instead of sharing `brain_chunks`, no
LLM is configured so the measurement is pure retrieval, and it is how a
platform embedding brain would actually call it.

OpenSearch comes from `docker compose up -d`; embeddings come from Vertex, with
the project and location read from `.env` like everything else. Embedding a
corpus is billed, so the indexer says what it is about to embed before it does.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from brain import Brain, BrainSettings
from brain.store.sqlite import SQLiteDocumentStore

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def _dotenv() -> dict[str, str]:
    path = ROOT.parent / ".env"
    if not path.exists():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return {k.strip(): v.strip() for k, v in pairs if not k.lstrip().startswith("#")}


def settings_for(dataset: str, *, answering: bool = False) -> BrainSettings:
    """Settings for one dataset's index.

    No LLM is configured on purpose. Query expansion and section selection
    both need one, so leaving it out is what makes this a measurement of
    retrieval rather than of retrieval plus a model's judgement. The two flags
    are also set explicitly, so the intent survives a model being set in .env.

    `answering` keeps the models from .env and every LLM step at its default,
    for the end-to-end answer benchmark: that one measures brain as deployed.

    The index and store are named after the embedding model. Vectors from two
    models are not comparable, and the store's dedupe would otherwise skip
    every document already indexed under the old model.
    """
    env = _dotenv()
    probe = BrainSettings()
    model = re.sub(r"[^a-z0-9]+", "_", probe.embedding_model_name.lower()).strip("_")
    overrides: dict = {
        "opensearch_index_name": f"beir_{dataset}_{model}",
        # These outlive this index and would be imposed on anything else
        # sharing the cluster, including the running brain_chunks index.
        "opensearch_set_cluster_settings": False,
        "image_summarization_enabled": False,
    }
    if not answering:
        overrides |= {
            "llm_model": None,
            "llm_fast_model": None,
            "query_expansion_enabled": False,
            "section_selection_enabled": False,
        }
    # .env describes the compose network; from the host, use the mapped port.
    if "BRAIN_OPENSEARCH_PORT" not in os.environ and env.get("OPENSEARCH_HOST_PORT"):
        overrides["opensearch_port"] = int(env["OPENSEARCH_HOST_PORT"])
    if "BRAIN_OPENSEARCH_PASSWORD" not in os.environ and env.get("OPENSEARCH_ADMIN_PASSWORD"):
        overrides["opensearch_password"] = env["OPENSEARCH_ADMIN_PASSWORD"]
    return BrainSettings(**overrides)


def brain_for(dataset: str, *, answering: bool = False) -> Brain:
    """A Brain against this dataset's index, with a store that survives runs.

    The store is on disk rather than in memory so a re-run of the indexer
    skips documents it has already embedded, and does not pay for them twice.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    settings = settings_for(dataset, answering=answering)
    store = SQLiteDocumentStore(f"sqlite:///{DATA / f'{settings.opensearch_index_name}.db'}")
    return Brain.from_settings(settings, document_store=store)


def require_stack(brain: Brain) -> None:
    """Fail early and clearly rather than deep inside a batch."""
    if not brain.index.client.ping():
        raise SystemExit("OpenSearch is not reachable; try `docker compose up -d`")
    if not brain.embedder.healthy():
        raise SystemExit(
            "Vertex credentials are not usable; try `gcloud auth application-default login`"
        )
