# brain

Document ingest, hybrid retrieval, and cited answering. Extracted from
[Onyx](https://github.com/onyx-dot-app/onyx-foss) (MIT), with the connectors,
Celery, Redis, multi-tenancy, and chat UI removed.

You hand it `Document` objects. It chunks them, embeds them, indexes them in
OpenSearch, retrieves with hybrid search, and streams answers with inline
citations. How those documents are produced is your business: brain has no
connectors, because your platform already has them.

## What is in the box

| Piece | What it does |
|---|---|
| `brain.extraction` | PDF, Office, HTML, CSV bytes to a `Document` |
| `brain.chunking` | 512-token chunks with title, metadata, and optional summaries folded in |
| `brain.embedding` | Client for the bundled model server (nomic-embed-text-v1, 768 dims) |
| `brain.index` | OpenSearch schema, hybrid query, and writes |
| `brain.ingest` | The pipeline, including the two-gate dedupe |
| `brain.retrieval` | Multi-query search, rank fusion, LLM section selection |
| `brain.answer` | The answer loop and the streaming citation processor |
| `brain.api` | Thin FastAPI service, if you would rather call it over HTTP |

## Quick start

```bash
docker compose up -d
```

That brings up OpenSearch, the embedding model server, and the brain API. The
model server's first build downloads the embedding model and bakes it into the
image, so later starts need no network.

In process:

```python
from brain import Brain, Document, TextSection, AccessScope

brain = Brain.from_settings()
brain.ensure_ready()

brain.ingest([
    Document(
        id="doc-1",
        source="gdrive",
        semantic_identifier="Q3 Planning",
        sections=[TextSection(text="...", link="https://docs.example/1")],
    )
])

for event in brain.answer("what did we decide about pricing?", access=AccessScope(user_email="you@example.com")):
    print(event)
```

Over HTTP, from Django:

```python
requests.post(f"{BRAIN_URL}/v1/ingest", json={"documents": [doc.model_dump(mode="json")]})
```

## Two things to know before you deploy

**Documents are public by default.** A `Document` with no `external_access` is
visible to every caller, which matches Onyx's behavior. If your corpus is not
uniformly readable, set `BRAIN_DEFAULT_DOCUMENT_PUBLIC=false` and supply
permissions at ingest time. Access is enforced at query time from the ACL
strings stored on each chunk, so a document indexed with the wrong permissions
stays wrong until it is re-indexed.

**Configuration is read once.** Everything lives in `BrainSettings`, populated
from `BRAIN_`-prefixed environment variables. No module reads the environment on
its own.

## Development

```bash
uv sync
uv run pytest -m "not external and not e2e"
uv run ruff check src tests
uv run lint-imports
```

Tests marked `external` need a running OpenSearch and model server; `e2e` needs
the full compose stack.

## License

MIT. Substantial portions derive from Onyx, also MIT. `NOTICE.md` maps each
ported module to its Onyx source file.
