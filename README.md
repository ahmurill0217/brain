# brain

Document ingest, hybrid retrieval, and cited answering, extracted from
[Onyx](https://github.com/onyx-dot-app/onyx-foss) and rebuilt as a library.

You hand it `Document` objects. It chunks them, embeds them, indexes them in
OpenSearch, retrieves with hybrid search, and streams answers with inline
citations. How those documents are produced is your business: brain has no
connectors, because the platform calling it already has them.

## Status

Working end to end and verified against a real stack. Not yet run against a
production corpus, and not yet deployed anywhere.

| | |
|---|---|
| Source | 16,300 lines across 94 modules |
| Tests | 14,900 lines, 873 unit + 9 end-to-end |
| Architecture contracts | 4, enforced in CI-able form |
| Onyx imports remaining | none |

The end-to-end suite runs against real OpenSearch and real embeddings. Its
retrieval cases use questions that share almost no vocabulary with the document
that should answer them, so passing means the vectors are doing semantic work
rather than keyword matching.

## What is here

| Package | Does | Tests |
|---|---|---|
| `models` | The shared types: documents, chunks, search results, access | 38 |
| `text` | Tokenizer, stopwords, chunk enrichment and its inverse | — |
| `extraction` | PDF, Office, HTML, CSV bytes to a `Document` | 126 |
| `chunking` | 512-token chunks, including a streaming table path | 105 |
| `embedding` | Client for the bundled model server | 39 |
| `index` | OpenSearch schema, hybrid query, writes | 55 |
| `store` | The dedupe gates, in memory and in SQLite | 45 |
| `llm` | Provider boundary, with a litellm adapter | 30 |
| `ingest` | The pipeline | 30 |
| `retrieval` | Multi-query search, rank fusion, narrowing | 153 |
| `answer` | The answer loop and the streaming citation processor | 225 |
| `api` | Thin FastAPI service | 16 |
| `facade` | The whole pipeline wired together | 11 |

Plus `model_server/`, a separate 550-line service that holds the embedding model
and is the only thing in the repository that needs PyTorch.

**What was deliberately left behind:** connectors, Celery, Redis, MinIO,
multi-tenancy, the secondary-index rebuild path, reranking (Onyx's current
retrieval has none), Vespa, deep research, web search, image generation, and
Slack. Auth, connector scheduling, and the UI stay in the calling platform.

## Quick start

```bash
docker compose up -d
```

OpenSearch, the embedding model server, and the brain API. The model server
bakes its weights into the image, so after the first build it starts offline.
Host ports default to 9201, 9100, and 8100 so the stack can run alongside an
existing Onyx.

In process:

```python
from brain import AccessScope, Brain, Document, TextSection

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

for event in brain.answer("what did we decide about pricing?",
                          access=AccessScope(user_email="you@example.com")):
    print(event)
```

Over HTTP:

```python
requests.post(f"{BRAIN_URL}/v1/ingest",
              json={"documents": [doc.model_dump(mode="json")]})
```

Endpoints: `/v1/ingest`, `/v1/extract`, `/v1/documents/delete`, `/v1/search`,
`/v1/answer` (server-sent events), `/v1/health`.

## Things that will bite you otherwise

**Documents are public by default.** A `Document` with no `external_access` is
visible to every caller, which matches Onyx. If your corpus is not uniformly
readable, set `BRAIN_DEFAULT_DOCUMENT_PUBLIC=false` and supply permissions at
ingest. Access is enforced at query time from strings stored on each chunk, so a
document indexed with the wrong permissions stays wrong until it is re-indexed.

**Re-ingest is cheap, and that cuts both ways.** Two gates skip unchanged
documents: a timestamp, then a content hash. Both compare a document to itself,
so after a chunk-size or embedding-model change every document still looks
current. Pass `force=True` to rebuild.

**Images without a vision model are dropped.** An image section gets its text
from a model summary. With no LLM configured that text is empty, and an empty
section is skipped because there is nothing to embed or match. Surrounding body
text is unaffected.

**Answering needs a model; ingest and search do not.** With no LLM configured,
query expansion and relevance narrowing switch off and retrieval still works,
which is also how you benchmark retrieval on its own.

## Configuration

Everything lives in `BrainSettings`, populated from `BRAIN_`-prefixed environment
variables and read exactly once. No module reads the environment on its own.
Defaults match the Onyx configuration this was benchmarked against: 512-token
chunks, no overlap, hybrid weights split evenly between vector and keyword, 500
candidates, 50 hits retrieved, 25 sections reaching the model.

See `.env.example`.

## Development

```bash
uv sync
uv run pytest -m "not external and not e2e"
uv run ruff check src tests
uv run lint-imports
```

The unit suite runs offline. `external` needs OpenSearch and the model server;
`e2e` needs the full compose stack and skips cleanly without it.

`lint-imports` enforces the layering. It is worth keeping in CI: Onyx has real
import cycles between indexing, retrieval, and the document index, and these
contracts are what make inheriting them impossible rather than merely discouraged.

## Next steps

Roughly in the order they matter:

1. **Point it at a real corpus.** Everything so far is synthetic or a handful of
   papers. Ingest a few thousand real documents and compare retrieval against the
   same corpus in Onyx before trusting it.
2. **Wire up identity.** The platform mints a short-lived token; brain maps it to
   an `AccessScope`. Until then the API is a shared key and every caller looks the
   same, which makes the per-document permissions decorative.
3. **Make ingest asynchronous.** It is synchronous today, so a large batch holds
   an HTTP request open. The pipeline is already batched internally; it needs a
   queue in front of it.
4. **Decide about reranking.** Onyx dropped the cross-encoder in favor of rank
   fusion plus LLM narrowing. That is what brain does. Whether a reranker beats it
   on your corpus is an experiment worth running.
5. **Observability.** All the Onyx metrics and tracing were stripped. Something
   should replace them before this runs unattended.
6. **CI.** The commands above should run on every push.

## License

MIT. Substantial portions derive from Onyx, also MIT. `LICENSE` carries both
copyright lines, and `NOTICE.md` maps each ported module to its Onyx source.

Ported modules also carry a one-line `# Derived from onyx/...` comment. That is
not a licensing requirement, which the `LICENSE` file already satisfies. It is
kept because it is a useful pointer: when brain's behavior needs comparing
against upstream, the comment says exactly which file to open.
