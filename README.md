# brain

Document ingest, hybrid retrieval, and cited answering, as a library.

You hand it `Document` objects. It chunks them, embeds them, indexes them in
OpenSearch, retrieves with hybrid search, and streams answers with inline
citations. How those documents are produced is your business: brain has no
connectors, because the platform calling it already has them.

Both models run on Vertex AI: Gemini for answering, and gemini-embedding-001
for vectors. There are no model weights in this repository.

## Status

Verified end to end against OpenSearch with the earlier local embedding model.
The move to Vertex is covered by unit tests against Vertex's documented request
and response shapes, but has not yet been run against a real Vertex project.
Not yet run against a production corpus, and not yet deployed anywhere.

| | |
|---|---|
| Source | 16,400 lines across 80 modules |
| Tests | 11,500 lines: 836 unit, 13 external, 10 end-to-end |
| Architecture contracts | 5, enforced in CI-able form |

The end-to-end suite runs against real OpenSearch and real Vertex embeddings.
Its retrieval cases use questions that share almost no vocabulary with the
document that should answer them, so passing means the vectors are doing
semantic work rather than keyword matching.

## What is here

| Package | Does | Tests |
|---|---|---|
| `models` | The shared types: documents, chunks, search results, access | 38 |
| `text` | Tokenizer, stopwords, chunk enrichment and its inverse | — |
| `extraction` | PDF, Office, HTML, CSV bytes to a `Document` | 126 |
| `chunking` | 512-token chunks, including a streaming table path | 102 |
| `vertex` | Vertex AI credentials, endpoints, and retries, shared by both models | 16 |
| `embedding` | gemini-embedding-001, batched under Vertex's request limits | 38 |
| `index` | OpenSearch schema, hybrid query, writes | 55 |
| `store` | The dedupe gates, in memory and in SQLite | 45 |
| `llm` | Provider boundary, with a Gemini adapter | 40 |
| `ingest` | The pipeline | 30 |
| `retrieval` | Multi-query search, rank fusion, narrowing | 147 |
| `answer` | The answer loop and the streaming citation processor | 169 |
| `api` | Thin FastAPI service | 16 |
| `facade` | The whole pipeline wired together | 14 |

**Deliberately out of scope:** connectors, Celery, Redis, MinIO,
multi-tenancy, a secondary-index rebuild path, reranking, Vespa, deep research,
web search, image generation, and Slack. Auth, connector scheduling, and the UI stay in the calling platform.

## Quick start

brain authenticates to Vertex with Application Default Credentials. Locally:

```bash
gcloud auth application-default login
```

The container needs that file mounted. Keep the mount in a
`docker-compose.override.yml`, which is gitignored, since it points at your own
credentials:

```yaml
services:
  brain-api:
    volumes:
      - ${HOME}/.config/gcloud/application_default_credentials.json:/run/secrets/adc.json:ro
```

Copy `.env.example` to `.env`, set `BRAIN_VERTEX_PROJECT`, then:

```bash
docker compose up -d
```

That starts OpenSearch and the brain API, on host ports 9201 and 8100. In a
deployment, mount a service account key instead of your own login, or run where
the platform supplies an identity.

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
visible to every caller. If your corpus is not uniformly
readable, set `BRAIN_DEFAULT_DOCUMENT_PUBLIC=false` and supply permissions at
ingest. Access is enforced at query time from strings stored on each chunk.

**Permission changes take effect on re-ingest.** Ingesting a document again
with new `external_access` updates who can see it, even when its content is
unchanged and it is otherwise skipped: the access strings on its chunks are
patched in place, without re-embedding. A source integration therefore has to
re-send a document whenever its sharing changes, not only when it is edited.

**Re-ingest is cheap, and that cuts both ways.** Two gates skip unchanged
documents: a timestamp, then a content hash. Both compare a document to itself,
so after a chunk-size or embedding-model change every document still looks
current. Pass `force=True` to rebuild.

**Changing the embedding model or dimension means re-indexing.** Vectors from
different models, or truncated to different sizes, are not comparable, and the
index mapping fixes the dimension. Re-ingest everything with `force=True` into a
fresh index.

**Ingest is billed per token.** Every chunk is an embedding call to Vertex, so a
bulk ingest runs into quota. Embedding batches stay under Vertex's per-request
limits (250 texts, 20,000 tokens), and indexing waits out rate limits rather
than failing.

**Images need an LLM.** An image section gets its text from a Gemini summary.
With no `BRAIN_LLM_MODEL` that text is empty, and an empty section is skipped
because there is nothing to embed or match. Surrounding body text is unaffected.

**Answering needs a model; ingest and search do not.** With no LLM configured,
query expansion and relevance narrowing switch off and retrieval still works,
which is also how you benchmark retrieval on its own.

## Configuration

Everything lives in `BrainSettings`, populated from `BRAIN_`-prefixed environment
variables and read exactly once. No module reads the environment on its own.

The settings you will touch first:

| Variable | Default | |
|---|---|---|
| `BRAIN_VERTEX_PROJECT` | your credentials' project | |
| `BRAIN_VERTEX_LOCATION` | `us-central1` | |
| `BRAIN_LLM_MODEL` | unset | A Gemini model id, e.g. `gemini-2.5-pro`. Unset turns answering off. |
| `BRAIN_LLM_LOCATION` | the Vertex location | Some Gemini models are served only from `global`. |
| `BRAIN_EMBEDDING_MODEL_NAME` | `gemini-embedding-001` | |
| `BRAIN_EMBEDDING_DIM` | `768` | Up to 3072. Changing it requires a reindex. |

The retrieval defaults are the ones it was benchmarked with: 512-token chunks,
no overlap, hybrid weights split evenly between vector and keyword, 500
candidates, 50 hits retrieved, 25 sections reaching the model. That benchmark
used the earlier local embedding model, so re-check them with
gemini-embedding-001.

Chunk sizes are counted with tiktoken, because Gemini's tokenizer is not
published. The counts are estimates, which costs little: chunks are 512 tokens
and the model accepts 2048.

See `.env.example`.

## Development

```bash
uv sync
uv run pytest -m "not external and not e2e"
uv run ruff check src tests
uv run lint-imports
```

Supported on Python 3.13 and 3.14. Dependencies are version ranges rather than
pins, so brain can share an environment with the application that hosts it;
the suite has been run against the newest and the oldest versions the ranges
allow.

The unit suite runs offline: Vertex is faked at the HTTP layer. `external`
needs OpenSearch; `e2e` needs OpenSearch and Vertex credentials, and skips
cleanly without either. Its answer test also needs `BRAIN_LLM_MODEL`.

`lint-imports` enforces the layering. It is worth keeping in CI: the
contracts make import cycles between indexing, retrieval, and the document index
impossible rather than merely discouraged.

## Next steps

Roughly in the order they matter:

1. **Run it against Vertex, then a real corpus.** Run the e2e suite against a
   real project first, then ingest a few thousand real documents and check
   retrieval quality before trusting it. The retrieval defaults were tuned with
   a different embedding model, so expect to revisit them.
2. **Wire up identity.** The platform mints a short-lived token; brain maps it to
   an `AccessScope`. Until then the API is a shared key and every caller looks the
   same, which makes the per-document permissions decorative.
3. **Make ingest asynchronous.** It is synchronous today, so a large batch holds
   an HTTP request open. The pipeline is already batched internally; it needs a
   queue in front of it.
4. **Decide about reranking.** brain uses rank fusion plus LLM narrowing rather
   than a cross-encoder. Whether a reranker beats it on your corpus is an
   experiment worth running.
5. **Observability.** There are no metrics or tracing yet. Something
   should be added before this runs unattended.
6. **CI.** The commands above should run on every push.

## License

MIT. See `LICENSE`.
