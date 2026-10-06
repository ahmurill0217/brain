# Ingestion

How documents get from Darwin's sources into brain. This is a design outline
for the Darwin integration. Nothing here is built yet, apart from brain's own
ingest pipeline.

## The contract

brain does not know or care where a document comes from. It ingests a list of
`Document`s, and each one carries:

- **id:** stable per source, e.g. `drive:<file id>` or `scout:<article id>`.
- **sections:** the text, plus a link for each part so citations can point
  back to it.
- **doc_updated_at:** when the source last changed it.
- **external_access:** who may see it. Darwin's Teams, Departments and Tags
  map to brain group IDs.
- **metadata:** optional, usable as search filters.

Drive files go through `brain.extract`, which reads PDF, DOCX, XLSX and PPTX,
before ingest.

Everything after that is shared:

- chunking and embedding;
- writing to the index;
- deduplication;
- propagating permission changes.

So each source needs only a thin **adapter** in Darwin that turns its records
into `Document`s. No source needs a change in brain.

**Retries are cheap.** brain skips a document whose content hash and
updated-at time are unchanged. It does still apply a permission change, in
place, without re-embedding. So a retried or duplicated task costs almost
nothing.

## Two kinds of source

### Data Darwin owns: push on change

This covers Postgres-backed apps such as SCOUT's scraped pages, ClickUp
mirrors and Fathom notes.

1. When a record is created or updated, Darwin queues an **index task** after
   the database transaction commits (`transaction.on_commit`). A deleted
   record queues a **delete task**.
2. The task loads the record, renders it as text with a link back to it in
   Darwin, and calls `ingest`, or `delete`.
3. A nightly **reconcile** job compares brain's document records with the
   source table. It queues tasks for anything missing, stale, or deleted at
   the source. This catches tasks that were lost or failed for good.

### External systems: pull on a schedule

This covers Google Drive, and anything else with an API rather than a table
in Darwin.

1. A scheduled job asks the source for changes since its last checkpoint.
   Drive's Changes API supports exactly this with a page token.
2. It queues one task per changed or removed file. Each task downloads the
   file, extracts it, maps its sharing settings to `external_access`, and
   ingests it.
3. The checkpoint advances only after its tasks are queued, so a crashed run
   repeats work rather than skipping it.

**Decide early: which identity reads Drive.** Darwin stores each user's own
Drive tokens (`DriveCredential`), captured at sign-in. Crawling company Drive
through personal tokens is fragile:

- each person sees a different slice of Drive;
- coverage breaks when someone leaves;
- one person's revoked consent silently removes documents.

A service account added to the shared drives, or one granted domain-wide
read access, is the usual approach. Either way, brain still enforces each
file's own sharing settings through `external_access`. This is a Darwin
integration decision, but it decides what brain can index.

**Gmail is excluded.** Barkley's rules forbid storing Gmail content, and
indexing it would store it.

## What does not belong in brain

RAG is for text people read: articles, notes, documents, policies. Questions
that are really database queries should go to SQL. For example: "how many
competitors raised a Series B this year?" or "which ClickUp tasks are
overdue?". Embedding those rows gives vague, partial answers.

For SCOUT, the scraped page text goes into brain. The structured fields
(company, funding round, date) stay in Postgres, used as search filters or
behind a separate SQL tool the answering model can call.

## Shared infrastructure

This mirrors what Barkley already does in Darwin's Terraform (`scheduler.tf`):

- Cloud Scheduler starts a Cloud Run job;
- the job queues one Cloud Task per item on a rate-limited queue;
- a daily batch job is the fail-safe for stragglers.

| Piece | Choice | Why |
|---|---|---|
| Ingest queue | **One** Cloud Tasks queue for every source | Every source shares the same limit, the Vertex embedding quota. One queue enforces it once, so a single source's backfill cannot starve the others |
| Queue rate | `max_dispatches_per_second` and `max_concurrent_dispatches` set below the Vertex embedding quota | Rate-limited tasks retry with backoff anyway, but sustained 429s waste task attempts and stall the queue. Set from the scale test's measured throughput (see [retrieval-benchmarks.md](retrieval-benchmarks.md)) |
| Task size | A small batch of records (tens), or one large file | Batches amortize the embedding round trip; one file per task keeps a huge PDF from timing out a whole batch |
| Retries | `max_attempts` a handful, with backoff | Safe because ingest is idempotent |
| Worker | Darwin's existing Cloud Run worker | Extraction and embedding are I/O-bound, and the worker already serves Cloud Tasks |
| Schedules | Cloud Scheduler: change polling per source, plus a nightly reconcile | Same as Barkley's sync and assess jobs |
| Concurrency | A per-document lock in the Postgres document store | brain's default lock is single-writer only. Two tasks for one document must not interleave, and the Django port's store needs a real row lock |

### Backfills

A backfill is the same index task, queued in bulk. Onboarding a new source, or
re-indexing after an embedding-model change, means enqueuing every ID, and
the queue's rate limit paces it. No separate bulk path is needed.

At 500k documents:

- **Time:** one process embeds about 12,400 tokens/s, as measured on the
  171K-document trec-covid run (0 failures, 5 throttled requests out of
  3,442). That is ~45–56 hours for 500k typical documents. Parallel workers
  divide it until the Vertex quota caps them; see
  [retrieval-benchmarks.md](retrieval-benchmarks.md#ingestion-at-scale).
- **Cost:** embedding is roughly $300–400 one-time, at $0.15 per million
  tokens.
- **Images:** image summarization is on by default, so each image in a
  document is one extra Flash call. Count the images in a sample before
  backfilling an image-heavy source.

## Open decisions

1. **Drive identity:** a service account on shared drives, or domain-wide
   delegation.
2. **Which sources first,** and which of their fields are text for brain and
   which are filters or SQL.
3. **Group mapping:** how Darwin's Teams, Departments and Tags become
   `external_access` groups, per source.
4. **The queue's rate limits,** from the scale test.
