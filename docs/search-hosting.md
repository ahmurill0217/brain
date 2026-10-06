# Search hosting

Where brain's OpenSearch index runs once brain is part of Darwin, what it
costs, and when it can be down.

**Decision:** one OpenSearch node on a Compute Engine VM, in Darwin's VPC.
It keeps the exact search the HR eval passes on, so accuracy is unchanged.

Prices are list prices for us-central1, checked October 2026. Re-check them
before committing to a term.

## Sizing

Estimated from the HR corpus, which averaged about 11 chunks per document.
Measure Darwin's real document types before relying on these.

| At 500k documents | Size |
|---|---|
| Chunks | ~6–8M |
| Content vectors (768 × float32 each) | ~20–25 GB |
| HNSW graph (m=32) | ~2 GB |
| Text, keyword index, stored source | ~15–25 GB on disk |

Vector search is fast only while the vectors sit in memory. brain uses the
Lucene engine, which serves vectors from the OS page cache rather than the
JVM heap. So the machine needs roughly the vector size in free RAM on top of
an ~8 GB heap.

Two settings change this:

- **Title vectors** are off by default (`BrainSettings.uses_title_vector`).
  Turning them on doubles the vector figures.
- **Quantized vectors** would cut vector memory about 4×. Lucene supports
  scalar quantization to bytes. brain does not use it yet, and it must pass
  the HR eval before it is turned on.

You won't start at 500k documents. Resizing the VM later is one stop and
start (see Downtime).

## The VM

| Item | Spec | Monthly |
|---|---|---|
| VM | e2-highmem-4: 4 vCPU, 32 GB, $0.1808/hr | $131.98 |
| Data disk | 200 GB balanced PD, $0.10/GB | $20.00 |
| Boot disk | 20 GB balanced PD | $2.00 |
| Backups | OpenSearch snapshots to a GCS bucket, ~50 GB at ~$0.02/GB | ~$1 |
| External IP | None: reached on the private network only | $0 |
| OpenSearch | Apache 2.0 | $0 |
| **Total, on demand** | | **~$155** |

The only discount on E2 is a committed-use discount (CUD); it gets no
sustained-use discount.

| Term | VM | Total |
|---|---|---|
| On demand | $131.98 | ~$155 |
| 1-year CUD | $83.15 | ~$106 |
| 3-year CUD | $59.39 | ~$83 |

**Run on demand until the document count settles, then commit.**

**Smaller option:** e2-highmem-2 (2 vCPU, 16 GB), about $66 for the VM. It
is enough if quantization passes the eval, or while the corpus is well
under 500k documents.

**Spot VMs:** not suitable. Google can reclaim them at any time, which is
wrong for an always-on, stateful server.

### Setup notes

- **Network:** put the VM in Darwin's VPC (`evolve-<env>-vpc`) with no
  external IP. Darwin's Cloud Run services already route private ranges
  through `vpc_access` (`PRIVATE_RANGES_ONLY`), so they can reach it without
  further network changes.
- **Terraform:** manage it in Darwin's Terraform next to Cloud SQL and
  Memorystore, so it is created, sized and destroyed with the rest of the
  environment.
- **Security:** turn on OpenSearch's security plugin with TLS and its own
  user. brain already takes `BRAIN_OPENSEARCH_USERNAME` and
  `BRAIN_OPENSEARCH_PASSWORD`; keep them in Secret Manager like Darwin's
  other secrets.
- **Snapshots:** take scheduled snapshots to a GCS bucket with the
  `repository-gcs` plugin. Snapshots are incremental, so a nightly one is
  cheap.
- **Auto-restart:** keep the Compute Engine defaults
  (`automaticRestart: true`, `onHostMaintenance: MIGRATE`). Then a crash
  restarts the VM on its own, and Google host maintenance moves it live
  without downtime.
- **Monitoring:** install the Ops Agent for disk, memory and JVM metrics,
  and alert on disk above 75%. OpenSearch stops writing to an index once
  disk passes its flood-stage watermark (95% by default).

## Downtime

**Adding documents: none.** Indexing runs while search keeps serving, and a
new document is searchable about a second after it is written. A large
first-time backfill can slow searches on a single node, so run it
overnight.

**Darwin and brain deploys: none.** The index lives on its own VM and disk,
and deploying code does not touch it.

The exceptions are changes to how the index is built:

- the embedding model;
- the text analyzer;
- title vectors on or off;
- quantization on or off.

Each needs a full re-index, which is a planned job, not a restart.

**What can actually take it down:**

| Event | How often | Downtime | Schedulable |
|---|---|---|---|
| OpenSearch upgrade or OS patching | A few times a year | 1–3 min restart | Yes, off-hours |
| Resizing the VM | Rarely | ~2–5 min stop and start | Yes |
| Google host maintenance | Occasionally | None (live migration) | Not needed |
| VM crash or out of memory | Rare | ~2–3 min auto-restart; data is on the PD | No |
| Zone outage or disk loss | Very rare | An hour or more: restore the last snapshot, re-ingest changes since | No |

While the server is down, answering fails with an error. Ingest calls fail
too; queue them through Cloud Tasks with retries so documents are retried
rather than lost.

Only the last two rows are unplanned. A 3-node cluster (~3× the cost)
removes them. Add one only once search becomes business-critical.

## Why not inside Darwin with no VM

Darwin runs on fully managed, stateless services:

- **Cloud Run:** the web service and the Cloud Tasks worker, 1 vCPU and
  1.25 GB each; the worker scales to zero.
- **Cloud SQL:** Postgres 18, `db-custom-2-8192`, regional high
  availability.
- **Memorystore:** Redis.

Nothing in Darwin runs on a VM or holds data on its own disk.

### OpenSearch inside a Cloud Run container

Not possible:

- Cloud Run instances have no persistent disk; their filesystem lives in
  memory and is gone on restart.
- Instances scale up, down and to zero, so each one would hold a different,
  partial index.
- Mounting GCS or Filestore under it does not help. Lucene is unreliable on
  network filesystems and needs a single writer.

OpenSearch is a long-running server that owns its disk, so it needs a VM, a
Kubernetes cluster, or a managed service.

### Replacing OpenSearch with pgvector in Darwin's Cloud SQL

This is the only option that adds no new server. It is not the default
because it would cost more than the VM and risk accuracy.

**Cost.** The vectors need to fit in the database's memory, and today's
database has 8 GB. Prod is regional HA, so every size increase is billed
twice. Enterprise rates are $0.0413 per vCPU-hour and $0.007 per GB-hour.

| Prod Cloud SQL (HA) | Monthly | Increase |
|---|---|---|
| Today: 2 vCPU / 8 GB | ~$202 | — |
| 4 vCPU / 16 GB (half-precision vectors) | ~$405 | +~$203 |
| 4 vCPU / 32 GB (full-precision vectors) | ~$568 | +~$366 |

Both increases exceed the whole VM, at ~$155.

**Contention.** Vector index builds and large backfills would compete with
Darwin's own transactional queries on the same database.

**Accuracy.** Postgres full-text ranking (`ts_rank`) is not BM25, and the
BM25 extensions are not available on Cloud SQL. Filtering by permissions
with an HNSW index also needs careful tuning (iterative scans) to keep
recall. Search would have to be rebuilt and re-validated against the HR
eval. It could work, but it is unproven, and we decided accuracy must not
drop.

brain's index is behind a swappable interface (`DocumentIndex`), so a
pgvector version can still be built later and run through the same eval
side by side. Revisit it only if the VM becomes a real operational burden.

### Other options considered

| Option | Monthly | Why not now |
|---|---|---|
| Managed OpenSearch on GCP (Aiven or similar) | ~$300–800+ | Same accuracy, no ops work, but 2–5× the cost |
| AWS OpenSearch Service | ~$150–700 | Another cloud: network latency, egress charges, separate auth |
| Vertex AI Vector Search | ~$100–600 per node | No built-in BM25, so brain would supply the keyword half itself; a large rewrite |
| 3-node OpenSearch cluster | ~$450–650 | Only buys protection from unplanned downtime; revisit later |

## Open items

1. Measure chunks per document on a sample of real Darwin document types,
   then pick the VM size.
2. Run the HR eval with quantized vectors. If it still passes 13/13, plan
   on 16 GB instead of 32 GB.
3. Decide how much unplanned downtime is acceptable. That decides when a
   3-node cluster is worth it.

## Sources

- [e2-highmem-4 pricing](https://gcloud-compute.com/e2-highmem-4.html)
- [Balanced persistent disk pricing](https://gcloud-compute.com/pd-balanced.html)
- [Compute Engine disk and image pricing](https://cloud.google.com/compute/disks-image-pricing)
- [Cloud SQL pricing](https://cloud.google.com/sql/pricing)
- Darwin's Terraform (`terraform/modules/*.tf`, `terraform/environments/prod/main.tf`)
