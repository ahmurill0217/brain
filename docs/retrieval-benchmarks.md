# Retrieval benchmarks

How well brain's search finds the right documents, measured against public
datasets with human relevance judgements. The scripts and how to re-run them
are in [benchmarks/](../benchmarks/README.md).

## What is measured

**Search only.** No LLM is involved: query rewriting and section selection are
off. For each question, brain runs its hybrid search (keyword plus vector) and
the ranked documents are scored against the documents that human judges marked
relevant to it.

This isolates retrieval. If the right passage is never retrieved, no answering
model can cite it, so this is the foundation the answer quality rests on.

Answer and citation quality are measured separately: on scifact, in
[Answers and citations](#answers-and-citations), and on our HR corpus (see
[Related](#related)).

Datasets come from BEIR, the standard public benchmark for retrieval. Using
it means the numbers can be compared with published baselines, not only with
our own earlier runs.

| Dataset | Corpus | Test questions | Why |
|---|---|---|---|
| scifact | 5,183 scientific abstracts | 300 | Quick and well-understood; good for tracking changes over time |
| nfcorpus | 3,633 medical documents | 323 | Medical domain, closest to a biotech corpus, and hard |
| trec-covid | 171,331 COVID-19 papers (CORD-19) | 50 | Scale: ingestion at volume, and ranking with 30× more distractor documents |

## Results

| Metric | scifact, Sep 16 2026 (nomic-embed-text-v1, local) | **scifact, Oct 5 2026** | **nfcorpus, Oct 5 2026** | **trec-covid, Oct 5 2026** |
|---|---|---|---|---|
| nDCG@10 | 0.7260 | **0.8217** | **0.4109** | **0.8455** |
| Recall@10 | 0.8556 | **0.9489** | 0.1992 | 0.0237 |
| Recall@100 | 0.9483 | **0.9933** | 0.4002 | 0.1712 |
| MRR@10 | 0.6888 | **0.7859** | 0.6185 | **0.9500** |
| Search latency p50 / p95 | 77 / 97 ms | 257 / 316 ms | 250 / 317 ms | 328 / 781 ms |
| Documents (chunks) in the index | 5,183 | 5,183 (5,629) | 3,633 (3,959) | 171,331 (182,220) |
| Indexing failures | — | 0 | 0 | 0 |

The October runs all use gemini-embedding-001.

**Published baselines** (BM25, keyword-only, from the BEIR paper; nDCG@10):

| | scifact | nfcorpus | trec-covid |
|---|---|---|---|
| BM25 | ~0.67 | ~0.33 | ~0.66 |
| brain | **0.82** | **0.41** | **0.85** |

### What the metrics mean

| Metric | Question it answers |
|---|---|
| **nDCG@10** | How good is the top-10 ranking, giving more credit for relevant documents near the top? The headline number in BEIR papers. 1.0 is perfect. |
| **Recall@10** | What share of the relevant documents appear in the top 10? |
| **Recall@100** | Is the relevant document anywhere in the candidate pool? Low recall here cannot be fixed by better ranking. |
| **MRR@10** | How high does the first relevant document rank? 1.0 means first, 0.5 means second, on average. |

### Reading the results

**scifact.** gemini-embedding-001 beats the old local embedder on every
quality metric:

- +0.096 nDCG@10.
- A relevant document is in the top 10 for 95% of questions (was 86%).
- It is in the top 100 for 99% of questions.

Chunk size (512 tokens), vector size (768), and the hybrid query and weights
were the same in both runs, so the embedding model is what changed.

**nfcorpus.** This dataset is hard by design, and 0.41 nDCG@10 is a solid
score. For reference, the BEIR paper's keyword-only (BM25) baseline is about
0.33. Its recall looks low because each question has dozens of relevant
documents, so a top 10 or top 100 can never hold them all. MRR@10 is 0.62,
so the first relevant document typically ranks first or second.

**trec-covid.** This is the scale test: 171K papers, 33× the scifact corpus.

- **nDCG@10 of 0.85** is well above the ~0.66 keyword-only baseline.
- **MRR@10 of 0.95** means the top result is relevant for nearly every
  question.

Ranking holds up with 171K documents competing for the top 10.

The recall figures are low by construction. Each question has hundreds of
documents judged relevant (a median of about 1,270 judged per question), so
100 results can hold only a fraction of them.

Its judgements were also collected in 2020 from the systems of that time. A
relevant paper none of them found is counted as not relevant, so the score
slightly understates a newer system.

**Latency.** Latency rose from about 80 ms to about 250 ms because each query
is now embedded by Vertex over the network instead of by a model on the same
machine. Next to a full answer, which takes 6–9 s, that is negligible.

At 182K chunks, the median search took about 80 ms longer than at 4–5K. The
p95 of 781 ms is from a 50-question run on a laptop's Docker OpenSearch with
a 2 GB heap, where the first queries also warm the cache. A sized VM (see
[search-hosting.md](search-hosting.md)) keeps the whole vector index in
memory.

### Configuration

| Setting | Value |
|---|---|
| Embedding model | gemini-embedding-001, 768 dimensions, `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types |
| Chunk size | 512 tokens |
| Hybrid query | Content vector plus keyword (title and content), fusion weights 0.5 / 0.5, min-max normalization |
| Title vector | Off (the default; see `BrainSettings.uses_title_vector`) |
| Vector index | OpenSearch Lucene HNSW, cosine, m=32, ef_construction=256 |
| Candidates per subquery | 500 |
| Results scored per question | 100 chunks |
| LLM | None (query rewriting and section selection off) |
| Access control | Every document indexed public, so permission filtering cannot affect the score |

## Cost and time

Vertex list price for gemini-embedding-001 is $0.15 per million input tokens.

| | scifact | nfcorpus | trec-covid |
|---|---|---|---|
| Tokens embedded at indexing | 1,660,491 | 1,244,550 | 40,303,177 |
| Indexing cost | ~$0.25 | ~$0.19 | ~$6.05 |
| Query embedding cost | < $0.01 | < $0.01 | < $0.01 |
| Indexing time | 289 s (17.9 docs/s) | 42 s (85.9 docs/s) | 3,257 s, 54 min (52.6 docs/s) |
| Evaluation time | 150 s | 216 s | 21 s |
| Chunks indexed | 5,629 | 3,959 | 182,220 |
| Index size on disk | 29 MB | 21 MB | 901 MB |

**Total cost for all three datasets: about $6.50.**

The token counts are upper bounds the indexer prints before embedding
anything. Re-running is free: documents already in the store are skipped.

The scifact indexing rate was about 3× slower than the other two. That run
most likely hit Vertex's embedding rate limit early and waited between
retries. The trec-covid run measured the steady rate, below.

## Ingestion at scale

trec-covid was indexed the way a backfill would run: batches of 200 documents
through `brain.ingest`, from one process with brain's default 8 embedding
threads.

| | trec-covid |
|---|---|
| Documents | 171,331, of which 171,331 indexed and 0 failed |
| Wall time | 54 min |
| Steady rate | 48–53 docs/s, flat from the first batch to the last |
| Token throughput | ~12,400 tokens/s (~740K tokens/min) |
| Vertex embedding requests | 3,442 |
| Rate-limited (429) responses | 5, all absorbed by retries |
| Other Vertex errors | 0 |

**Documents per second depends on document length.** trec-covid entries are
abstracts of about 235 tokens, so the steady rate is better read as tokens
per second. Projected to a Darwin backfill at this single-process rate:

| Corpus | Tokens (est.) | Time at ~12.4K tokens/s | Embedding cost |
|---|---|---|---|
| 500k short documents (~250 tokens) | ~125M | ~3 h | ~$19 |
| 500k typical documents (~4–5K tokens) | ~2–2.5B | ~45–56 h | ~$300–375 |

**Parallelism shortens the time.** Several Cloud Tasks workers running at
once divide the time, until the Vertex project's per-minute quota for
gemini-embedding-001 caps them. Five throttled requests out of 3,442 means
one process ran close to, but under, that quota.

The quota sets the ingest queue's rate limit (see
[ingestion.md](ingestion.md)). Check it in the console, and request an
increase before a large backfill.

**Index size scales with chunks.** About 4.9 KB per chunk on disk, which
covers the vector, the text, and the keyword index. 500k typical documents
at about 11 chunks each is roughly 5.5M chunks, or about 27 GB. That matches
the sizing in [search-hosting.md](search-hosting.md).

The September run used a model hosted on our own machine, so it had no
per-token cost but took about 18 minutes of CPU time to index scifact.

## Answers and citations

The searches above stop at the ranked list. This run asks brain's full
answer pipeline, as deployed, to judge 100 scifact claims:

- Gemini 2.5 Pro answers; Gemini 2.5 Flash does query rewriting and section
  selection.
- The model decides what to search for.
- It streams an answer with citations.

Each claim was asked as:

> Does the research support or contradict this claim? Start your answer with
> exactly one of SUPPORTED, CONTRADICTED, or NOT ENOUGH EVIDENCE, then explain
> with citations.

**Why scifact:** most claims have exactly one abstract that experts judged as
its evidence. That makes the core check strict: did brain cite that specific
abstract, out of 5,183?

Script: `benchmarks/answer_beir.py --limit 100`. Run on Oct 6 2026 against
the `beir_scifact_gemini_embedding_001` index.

| | Result |
|---|---|
| Answered without error | 100/100 |
| Cited at least one document | 99/100 |
| **Search retrieved the judged abstract** | **99/100** |
| **Answer cited the judged abstract** | **88/100** |
| Share of citations that are judged-relevant | 48.7% (a lower bound; see below) |
| **Verdict matches the expert label** (65 claims with a label) | **61/65 (93.8%)** |
| Verdicts on the 35 claims with no label | 15 supported, 10 contradicted, 10 not enough evidence |
| Citations per answer, median | 3 |
| Time to first answer text, median | 12.7 s (4 claims answered at once) |
| Cost | $2.63 for 100 claims, $0.026 per claim |
| Tokens (in / out) | Pro 454K / 141K; Flash 2.11M / 9K |

### Reading the results

**Retrieval is not the bottleneck.** The judged abstract reached the model
for 99 of 100 claims. The one exception (#437) was a complex claim the
search did not surface.

**Most citation misses are a strict-grading artifact.** In 11 of the 12
misses, the judged abstract was retrieved but the model cited other
abstracts on the same topic. For example, it backed "a deficiency of vitamin
B12 increases blood levels of homocysteine" with several studies, none of
them the single one the experts judged.

scifact judges only one abstract per claim, so these answers are likely
well-supported but count as misses. For the same reason, the 48.7% precision
is a floor: an unjudged citation is not necessarily irrelevant.

**Verdicts are strong.** 61 of 65 matched the expert label. All 4 wrong
verdicts cited the judged abstract: the model read the right evidence and
judged it differently. At least one label is arguable (#208, "CHEK2 is not
associated with breast cancer", labeled SUPPORTED).

**One real defect.** For claim #53 the answer stopped mid-sentence: no
citations and no error event. It read, in full:

> SUPPORTED
>
> Research indicates that the expression of Aldehyde Dehydrogenase 1 (ALDH1),
> a recognized marker for cancer stem cells, is linked to a poorer prognosis
> in individuals with breast cancer

The stream most likely ended early, either at the output-token limit or on
Vertex's side. brain passed the truncated text on as a normal answer. It was
1 in 100, but it is silent: a user would get an uncited, unfinished answer
with no sign anything went wrong. Fix it before the Darwin integration.

**Latency.** The median of 12.7 s to first text is higher than the HR eval's
6.2 s. Four claims ran at once, and research-style claims draw longer
reasoning and more searches than HR questions.

## Limits

These benchmarks deliberately leave several things out:

- **Answer quality at scale.** The answer benchmark runs on scifact's 5K
  abstracts, not on the 171K of trec-covid. trec-covid's questions have
  hundreds of relevant papers each, so citing one proves little.
- **Full production scale.** trec-covid is 171K documents. 500k is
  projected from it, not measured.
- **Our own documents.** These are public scientific and medical texts, not
  Darwin documents. They also contain none of the near-duplicate or outdated
  versions that real company documents often have.
- **Access control.** Every document is public here. Permission filtering is
  covered by brain's unit and OpenSearch integration tests instead.

## Related

- **End-to-end HR eval.** It runs the full pipeline, including answering and
  citations, on a real 7-document HR corpus (kept out of git).
  - Passed 13/13 after each change since the move to Vertex.
  - Most recent run, on Oct 5 2026: about $0.023 per question, with a median
    of 6.2 s to the first answer text.
- **Planned next.**
  1. Fix the truncated-answer defect: detect an early stream end and report
     it instead of passing it on as an answer.
  2. Re-run the 100 claims against scifact and trec-covid merged into one
     176K-document index (~$2.60, no new embedding): citation accuracy with
     34× more distractors.
  3. An "old version" trap built on the HR corpus (< $0.50).
- **Search hosting.** See [search-hosting.md](search-hosting.md).
