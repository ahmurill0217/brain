# benchmarks

Retrieval measured on its own, against human relevance judgements.

This is the layer `testcorpus/` cannot be. That suite asks whether the whole
pipeline produces a good answer, which is slow, non-deterministic, costs an LLM
call per question, and — worst of all — conflates retrieval quality with
generation quality, so a regression tells you nothing about where it came from.

Here there is no LLM at all. Query expansion and section selection are switched
off, so what is measured is the hybrid query and the embeddings. A number that
moves points at one of those.

## Running it

```bash
docker compose up -d
uv run python benchmarks/fetch_beir.py scifact
uv run python benchmarks/index_beir.py scifact     # ~1–5 min, ~$0.25 of embeddings
uv run python benchmarks/eval_beir.py scifact      # ~3 min, a few cents of query embeddings
```

Embeddings come from Vertex (gemini-embedding-001), with the project read from
`.env`, so indexing is billed. The indexer prints an upper-bound cost before it
embeds anything.

Each dataset gets its own OpenSearch index (`beir_<dataset>_<embedding model>`),
so none of this touches the `brain_chunks` index the API serves, and switching
embedding models starts a fresh index rather than mixing incomparable vectors. Brain is driven in process
rather than over HTTP — that is how a platform embedding it would call it, and
it makes the index and the absent LLM easy to control.

Re-running the indexer is free: the document store is on disk, so a second run
skips everything unchanged. Pass `--force` after a chunk-size change, which
neither dedupe gate can detect.

## Datasets

| | Size | Queries | Why |
|---|---|---|---|
| `scifact` | 5.2K | 300 | Small and quick. Good for iterating on the harness. |
| `nfcorpus` | 3.6K | 323 | Medical. Closest to a biotech corpus, and genuinely hard — nDCG@10 near 0.3 is normal. |
| `trec-covid` | 171K | 50 | For how retrieval behaves at scale, not how accurate it is. |

## Results

Recorded, with costs and configuration, in
[docs/retrieval-benchmarks.md](../docs/retrieval-benchmarks.md).

## Reading the numbers

`nDCG@10` is the headline and the one BEIR papers report, so it can be compared
against published baselines rather than only against last week. `Recall@100`
answers a different and often more useful question: is the right document
anywhere in the candidate pool? If recall is high and nDCG is low, ranking is
the problem. If recall is low, no amount of reranking will save it.

Treat the first run as a baseline, not a grade. What matters is that the number
moves when someone changes the chunk size, the embedding model, or the hybrid
weighting — which is exactly what nothing in this repository currently detects.

## What this does not cover

Groundedness and citation accuracy, which need the LLM and belong in their own
suite; whether the system abstains when the answer is absent; access control,
which is deliberately excluded here (every document is indexed public so the
ACL filter cannot influence the score); and ingest throughput under load, though
the indexer prints a docs/sec rate that is worth watching.
