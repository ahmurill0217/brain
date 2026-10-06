"""Measure brain's retrieval against BEIR relevance judgements.

    uv run python benchmarks/eval_beir.py scifact

Reports nDCG@10, Recall@10, Recall@100 and MRR@10, which is the set BEIR
papers publish, so the numbers can be read against a known baseline instead of
only against yesterday's run.

No LLM is involved. Query expansion and section selection are off, so what is
measured is the hybrid query and the embeddings, and a regression points at
one of those rather than at a model's mood.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict

from common import DATA, brain_for, require_stack

from brain import AccessScope
from brain.models.results import SearchOptions


def load_queries(dataset: str) -> dict[str, str]:
    path = DATA / dataset / "queries.jsonl"
    with path.open(encoding="utf-8") as handle:
        return {r["_id"]: r["text"] for r in map(json.loads, handle)}


def load_qrels(dataset: str, split: str) -> dict[str, dict[str, int]]:
    """query id -> {document id: graded relevance}."""
    path = DATA / dataset / "qrels" / f"{split}.tsv"
    if not path.exists():
        raise SystemExit(f"{path} not found; run fetch_beir.py {dataset} first")

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        # Older dumps omit the header; put the row back if it looks like data.
        if header and header[0] not in ("query-id", "query_id"):
            handle.seek(0)
            reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            # A few judgements are negative (trec-covid has two). trec_eval,
            # which BEIR scores with, treats them as plain non-relevant.
            qrels[row[0]][row[1]] = max(int(row[2]), 0)
    return dict(qrels)


def ndcg_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    gain = sum(relevance.get(doc, 0) / math.log2(rank + 2)
               for rank, doc in enumerate(ranked[:k]))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    best = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return gain / best if best else 0.0


def recall_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    wanted = {doc for doc, rel in relevance.items() if rel > 0}
    if not wanted:
        return 0.0
    return len(wanted & set(ranked[:k])) / len(wanted)


def mrr_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    for rank, doc in enumerate(ranked[:k], start=1):
        if relevance.get(doc, 0) > 0:
            return 1.0 / rank
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only N queries")
    parser.add_argument("--hits", type=int, default=100, help="chunks retrieved per query")
    args = parser.parse_args()

    brain = brain_for(args.dataset)
    require_stack(brain)

    queries = load_queries(args.dataset)
    qrels = load_qrels(args.dataset, args.split)
    query_ids = [q for q in qrels if q in queries]
    if args.limit:
        query_ids = query_ids[: args.limit]
    if not query_ids:
        raise SystemExit("no queries with judgements found")

    print(f"{args.dataset}: {len(query_ids)} queries, retrieving {args.hits} chunks each\n")

    # Public documents only, which is the ordinary read path. Everything was
    # indexed public, so this measures retrieval; a zero here would point at
    # the ACL filter rather than at ranking.
    access = AccessScope.anonymous()
    options = SearchOptions(
        num_hits=args.hits, expand_queries=False, select_sections=False, expand_sections=False
    )

    scores: dict[str, list[float]] = defaultdict(list)
    latencies: list[float] = []
    failures = 0
    started = time.monotonic()

    for i, query_id in enumerate(query_ids, start=1):
        began = time.monotonic()
        try:
            result = brain.search(queries[query_id], access=access, options=options)
            ranked = [doc.document_id for doc in result.search_docs]
        except Exception as exc:
            failures += 1
            print(f"  query {query_id} failed: {type(exc).__name__}: {exc}")
            continue
        latencies.append(time.monotonic() - began)

        relevance = qrels[query_id]
        scores["nDCG@10"].append(ndcg_at_k(ranked, relevance, 10))
        scores["Recall@10"].append(recall_at_k(ranked, relevance, 10))
        scores["Recall@100"].append(recall_at_k(ranked, relevance, 100))
        scores["MRR@10"].append(mrr_at_k(ranked, relevance, 10))

        if i % 25 == 0 or i == len(query_ids):
            print(f"  {i}/{len(query_ids)} queries", flush=True)

    elapsed = time.monotonic() - started
    print(f"\n{args.dataset} / {args.split}  ({len(latencies)} queries in {elapsed:.0f}s)")
    print("-" * 46)
    for metric in ("nDCG@10", "Recall@10", "Recall@100", "MRR@10"):
        if scores[metric]:
            print(f"  {metric:<12} {statistics.mean(scores[metric]):.4f}")
    if latencies:
        print(f"  {'latency p50':<12} {statistics.median(latencies) * 1000:.0f} ms")
        print(f"  {'latency p95':<12} "
              f"{sorted(latencies)[int(len(latencies) * 0.95) - 1] * 1000:.0f} ms")
    if failures:
        print(f"  {'failures':<12} {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
