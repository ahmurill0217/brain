"""Index a BEIR corpus into its own brain index.

    uv run python benchmarks/index_beir.py scifact

Safe to re-run: the document store persists, so a second run skips everything
unchanged and costs almost nothing. Pass --force after changing the chunk size
or the embedding model, neither of which the dedupe gates can see.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

from common import DATA, brain_for, require_stack

from brain import Document, ExternalAccess, TextSection

BATCH = 200


def documents(dataset: str):
    """BEIR corpus lines as brain Documents.

    Title and body are kept as one section rather than two. They are one
    passage in BEIR's own evaluation, and splitting them would let a title-only
    match count as a hit against qrels that never judged it that way.
    """
    path = DATA / dataset / "corpus.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found; run fetch_beir.py {dataset} first")

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            title = (record.get("title") or "").strip()
            body = (record.get("text") or "").strip()
            if not body and not title:
                continue
            yield Document(
                id=record["_id"],
                source="beir",
                semantic_identifier=title or record["_id"],
                title=title or None,
                sections=[TextSection(text=f"{title}\n\n{body}".strip())],
                # Explicit rather than relying on the public-by-default
                # setting, so the benchmark measures retrieval and never
                # accidentally measures the ACL filter.
                external_access=ExternalAccess.public(),
            )


# gemini-embedding-001, USD per million input tokens.
EMBEDDING_PRICE_PER_M = 0.15


def estimate_cost(brain, dataset: str) -> None:
    """Print what embedding the corpus will cost before paying for it.

    An upper bound: every chunk repeats its title, and documents the store
    already holds are skipped and cost nothing.
    """
    tokens = sum(
        len(brain.tokenizer.encode(section.text))
        for document in documents(dataset)
        for section in document.sections
    )
    print(
        f"{tokens:,} tokens to embed, about ${tokens / 1e6 * EMBEDDING_PRICE_PER_M:.2f} "
        "at most (documents already in the store are skipped)"
    )


def count_vertex_responses(brain) -> Counter:
    """Tally every HTTP status Vertex returns, retries included.

    The embedder retries rate limits on its own, so they never surface as
    failures; this is how a run shows how often it was throttled, which is
    what sizes an ingest queue's rate limit.
    """
    statuses: Counter = Counter()
    session = brain.embedder._client._session
    post = session.post

    def counted(*args, **kwargs):
        response = post(*args, **kwargs)
        statuses[response.status_code] += 1
        return response

    session.post = counted
    return statuses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="scifact")
    parser.add_argument("--force", action="store_true", help="reindex unchanged documents")
    parser.add_argument("--estimate", action="store_true", help="print the embedding cost and stop")
    args = parser.parse_args()

    brain = brain_for(args.dataset)
    require_stack(brain)
    brain.ensure_ready()
    print(
        f"index {brain.settings.opensearch_index_name}, "
        f"embedding with {brain.settings.embedding_model_name}"
    )
    estimate_cost(brain, args.dataset)
    if args.estimate:
        return 0
    statuses = count_vertex_responses(brain)

    batch: list[Document] = []
    totals = {"seen": 0, "indexed": 0, "skipped": 0, "chunks": 0, "failures": 0}
    started = time.monotonic()

    def flush() -> None:
        if not batch:
            return
        result = brain.ingest(batch, force=args.force)
        totals["indexed"] += result.indexed_documents
        totals["skipped"] += result.skipped_documents
        totals["chunks"] += result.total_chunks
        totals["failures"] += len(result.failures)
        if result.failures:
            print(f"  {len(result.failures)} failures, first: {result.failures[0]}")
        rate = totals["seen"] / max(time.monotonic() - started, 1e-9)
        print(
            f"  {totals['seen']:>6} docs  "
            f"{totals['indexed']:>6} indexed  {totals['skipped']:>6} skipped  "
            f"{totals['chunks']:>6} chunks  {rate:5.1f} docs/s",
            flush=True,
        )
        batch.clear()

    for document in documents(args.dataset):
        batch.append(document)
        totals["seen"] += 1
        if len(batch) >= BATCH:
            flush()
    flush()

    elapsed = time.monotonic() - started
    print(
        f"\n{totals['seen']} documents in {elapsed:.0f}s "
        f"({totals['seen'] / max(elapsed, 1e-9):.1f}/s), "
        f"{totals['chunks']} chunks indexed, {totals['failures']} failures"
    )
    throttled = statuses.get(429, 0)
    print(
        f"Vertex embedding requests: {sum(statuses.values())}, "
        f"rate-limited (429): {throttled}, other errors: "
        f"{sum(n for code, n in statuses.items() if code not in (200, 429))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
