"""Merge already-indexed BEIR datasets into one index, without re-embedding.

    uv run python benchmarks/combine_indexes.py scifact trec-covid

Copies every chunk, vectors included, from each dataset's index into a new
index named after all of them (here `beir_scifact-trec-covid_<model>`), using
OpenSearch's server-side reindex. Nothing is sent to Vertex, so it costs
nothing beyond disk.

The point is a bigger haystack for a strict benchmark: scifact's claims each
have about one judged abstract, and merging in trec-covid's 171K biomedical
papers surrounds those abstracts with realistic distractors. Pass the merged
name to the answer benchmark with `--corpus scifact-trec-covid`.

Document ids must not collide across the datasets. scifact's are numeric and
trec-covid's are alphanumeric, and the chunk counts are checked afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time

from common import brain_for, require_stack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="+", help="datasets already indexed by index_beir.py")
    args = parser.parse_args()

    sources = [brain_for(d).settings.opensearch_index_name for d in args.datasets]
    target = brain_for("-".join(args.datasets))
    require_stack(target)
    raw = target.index.client._client
    name = target.settings.opensearch_index_name

    if raw.indices.exists(index=name):
        raise SystemExit(f"{name} already exists; delete it first to rebuild it")
    missing = [s for s in sources if not raw.indices.exists(index=s)]
    if missing:
        raise SystemExit(f"not indexed yet: {missing}; run index_beir.py first")

    # Brain's own mapping and settings, so the merged index searches exactly
    # like one brain built itself.
    target.ensure_ready()
    expected = sum(raw.count(index=s)["count"] for s in sources)
    print(f"copying {expected:,} chunks from {sources} into {name}")

    started = time.monotonic()
    task = raw.reindex(
        body={"source": {"index": sources}, "dest": {"index": name}},
        wait_for_completion=False,
    )["task"]
    while True:
        status = raw.tasks.get(task_id=task)
        progress = status["task"]["status"]
        print(f"  {progress['created']:,}/{progress['total']:,}", flush=True)
        if status.get("completed"):
            break
        time.sleep(15)
    if status.get("error") or status.get("response", {}).get("failures"):
        raise SystemExit(f"reindex failed: {status.get('error') or status['response']['failures']}")

    raw.indices.refresh(index=name)
    copied = raw.count(index=name)["count"]
    print(f"\n{copied:,} chunks in {name} after {time.monotonic() - started:.0f}s")
    if copied != expected:
        raise SystemExit(f"expected {expected:,}; document ids collided or the copy is partial")
    return 0


if __name__ == "__main__":
    sys.exit(main())
