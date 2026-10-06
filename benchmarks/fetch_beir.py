"""Download a BEIR dataset.

    uv run python benchmarks/fetch_beir.py scifact

BEIR ships human relevance judgements, which is the whole point: it makes
recall and nDCG real numbers rather than an impression, and the published
baselines give something to compare against.

Datasets worth starting with:

    scifact     5.2K abstracts, 300 test queries. Small and fast; good for
                iterating on the evaluator itself.
    nfcorpus    3.6K medical documents, 323 queries. Closest domain to a
                biotech corpus, and hard: nDCG@10 around 0.3 is normal.
    trec-covid  171K documents, 50 queries. Use it when the question is how
                retrieval behaves at scale rather than how accurate it is.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile

import requests
from common import DATA

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", default="scifact")
    args = parser.parse_args()

    target = DATA / args.dataset
    if (target / "corpus.jsonl").exists():
        print(f"{target} already present")
        return 0

    DATA.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{args.dataset}.zip"
    print(f"downloading {url}")

    response = requests.get(url, timeout=900, stream=True)
    if response.status_code == 404:
        print(f"no such BEIR dataset: {args.dataset}", file=sys.stderr)
        return 1
    response.raise_for_status()

    payload = response.content
    print(f"  {len(payload) / 1e6:.1f} MB, extracting")
    # The archives carry their own top-level directory named for the dataset,
    # so extracting into DATA lands it at DATA/<dataset> already.
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(DATA)

    for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        path = target / name
        print(f"  {'ok ' if path.exists() else 'MISSING'} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
