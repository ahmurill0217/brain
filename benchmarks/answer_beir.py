"""Run scifact claims through brain's full answer loop and grade the citations.

    uv run python benchmarks/answer_beir.py --limit 100

The retrieval benchmarks stop at the ranked list. This asks the deployed
pipeline (answer model, fast model, query expansion, section selection) to
judge each claim, and checks what it cites against the human judgements:

  citation hit        the answer cites at least one judged-relevant abstract
  citation precision  share of cited abstracts that are judged relevant. A
                      lower bound: scifact judges only a claim's evidence, so
                      an unjudged abstract can still be on topic
  retrieval hit       a judged-relevant abstract was in some search result,
                      which separates "search missed it" from "the model
                      did not cite it"
  verdict accuracy    SUPPORTED or CONTRADICTED matches scifact's label, for
                      claims that have one

Index the corpus first (index_beir.py scifact). Answering is billed: about
$0.02-0.03 per claim with the Pro answer model and the Flash fast model.
A JSON report with every answer is written to benchmarks/data/results/.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from common import DATA, brain_for, require_stack
from eval_beir import load_qrels

from brain import AccessScope

QUESTION = (
    "Does the research support or contradict this claim? Start your answer "
    "with exactly one of SUPPORTED, CONTRADICTED, or NOT ENOUGH EVIDENCE, then "
    "explain with citations.\n\nClaim: {claim}"
)
VERDICTS = {"SUPPORT": "SUPPORTED", "CONTRADICT": "CONTRADICTED"}
VERDICT = re.compile(r"\b(SUPPORTED|CONTRADICTED|NOT ENOUGH EVIDENCE)\b")

# USD per million tokens (input, output), Vertex list prices.
PRICES = {"gemini-2.5-pro": (1.25, 10.00), "gemini-2.5-flash": (0.30, 2.50)}
USAGE: dict[str, list[int]] = defaultdict(lambda: [0, 0])
_usage_lock = threading.Lock()


def install_usage_hooks() -> None:
    """Count every Gemini call's tokens, per model, without touching brain."""
    from brain.llm.vertex import VertexGeminiLLM

    invoke, stream = VertexGeminiLLM.invoke, VertexGeminiLLM.stream

    def record(model: str, usage: Any) -> None:
        with _usage_lock:
            USAGE[model][0] += usage.prompt_tokens
            USAGE[model][1] += usage.completion_tokens

    def counted_invoke(self, *args, **kwargs):
        response = invoke(self, *args, **kwargs)
        if response.usage:
            record(self.config.model_name, response.usage)
        return response

    def counted_stream(self, *args, **kwargs):
        last = None
        for chunk in stream(self, *args, **kwargs):
            last = chunk.usage or last
            yield chunk
        if last:
            record(self.config.model_name, last)

    VertexGeminiLLM.invoke = counted_invoke
    VertexGeminiLLM.stream = counted_stream


def cost() -> float:
    return sum(
        tokens_in / 1e6 * PRICES.get(model, (0, 0))[0]
        + tokens_out / 1e6 * PRICES.get(model, (0, 0))[1]
        for model, (tokens_in, tokens_out) in USAGE.items()
    )


def load_claims() -> dict[str, dict[str, Any]]:
    path = DATA / "scifact" / "queries.jsonl"
    with path.open(encoding="utf-8") as handle:
        return {r["_id"]: r for r in map(json.loads, handle)}


def expected_verdict(claim: dict[str, Any]) -> str | None:
    labels = {e["label"] for evidence in claim["metadata"].values() for e in evidence}
    return VERDICTS[labels.pop()] if len(labels) == 1 else None


def ask(brain, claim: dict[str, Any], relevant: set[str]) -> dict[str, Any]:
    started = time.perf_counter()
    first_text = None
    text, cited, retrieved, errors, searches = [], [], set(), [], []
    for event in brain.answer(QUESTION.format(claim=claim["text"]), access=AccessScope.anonymous()):
        if event.type == "answer_delta":
            first_text = first_text or time.perf_counter() - started
            text.append(event.text)
        elif event.type == "citation":
            cited.append(event.document_id)
        elif event.type == "search_documents":
            retrieved.update(doc.document_id for doc in event.documents)
        elif event.type == "search_started":
            searches.append(event.queries)
        elif event.type == "answer_error":
            errors.append(event.message)

    answer = "".join(text)
    found = VERDICT.search(answer)
    return {
        "id": claim["_id"],
        "claim": claim["text"],
        "answer": answer,
        "errors": errors,
        "searches": searches,
        "relevant": sorted(relevant),
        "cited": cited,
        "citation_hit": bool(relevant & set(cited)),
        "precision": len(relevant & set(cited)) / len(cited) if cited else None,
        "retrieval_hit": bool(relevant & retrieved),
        "expected": expected_verdict(claim),
        "verdict": found.group(1) if found else None,
        "first_text_s": first_text,
        "total_s": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=100, help="claims to run, in qrels order")
    parser.add_argument("--workers", type=int, default=4, help="claims answered at once")
    args = parser.parse_args()

    install_usage_hooks()
    brain = brain_for("scifact", answering=True)
    require_stack(brain)
    if brain.llm is None:
        raise SystemExit("BRAIN_LLM_MODEL is not set in .env; answering needs a model.")
    print(
        f"answer model {brain.llm.config.model_name}, "
        f"fast model {brain.fast_llm.config.model_name}, "
        f"index {brain.settings.opensearch_index_name}"
    )

    claims = load_claims()
    qrels = load_qrels("scifact", "test")
    ids = [q for q in qrels if q in claims][: args.limit]

    started = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(ask, brain, claims[q], {d for d, r in qrels[q].items() if r > 0})
            for q in ids
        ]
        results = []
        for i, future in enumerate(futures, start=1):
            results.append(future.result())
            if i % 10 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} claims, ${cost():.2f} so far", flush=True)
    elapsed = time.monotonic() - started

    def share(values: list[bool]) -> str:
        return f"{sum(values) / len(values):.1%} ({sum(values)}/{len(values)})"

    answered = [r for r in results if not r["errors"]]
    labeled = [r for r in answered if r["expected"]]
    unlabeled = [r for r in answered if not r["expected"]]
    precisions = [r["precision"] for r in answered if r["precision"] is not None]
    firsts = [r["first_text_s"] for r in answered if r["first_text_s"]]

    print(f"\nscifact answers  ({len(results)} claims in {elapsed:.0f}s)")
    print("-" * 58)
    print(f"  answered without error   {share([not r['errors'] for r in results])}")
    print(f"  cited something          {share([bool(r['cited']) for r in answered])}")
    print(f"  citation hit             {share([r['citation_hit'] for r in answered])}")
    print(f"  retrieval hit            {share([r['retrieval_hit'] for r in answered])}")
    if precisions:
        print(f"  citation precision       {statistics.mean(precisions):.1%} (lower bound)")
    if labeled:
        print(
            f"  verdict accuracy         {share([r['verdict'] == r['expected'] for r in labeled])}"
        )
    if unlabeled:
        print(
            "  unlabeled claims called  "
            + ", ".join(
                f"{v or 'no verdict'} {sum(r['verdict'] == v for r in unlabeled)}"
                for v in ("SUPPORTED", "CONTRADICTED", "NOT ENOUGH EVIDENCE", None)
            )
        )
    if firsts:
        print(f"  first text p50           {statistics.median(firsts):.1f}s")
    print(f"  cost                     ${cost():.2f} (${cost() / len(results):.4f} per claim)")
    print(f"  tokens (in, out)         {dict(USAGE)}")

    out_dir = DATA / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"scifact-answers-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"usage": dict(USAGE), "results": results}, indent=2))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
