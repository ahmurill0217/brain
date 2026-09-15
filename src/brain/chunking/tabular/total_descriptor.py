# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/indexing/chunking/tabular_section_chunker/total_descriptor.py.
"""The "what do the numbers add up to" chunk.

"What did we spend in total" is answered by no row in the sheet, so retrieval
has nothing to return and the model either refuses or invents a figure. This
computes the aggregate and writes it into a chunk, with a header that
deliberately spells out the vocabulary such questions use, so the question
matches the chunk that literally contains its answer.
"""

from __future__ import annotations

from collections import Counter

from brain.chunking.tabular.analysis import NumericAggregate, SheetAnalysis
from brain.chunking.tabular.util import label, pack_lines
from brain.text.tokenizer import BaseTokenizer

TOTALS_HEADER = (
    "Totals and overall aggregates across all rows. This sheet can answer "
    "whole-dataset questions about total, overall, grand total, sum across "
    "all, average, combined, mean, minimum, maximum, and count of values."
)


def build_total_descriptor_chunks(
    headers: list[str],
    analysis: SheetAnalysis,
    heading: str,
    tokenizer: BaseTokenizer,
    max_tokens: int,
) -> list[str]:
    if analysis.row_count == 0:
        return []

    lines: list[str] = [
        _numeric_totals_line(headers[idx], analysis.numeric_stats[idx])
        for idx in analysis.numeric_cols
    ]
    for idx in analysis.categorical_cols:
        line = _categorical_top_line(headers[idx], analysis.categorical_counts[idx])
        if line:
            lines.append(line)

    # A sheet of free text has nothing to total; emitting the header alone would
    # be a chunk that matches aggregate questions and then answers none of them.
    if not lines:
        return []

    lines.append(f"Total row count: {analysis.row_count}.")

    prefix = (f"{heading}\n" if heading else "") + TOTALS_HEADER
    return pack_lines(
        lines=lines,
        prefix=prefix,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )


def _numeric_totals_line(name: str, agg: NumericAggregate) -> str:
    return (
        f"Column {label(name)}: total (sum across all rows) = {_fmt(agg.total)}, "
        f"average = {_fmt(agg.average)}, minimum = {_fmt(agg.minimum)}, "
        f"maximum = {_fmt(agg.maximum)}, count = {agg.count}."
    )


def _categorical_top_line(name: str, counts: Counter[str]) -> str:
    top = counts.most_common(1)
    if not top:
        return ""
    val, n = top[0]
    return f"Column {label(name)} most frequent value: {val} ({n} occurrences)."


def _fmt(num: float) -> str:
    # Counts and currency read as "600", not "600.0"; the guard keeps floats
    # too large for an exact int round-trip out of the integer branch.
    if abs(num) < 1e15 and num == int(num):
        return str(int(num))
    return f"{num:.6g}"
