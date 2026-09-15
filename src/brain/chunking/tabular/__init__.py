# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/indexing/chunking/tabular_section_chunker/__init__.py.
"""Chunking for spreadsheets and CSVs."""

from brain.chunking.tabular.analysis import SheetAnalysis, analyze_sheet
from brain.chunking.tabular.chunker import BlobReader, TabularChunker
from brain.chunking.tabular.sheet_descriptor import build_sheet_descriptor_chunks
from brain.chunking.tabular.total_descriptor import TOTALS_HEADER, build_total_descriptor_chunks

__all__ = [
    "TOTALS_HEADER",
    "BlobReader",
    "SheetAnalysis",
    "TabularChunker",
    "analyze_sheet",
    "build_sheet_descriptor_chunks",
    "build_total_descriptor_chunks",
]
