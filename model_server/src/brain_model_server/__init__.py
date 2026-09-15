# MIT License. Copyright (c) 2026 Angel Murillo.
"""The embedding model server: one FastAPI process wrapping one SentenceTransformer.

Separate from `brain` on purpose. It is the only thing in the system that needs
torch, and keeping it in its own distribution keeps a two-gigabyte dependency out
of every process that only wants to call it.
"""
