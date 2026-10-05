"""Tokenization.

Everything is measured with tiktoken. Chunk sizes are meant in the embedding
model's tokens, but Gemini's tokenizer is not published, so cl100k_base stands
in for it. The two disagree by a few percent on English, which is harmless
here: chunks are 512 tokens and the model accepts 2048, and the per-request
embedding budget leaves room for the error.

tiktoken downloads its encoding file the first time it is used. The Dockerfile
bakes it in so a container starts without reaching the network.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

TRIM_SEP_PAT = "\n... {n} tokens removed...\n"

# Cap per encode() call. tiktoken can blow the stack on very large inputs.
_ENCODE_CHUNK_SIZE = 500_000


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, string: str) -> list[int]: ...

    @abstractmethod
    def tokenize(self, string: str) -> list[str]: ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str: ...


class TiktokenTokenizer(BaseTokenizer):
    """One instance per encoding name; building them is not cheap."""

    _instances: dict[str, TiktokenTokenizer] = {}

    def __new__(cls, encoding_name: str = "cl100k_base") -> TiktokenTokenizer:
        if encoding_name not in cls._instances:
            cls._instances[encoding_name] = super().__new__(cls)
        return cls._instances[encoding_name]

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        if not hasattr(self, "encoder"):
            import tiktoken

            self.encoder = tiktoken.get_encoding(encoding_name)

    def encode(self, string: str) -> list[int]:
        # encode_ordinary ignores special tokens, which would otherwise let
        # document text containing "<|endoftext|>" raise.
        return self.encoder.encode_ordinary(string)

    def tokenize(self, string: str) -> list[str]:
        return [self.encoder.decode([t]) for t in self.encode(string)]

    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)


def get_tokenizer(encoding_name: str = "cl100k_base") -> BaseTokenizer:
    """The tokenizer chunks are measured with."""
    return TiktokenTokenizer(encoding_name)


def get_llm_tokenizer(encoding_name: str = "cl100k_base") -> BaseTokenizer:
    return TiktokenTokenizer(encoding_name)


def count_tokens(text: str, tokenizer: BaseTokenizer, token_limit: int | None = None) -> int:
    """Token count, chunked so a huge input cannot overflow the encoder.

    With `token_limit`, stops as soon as the count exceeds it: the result is
    then only a lower bound, which is all a budget check needs.
    """
    if len(text) <= _ENCODE_CHUNK_SIZE:
        return len(tokenizer.encode(text))
    total = 0
    for start in range(0, len(text), _ENCODE_CHUNK_SIZE):
        total += len(tokenizer.encode(text[start : start + _ENCODE_CHUNK_SIZE]))
        if token_limit is not None and total > token_limit:
            return total
    return total


def tokenizer_trim_content(content: str, desired_length: int, tokenizer: BaseTokenizer) -> str:
    """Truncate to at most `desired_length` tokens."""
    tokens = tokenizer.encode(content)
    if len(tokens) <= desired_length:
        return content
    return tokenizer.decode(tokens[:desired_length])


def tokenizer_trim_middle(
    tokens: list[int], desired_length: int, tokenizer: BaseTokenizer
) -> str:
    """Drop the middle, keeping both ends.

    Better than truncation for documents whose beginning and end carry the most
    signal, and the marker tells the model something was removed.
    """
    if len(tokens) <= desired_length:
        return tokenizer.decode(tokens)
    sep_str = TRIM_SEP_PAT.format(n=len(tokens) - desired_length)
    sep_tokens = tokenizer.encode(sep_str)
    slice_size = (desired_length - len(sep_tokens)) // 2
    if slice_size <= 0:
        raise ValueError("desired_length is too short to fit the trim marker")
    return tokenizer.decode(tokens[:slice_size]) + sep_str + tokenizer.decode(tokens[-slice_size:])


def split_text_by_tokens(text: str, tokenizer: BaseTokenizer, max_tokens: int) -> list[str]:
    """Split into pieces of at most `max_tokens`.

    Best effort: re-tokenizing a piece can drift by a token or two because BPE
    merges differ at the new boundaries. Fine for splitting oversized content,
    not for enforcing a hard limit.
    """
    if not text:
        return []
    token_ids: list[int] = []
    for start in range(0, len(text), _ENCODE_CHUNK_SIZE):
        token_ids.extend(tokenizer.encode(text[start : start + _ENCODE_CHUNK_SIZE]))
    return [
        tokenizer.decode(token_ids[start : start + max_tokens])
        for start in range(0, len(token_ids), max_tokens)
    ]
