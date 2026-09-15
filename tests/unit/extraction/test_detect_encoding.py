# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx tests/unit/onyx/file_processing/test_detect_encoding.py.
from __future__ import annotations

from io import BytesIO
from unittest import mock

from brain.extraction.extract import detect_encoding


def test_utf8_cyrillic_returns_utf8_without_chardet() -> None:
    """Valid UTF-8 Cyrillic must be identified as utf-8 without calling chardet.

    chardet reads this as windows-1251 often enough that asking it at all is the
    bug: the result is mojibake in the index.
    """
    cyrillic = "Привет мир".encode()
    with mock.patch("brain.extraction.extract.chardet.detect") as m:
        result = detect_encoding(BytesIO(cyrillic))
    assert result == "utf-8"
    m.assert_not_called()


def test_legacy_encoded_bytes_falls_back_to_chardet() -> None:
    windows1251 = "Привет мир".encode("windows-1251")
    result = detect_encoding(BytesIO(windows1251))
    assert result.lower() != "utf-8"


def test_file_seek_reset_after_call() -> None:
    f = BytesIO(b"hello world")
    detect_encoding(f)
    assert f.tell() == 0
