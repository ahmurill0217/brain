"""Unit-suite isolation from the developer's environment.

Every unit test builds its settings explicitly, but pydantic-settings still
reads `BRAIN_*` variables from the process environment. Those can arrive
without anyone exporting them: importing magika (markitdown's file-type
detector) calls `load_dotenv(find_dotenv())`, which finds the project's `.env`
and copies it into `os.environ`. Whether a test saw it would then depend on
whether an extraction test happened to run first.

The external and e2e suites read `BRAIN_*` on purpose, so this lives here
rather than in the top-level conftest.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _no_brain_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [k for k in os.environ if k.startswith("BRAIN_")]:
        monkeypatch.delenv(key)
