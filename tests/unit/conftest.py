"""Unit-suite isolation from the developer's environment.

Every unit test builds its settings explicitly, but pydantic-settings still
reads `BRAIN_*` variables from the process environment, so a developer's shell,
or any library that loads `.env` into `os.environ`, would leak into the
results. (magika, once pulled in by markitdown, did exactly that on import.)

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
