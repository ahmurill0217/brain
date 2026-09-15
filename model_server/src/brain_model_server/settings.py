# Values derived from shared_configs/configs.py.
"""Model server configuration.

The `BRAIN_MODEL_SERVER_` prefix is chosen so the variables line up with the
client's: brain's `BrainSettings.model_server_port` also reads
`BRAIN_MODEL_SERVER_PORT`, so one environment variable configures both ends of
the connection instead of two that can disagree.

There is deliberately no bind-host setting. `BRAIN_MODEL_SERVER_HOST` is the
address *clients* dial; letting it drive the bind would make a container
configured with `localhost` reachable only by its own healthcheck.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BRAIN_MODEL_SERVER_",
        extra="ignore",
        frozen=True,
    )

    port: int = 9000

    # Exit cleanly instead of loading torch, for deployments that point brain at
    # an external model server but still run this container.
    disabled: bool = False

    # Only this model is baked into the image, so only this model may be loaded
    # with the network off; see `local_files_only` in encoders.py.
    default_model: str = "nomic-ai/nomic-embed-text-v1"

    # Floor for torch's intra-op thread pool, kept from Onyx.
    min_threads: int = 1

    # Relative to the image's WORKDIR (/app). The build stage downloads weights
    # into the temp path; the first startup merges it into the real cache so a
    # user-mounted cache volume is not clobbered.
    hf_cache_path: Path = Path(".cache/huggingface")
    temp_hf_cache_path: Path = Path(".cache/temp_huggingface")


@lru_cache(maxsize=1)
def get_settings() -> ModelServerSettings:
    """Process-wide settings, read from the environment once."""
    return ModelServerSettings()
