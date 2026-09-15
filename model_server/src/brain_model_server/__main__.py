# Derived from model_server/__main__.py.
"""Process entry point: `python -m brain_model_server`.

The disabled gate runs here, ahead of `main`'s torch and transformers imports,
so a container that has nothing to do exits in milliseconds instead of loading
two gigabytes of model stack first.
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

logger = logging.getLogger(__name__)


def main() -> None:
    from brain_model_server.settings import get_settings

    if get_settings().disabled:
        logger.info("BRAIN_MODEL_SERVER_DISABLED is set; not starting the model server.")
        sys.exit(0)

    from brain_model_server.main import run_server

    run_server()

    # uvicorn.run() returns only once the server has stopped serving, so getting
    # here at all is a failure. Exit non-zero, or compose's `restart: on-failure`
    # and k8s's `restartPolicy: OnFailure` would treat the crash as a clean stop.
    sys.exit(1)


if __name__ == "__main__":
    main()
