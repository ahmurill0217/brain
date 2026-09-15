"""HTTP service, for callers that would rather not embed brain in-process."""

from brain.api.app import create_app

__all__ = ["create_app"]
