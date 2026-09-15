"""brain: document ingest, hybrid retrieval, and cited answering.

Everything most callers need is here:

    from brain import Brain, Document, TextSection, AccessScope

    brain = Brain.from_settings()
    brain.ensure_ready()
    brain.ingest([Document(...)])
    for event in brain.answer("...", access=AccessScope(user_email="you@example.com")):
        ...

Deeper imports are stable too, but this is the surface that will not move.
"""

from brain.config import BrainSettings, get_settings, set_settings
from brain.facade import Brain, NoLLMConfiguredError
from brain.models.acl import AccessScope, ExternalAccess
from brain.models.document import (
    Document,
    DocumentFailure,
    ExpertInfo,
    ImageSection,
    Section,
    SectionType,
    TabularSection,
    TextSection,
)
from brain.models.results import (
    AnswerOptions,
    DeleteResult,
    IngestResult,
    SearchOptions,
    SearchResult,
)
from brain.models.search import (
    CitationInfo,
    InferenceSection,
    SearchDoc,
    SearchFilters,
    Tag,
    TimeRange,
)

__version__ = "0.1.0"

__all__ = [
    "AccessScope",
    "AnswerOptions",
    "Brain",
    "BrainSettings",
    "CitationInfo",
    "DeleteResult",
    "Document",
    "DocumentFailure",
    "ExpertInfo",
    "ExternalAccess",
    "ImageSection",
    "InferenceSection",
    "IngestResult",
    "NoLLMConfiguredError",
    "SearchDoc",
    "SearchFilters",
    "SearchOptions",
    "SearchResult",
    "Section",
    "SectionType",
    "TabularSection",
    "Tag",
    "TextSection",
    "TimeRange",
    "__version__",
    "get_settings",
    "set_settings",
]
