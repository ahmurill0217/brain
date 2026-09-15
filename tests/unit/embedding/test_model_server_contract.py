# MIT License. Copyright (c) 2026 Angel Murillo.
"""The client and the server must agree on the wire format.

`brain/embedding/protocol.py` and `model_server/.../schemas.py` define the same
two models twice, on purpose: the server is a separate distribution with torch in
it and must not depend on brain. Two copies drift, and the drift shows up as a
422 from a running container rather than a failing build. So the build checks.

The server's schemas module is loaded straight off disk rather than imported: it
is a different package, in a different venv, and installing it here would pull in
two gigabytes of model stack to compare two pydantic classes. The module is kept
free of heavy imports for exactly this reason.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from brain.embedding import protocol

SERVER_SCHEMAS_PATH = (
    Path(__file__).parents[3] / "model_server" / "src" / "brain_model_server" / "schemas.py"
)


def _load_server_schemas() -> ModuleType:
    spec = importlib.util.spec_from_file_location("brain_model_server_schemas", SERVER_SCHEMAS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it is executed: the module uses postponed annotations, and
    # pydantic resolves those by looking the module up in sys.modules by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _without_descriptions(schema: Any) -> Any:
    """Strip docstring-derived text.

    The two copies describe themselves differently (one is the contract, the
    other is a mirror of it), and prose is the one difference that cannot break
    a request. Field names, types, defaults, and required-ness all survive.
    """
    if isinstance(schema, dict):
        return {k: _without_descriptions(v) for k, v in schema.items() if k != "description"}
    if isinstance(schema, list):
        return [_without_descriptions(v) for v in schema]
    return schema


@pytest.fixture(scope="module")
def server_schemas() -> ModuleType:
    return _load_server_schemas()


def test_embed_request_schemas_match(server_schemas: ModuleType) -> None:
    assert _without_descriptions(server_schemas.EmbedRequest.model_json_schema()) == (
        _without_descriptions(protocol.EmbedRequest.model_json_schema())
    )


def test_embed_response_schemas_match(server_schemas: ModuleType) -> None:
    assert _without_descriptions(server_schemas.EmbedResponse.model_json_schema()) == (
        _without_descriptions(protocol.EmbedResponse.model_json_schema())
    )


def test_text_types_match(server_schemas: ModuleType) -> None:
    assert [t.value for t in server_schemas.EmbedTextType] == [
        t.value for t in protocol.EmbedTextType
    ]


def test_no_cloud_provider_fields_survived_the_port(server_schemas: ModuleType) -> None:
    """brain talks to one model server. Every field that only existed to reach a
    cloud provider was dropped, and neither copy may grow one back."""
    dropped = {
        "api_key",
        "provider_type",
        "deployment_name",
        "api_url",
        "api_version",
        "reduced_dimension",
    }
    assert dropped.isdisjoint(server_schemas.EmbedRequest.model_fields)
    assert dropped.isdisjoint(protocol.EmbedRequest.model_fields)


def test_a_request_built_by_the_client_validates_on_the_server(
    server_schemas: ModuleType,
) -> None:
    """The schema comparison is structural; this is the round trip it stands in for."""
    client_request = protocol.EmbedRequest(
        texts=["hello"],
        model_name="nomic-ai/nomic-embed-text-v1",
        max_context_length=512,
        normalize_embeddings=True,
        text_type=protocol.EmbedTextType.PASSAGE,
        manual_passage_prefix="search_document: ",
    )

    server_request = server_schemas.EmbedRequest.model_validate(
        client_request.model_dump(mode="json")
    )

    assert server_request.texts == ["hello"]
    assert server_request.text_type is server_schemas.EmbedTextType.PASSAGE
    assert server_request.manual_passage_prefix == "search_document: "
