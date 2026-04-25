"""Tests for POST /ml/query — the cognitive chat relay endpoint.

These tests call ``routes_query.ml_query`` directly (no TestClient) to
stay independent of ``httpx`` / ``starlette.testclient`` availability.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest
from fastapi import HTTPException

from iot_machine_learning.ml_service.api import routes_query
from iot_machine_learning.ml_service.api.routes_query import QueryRequest


def _invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the async handler synchronously against a payload dict."""
    req = QueryRequest(**payload)
    return asyncio.run(routes_query.ml_query(req, _="dev-mode"))


# ── Happy path ───────────────────────────────────────────────────────────


def test_query_returns_response_text() -> None:
    """Real engine, real analyzers. ``response_text`` is a non-empty str."""
    body = _invoke({
        "session_id": "sess-happy",
        "message": "test query",
        "tenant_id": "tenant-a",
    })
    assert isinstance(body["response_text"], str)
    assert body["response_text"] != ""
    assert body["metadata"] is not None


# ── Validation ───────────────────────────────────────────────────────────


def test_query_empty_message_returns_400() -> None:
    """Empty / whitespace messages must raise ``HTTPException(400)``."""
    with pytest.raises(HTTPException) as exc:
        _invoke({
            "session_id": "sess-x",
            "message": "   ",
            "tenant_id": "t",
        })
    assert exc.value.status_code == 400


# ── Weaviate graceful-fail ───────────────────────────────────────────────


def test_query_weaviate_unavailable_still_returns_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weaviate blowing up must not break the pipeline."""
    def _boom(*_a: Any, **_kw: Any):
        raise RuntimeError("weaviate down")

    monkeypatch.setattr(
        routes_query, "resolve_weaviate_url", lambda: "http://fake:8080",
    )
    monkeypatch.setattr(routes_query, "recall_similar_documents", _boom)

    body = _invoke({
        "session_id": "sess-wv",
        "message": "hola, ¿cómo están los servidores?",
        "tenant_id": "t",
    })
    assert isinstance(body["response_text"], str)
    assert body["response_text"] != ""


# ── Engine error swallow ─────────────────────────────────────────────────


def test_query_engine_error_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine explosion must yield empty response_text (never propagate)."""

    class _BoomEngine:
        def analyze(self, *_a: Any, **_kw: Any):
            raise RuntimeError("engine exploded")

    monkeypatch.setattr(routes_query, "_engine", _BoomEngine())

    body = _invoke({
        "session_id": "sess-err",
        "message": "anything",
        "tenant_id": "t",
    })
    assert body["response_text"] == ""
    assert body["metadata"] is None


# ── Metadata shape ───────────────────────────────────────────────────────


def test_query_metadata_contains_conclusion() -> None:
    """Metadata dict must expose the ``conclusion`` key mirroring response_text."""
    body = _invoke({
        "session_id": "sess-meta",
        "message": "El servidor está caído, hay un incidente crítico",
        "tenant_id": "t",
    })
    meta: Dict[str, Any] = body["metadata"]
    assert isinstance(meta, dict)
    assert "conclusion" in meta
    assert meta["conclusion"] == body["response_text"]
