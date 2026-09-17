"""Weaviate document writer helpers."""
from __future__ import annotations
import json
import logging
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_WEAVIATE_CLASS = "ZeninDocument"


def index_document(
    base_url: str,
    timeout: int,
    dry_run: bool,
    text: str,
    source: str,
    classification: str,
    tenant_id: str,
    analysis_result_id: Optional[str] = None,
    context_type: str = "documental",
) -> Optional[str]:
    """Index a document chunk into Weaviate."""
    doc_id = str(uuid.uuid4())
    payload: Dict[str, Any] = {
        "class": _WEAVIATE_CLASS,
        "id": doc_id,
        "properties": {
            "content": text,
            "source": source,
            "classification": classification,
            "tenantId": tenant_id,
            "analysisResultId": analysis_result_id or "",
            "contextType": context_type,
        },
    }

    if dry_run:
        logger.debug(
            "[COGNITIVE] dry_run remember_document doc_id=%s tenant=%s context=%s",
            doc_id,
            tenant_id,
            context_type,
        )
        return doc_id

    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/v1/objects",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("id", doc_id)
    except urllib.error.HTTPError as exc:
        logger.error(
            "[COGNITIVE] remember_document HTTP %d: %s",
            exc.code,
            exc.reason,
        )
    except Exception as exc:
        logger.error("[COGNITIVE] remember_document failed: %s", exc)
    return None
