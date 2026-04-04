"""Job processing: row parsing, payload building, analysis dispatch.

Single responsibility: transform a raw DB row into a typed ``QueueItem``,
build the normalized payload, and dispatch to ``DocumentAnalyzer``.
"""

from iot_machine_learning.ml_service.workers.queue_repository import find_analysis_result

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    """Typed representation of a row from zenin_docs.ingestion_queue.

    Replaces raw tuple indexing (``row[0]``, ``row[3]``, etc.) with
    named, documented fields.
    """

    queue_id: Any
    tenant_id: Any
    user_id: Any
    content_type: str
    source_type: Optional[str]
    filename: str
    file_extension: str
    content: str
    metadata: Dict[str, Any]
    created_at: Any


def parse_queue_row(row) -> QueueItem:
    """Parse a raw SQLAlchemy row into a ``QueueItem``.

    Column order must match ``queue_repository.SELECT_PENDING``:
    Id, TenantId, UserId, ContentType, SourceType,
    OriginalFilename, FileExtension, Content, Metadata, CreatedAt
    """
    metadata_json = row[8]
    metadata: Dict[str, Any] = {}
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    # DEBUG: Log content extraction
    content = row[7] or ""
    logger.info(f"[STAGE-1] queue item received, id={row[0]}, content_length={len(content)}")

    return QueueItem(
        queue_id=row[0],
        tenant_id=row[1],
        user_id=row[2],
        content_type=row[3] or "text",
        source_type=row[4],
        filename=row[5] or "unknown",
        file_extension=row[6] or "",
        content=content,
        metadata=metadata,
        created_at=row[9],
    )


def build_payload(item: QueueItem) -> Dict[str, Any]:
    """Build the normalized_payload dict that DocumentAnalyzer expects.

    Routes by ``content_type``:
    - ``"text"`` → full_text + word/char/paragraph counts
    - ``"tabular"`` / ``"numeric"`` → parsed JSON series + headers
    - anything else → treat as text
    """
    content = item.content
    content_type = item.content_type
    metadata = item.metadata

    if content_type == "text":
        return _build_text_payload(content)

    if content_type in ("tabular", "numeric"):
        return _build_tabular_payload(content, metadata)

    # mixed or unknown — treat as text
    return _build_text_payload(content)


def _build_text_payload(content: str) -> Dict[str, Any]:
    """Build payload for text content."""
    words = content.split() if content else []
    paragraphs = content.split("\n\n") if content else []
    
    # DEBUG: Log payload building
    logger.info(f"[STAGE-2] payload built, keys={['data']}")
    
    payload = {
        "data": {
            "full_text": content,
            "word_count": len(words),
            "char_count": len(content),
            "paragraph_count": len(paragraphs),
        }
    }
    
    # DEBUG: Log full text
    logger.info(f"[STAGE-3] full_text length={len(content)}")
    
    return payload


def _build_tabular_payload(
    content: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Build payload for tabular/numeric content.

    Content is a JSON string: ``{ "col_name": [v1, v2, ...], ... }``
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = {}

    raw_series: Dict[str, list] = {}
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            if isinstance(v, list):
                raw_series[k] = v

    return {
        "data": {
            "row_count": metadata.get("record_count", 0),
            "numeric_columns": list(raw_series.keys()),
            "headers": list(parsed.keys()) if isinstance(parsed, dict) else [],
            "raw_series": raw_series,
            "sample_rows": [],
        }
    }
