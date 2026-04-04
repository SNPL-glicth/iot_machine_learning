"""SQL queries and DB operations for zenin_docs.ingestion_queue.

Single responsibility: all database interactions for the ZENIN queue
pipeline.  If the DB schema changes, only this file changes.

Every function receives a SQLAlchemy ``Connection`` — no global state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

SELECT_PENDING = text("""
    SELECT TOP (:batch_size)
        Id, TenantId, UserId, ContentType, SourceType,
        OriginalFilename, FileExtension, Content, Metadata, CreatedAt
    FROM zenin_docs.ingestion_queue
    WHERE Status = 'pending'
    ORDER BY CreatedAt ASC
""")

MARK_PROCESSING = text("""
    UPDATE zenin_docs.ingestion_queue
    SET Status = 'processing', StartedAt = GETUTCDATE()
    WHERE Id = :id AND Status = 'pending'
""")

MARK_COMPLETED = text("""
    UPDATE zenin_docs.ingestion_queue
    SET Status      = 'completed',
        CompletedAt = GETUTCDATE(),
        AnalysisResultId = :analysis_id
    WHERE Id = :id
""")

MARK_ERROR = text("""
    UPDATE zenin_docs.ingestion_queue
    SET Status       = 'error',
        ErrorMessage = :error,
        CompletedAt  = GETUTCDATE()
    WHERE Id = :id
""")

UPDATE_ANALYSIS_RESULT = text("""
    UPDATE zenin_docs.analysis_results
    SET Classification  = :classification,
        MlResult        = :ml_result,
        Conclusion      = :conclusion,
        WeaviateDocId   = :ml_doc_id,
        SemanticName    = :semantic_name,
        Status          = 'analyzed',
        AnalyzedAt      = GETUTCDATE()
    WHERE Id = :analysis_id
""")

INSERT_ANALYSIS_RESULT = text("""
    INSERT INTO zenin_docs.analysis_results (
        Id, TenantId, UserId, OriginalFilename, FileExtension,
        FileSizeBytes, Classification, MlResult, Conclusion,
        WeaviateDocId, Status, AnalyzedAt, CreatedAt
    )
    OUTPUT INSERTED.Id
    VALUES (
        NEWID(), :tenant_id, :user_id, :filename, :extension,
        :file_size, :classification, :ml_result, :conclusion,
        :ml_doc_id, 'analyzed', GETUTCDATE(), GETUTCDATE()
    )
""")

FIND_ANALYSIS_RESULT = text("""
    SELECT TOP 1 Id
    FROM zenin_docs.analysis_results
    WHERE TenantId = :tenant_id
      AND UserId = :user_id
      AND OriginalFilename = :filename
      AND Status IN ('pending', 'processing')
    ORDER BY CreatedAt DESC
""")


# ---------------------------------------------------------------------------
# Repository functions
# ---------------------------------------------------------------------------


def fetch_pending(conn, batch_size: int) -> List:
    """Fetch pending items from ingestion_queue."""
    return conn.execute(
        SELECT_PENDING, {"batch_size": batch_size}
    ).fetchall()


def mark_processing(conn, queue_id) -> int:
    """Mark item as processing (optimistic lock). Returns affected rows."""
    affected = conn.execute(MARK_PROCESSING, {"id": queue_id}).rowcount
    conn.commit()
    return affected


def mark_completed(conn, queue_id, analysis_id) -> None:
    """Mark item as completed with analysis_result reference."""
    conn.execute(MARK_COMPLETED, {
        "id": queue_id,
        "analysis_id": analysis_id,
    })
    conn.commit()


def mark_error(conn, queue_id) -> None:
    """Mark item as error with traceback."""
    import traceback
    error_msg = traceback.format_exc()[:1000]
    try:
        conn.execute(MARK_ERROR, {"id": queue_id, "error": error_msg})
        conn.commit()
    except Exception:
        logger.exception(
            "[QUEUE_REPO] Failed to mark %s as error", queue_id
        )


def find_analysis_result(
    conn, tenant_id, user_id, filename
) -> Optional[str]:
    """Find existing analysis_results row created by .NET."""
    result = conn.execute(FIND_ANALYSIS_RESULT, {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "filename": filename,
    })
    row = result.fetchone()
    return row[0] if row else None


def update_analysis_result(
    conn,
    analysis_id: str,
    classification: str,
    ml_result_json: str,
    conclusion: str,
    ml_doc_id: Optional[str],
    semantic_name: Optional[str] = None,
) -> None:
    """Update existing analysis_results row."""
    conn.execute(UPDATE_ANALYSIS_RESULT, {
        "classification": classification,
        "ml_result": ml_result_json,
        "conclusion": conclusion,
        "ml_doc_id": ml_doc_id,
        "semantic_name": semantic_name,
        "analysis_id": analysis_id,
    })


def insert_analysis_result(
    conn,
    *,
    tenant_id,
    user_id,
    filename: str,
    extension: str,
    file_size: int,
    classification: str,
    ml_result_json: str,
    conclusion: str,
    semantic_name: str,
    ml_doc_id: Optional[str],
) -> Optional[str]:
    """Insert new analysis_results row. Returns the new Id."""
    result = conn.execute(
        INSERT_ANALYSIS_RESULT,
        {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "filename": filename,
            "extension": extension,
            "file_size": file_size,
            "classification": classification,
            "ml_result": ml_result_json,
            "conclusion": conclusion,
            "semantic_name": semantic_name,
            "ml_doc_id": ml_doc_id,
        },
    )
    row = result.fetchone()
    return row[0] if row else None
