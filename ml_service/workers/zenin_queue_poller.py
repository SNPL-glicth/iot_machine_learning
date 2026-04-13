"""Worker that polls zenin_docs.ingestion_queue and processes documents.

Thin orchestrator — delegates to focused sub-modules:
- ``queue_repository``  — all SQL queries and DB operations
- ``job_processor``     — row parsing, payload building
- ``result_writer``     — analysis_result persistence + Weaviate

Runs as a daemon thread inside the ML Service.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict

from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import ZeninDbConnection
from iot_machine_learning.ml_service.api.services.analysis.document_analyzer_factory import create_document_analyzer

from iot_machine_learning.ml_service.workers import queue_repository as repo
from iot_machine_learning.ml_service.workers.job_processor import parse_queue_row, build_payload
from iot_machine_learning.ml_service.workers.result_writer import save_to_weaviate, write_result, resolve_weaviate_url

logger = logging.getLogger(__name__)


class ZeninQueuePoller:
    """Polls zenin_docs.ingestion_queue and processes documents."""

    def __init__(self) -> None:
        self.poll_interval = int(
            os.environ.get("ZENIN_QUEUE_POLL_INTERVAL_SECONDS", "30")
        )
        self.batch_size = int(
            os.environ.get("ZENIN_QUEUE_BATCH_SIZE", "1")
        )
        
        # Load feature flags explicitly for daemon thread context
        feature_flags = None
        try:
            from iot_machine_learning.ml_service.config.feature_flags import get_feature_flags
            from iot_machine_learning.ml_service.config.loader import reset_feature_flags
            # Reset singleton to force reload from env vars in this thread
            reset_feature_flags()
            feature_flags = get_feature_flags()
            logger.info(
                "[ZENIN_POLLER] Feature flags loaded: decision_enabled=%s, strategy=%s",
                feature_flags.ML_ENABLE_DECISION_ENGINE,
                feature_flags.ML_DECISION_ENGINE_STRATEGY,
            )
        except Exception as e:
            logger.warning(f"[ZENIN_POLLER] Could not load feature flags: {e}")
        
        self.document_analyzer = create_document_analyzer(feature_flags=feature_flags)
        self._weaviate_url = resolve_weaviate_url()

        # Stats
        self._total_processed = 0
        self._total_errors = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Main polling loop (blocking — run in a daemon thread)."""
        logger.info(
            "[ZENIN_POLLER] Starting (interval=%ds, batch=%d)",
            self.poll_interval,
            self.batch_size,
        )

        while True:
            try:
                self._poll_cycle()
            except Exception:
                logger.exception("[ZENIN_POLLER] Error in poll cycle")
                self._total_errors += 1

            time.sleep(self.poll_interval)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_processed": self._total_processed,
            "total_errors": self._total_errors,
            "poll_interval": self.poll_interval,
        }

    # ------------------------------------------------------------------
    # Poll cycle
    # ------------------------------------------------------------------

    def _poll_cycle(self) -> None:
        """Fetch pending items and process each one."""
        logger.debug("[ZENIN_POLLER] Polling zenin_docs.ingestion_queue...")
        with ZeninDbConnection.get_connection() as conn:
            rows = repo.fetch_pending(conn, self.batch_size)

            if not rows:
                logger.debug("[ZENIN_POLLER] No pending items")
                return

            logger.info("[ZENIN_POLLER] Found %d pending item(s)", len(rows))

            for row in rows:
                queue_id = row[0]
                try:
                    self._process_item(conn, row)
                    self._total_processed += 1
                except Exception:
                    logger.exception(
                        "[ZENIN_POLLER] Failed to process %s", queue_id
                    )
                    self._total_errors += 1
                    repo.mark_error(conn, queue_id)

    # ------------------------------------------------------------------
    # Process single item
    # ------------------------------------------------------------------

    def _process_item(self, conn, row) -> None:
        """Parse → lock → build payload → analyze → persist."""
        item = parse_queue_row(row)

        # 1. Optimistic lock
        affected = repo.mark_processing(conn, item.queue_id)
        if affected == 0:
            logger.warning(
                "[ZENIN_POLLER] Item %s already claimed, skipping",
                item.queue_id,
            )
            return

        logger.info(
            "[ZENIN_POLLER] Processing %s (%s, %s)",
            item.queue_id, item.content_type, item.filename,
        )

        # 2. Build payload + analyze
        normalized_payload = build_payload(item)
        
        # DEBUG: Log payload before analysis
        logger.info(f"[ZENIN_POLLER] Built payload for {item.queue_id}: payload_keys={list(normalized_payload.keys())}")
        if "data" in normalized_payload:
            data = normalized_payload["data"]
            logger.info(f"[ZENIN_POLLER] Payload data keys: {list(data.keys())}")
            if "full_text" in data:
                full_text = data["full_text"]
                logger.info(f"[ZENIN_POLLER] Full text length: {len(full_text)}, preview: {full_text[:100]!r}")
            else:
                logger.warning(f"[ZENIN_POLLER] No full_text in payload data! Available keys: {list(data.keys())}")
        else:
            logger.warning(f"[ZENIN_POLLER] No 'data' key in payload! Available keys: {list(normalized_payload.keys())}")

        # Inject semantic context for text analysis enrichment
        normalized_payload["_weaviate_url"] = self._weaviate_url
        normalized_payload["_tenant_id"] = str(item.tenant_id)
        normalized_payload["_analysis_id"] = str(
            item.metadata.get("analysis_result_id", item.queue_id)
        )
        normalized_payload["_filename"] = item.filename

        analysis = self.document_analyzer.analyze(
            document_id=str(item.queue_id),
            content_type=item.content_type,
            normalized_payload=normalized_payload,
        )

        # 3. Weaviate (optional, fire-and-forget)
        ml_doc_id = save_to_weaviate(
            self._weaviate_url,
            tenant_id=str(item.tenant_id),
            filename=item.filename,
            content=item.content,
            analysis=analysis,
        )

        # 4. Persist result + mark completed
        write_result(
            conn,
            queue_id=item.queue_id,
            tenant_id=item.tenant_id,
            user_id=item.user_id,
            filename=item.filename,
            file_extension=item.file_extension,
            content_type=item.content_type,
            metadata=item.metadata,
            analysis=analysis,
            ml_doc_id=ml_doc_id,
        )
