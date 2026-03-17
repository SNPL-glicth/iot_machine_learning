"""Document Processor Runner.

Async worker que hace polling de zenin_docs.documents con status='pending',
procesa con ML, y actualiza resultados en BD.
NO usa HTTP callbacks.
"""

import logging
import time
import json
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml_service.api.services.document_analyzer import DocumentAnalyzer
from iot_ingest_services.common.db import get_engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Async document processor that polls from DB."""
    
    def __init__(self, poll_interval_seconds: int = 5):
        """Initialize processor.
        
        Args:
            poll_interval_seconds: Seconds to wait between polls
        """
        self.poll_interval = poll_interval_seconds
        self.analyzer = DocumentAnalyzer()
        self.engine = get_engine()
        self.running = False
    
    def start(self):
        """Start the polling loop."""
        self.running = True
        logger.info("[DOCUMENT-PROCESSOR] Starting...")
        
        while self.running:
            try:
                self._process_batch()
                time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                logger.info("[DOCUMENT-PROCESSOR] Interrupted by user")
                break
            except Exception as e:
                logger.exception(f"[DOCUMENT-PROCESSOR] Error in polling loop: {e}")
                time.sleep(self.poll_interval)
    
    def stop(self):
        """Stop the polling loop."""
        self.running = False
        logger.info("[DOCUMENT-PROCESSOR] Stopping...")
    
    def _process_batch(self):
        """Process one batch of pending documents."""
        with self.engine.connect() as conn:
            # Get pending documents (limit 10 per batch)
            result = conn.execute(text("""
                SELECT id, content_type, normalized_payload
                FROM zenin_docs.documents
                WHERE status = 'pending'
                ORDER BY uploaded_at ASC
                LIMIT 10
                FOR UPDATE SKIP LOCKED
            """))
            
            documents = result.fetchall()
            
            if not documents:
                return
            
            logger.info(f"[DOCUMENT-PROCESSOR] Processing {len(documents)} documents")
            
            for doc in documents:
                try:
                    self._process_document(conn, doc)
                except Exception as e:
                    logger.exception(f"[DOCUMENT-PROCESSOR] Error processing document {doc.id}: {e}")
                    self._mark_error(conn, doc.id, str(e))
            
            conn.commit()
    
    def _process_document(self, conn, doc):
        """Process a single document.
        
        Args:
            conn: SQLAlchemy connection
            doc: Document row from DB
        """
        document_id = str(doc.id)
        content_type = doc.content_type
        normalized_payload = json.loads(doc.normalized_payload) if doc.normalized_payload else {}
        
        logger.info(f"[DOCUMENT-PROCESSOR] Analyzing document {document_id} ({content_type})")
        
        # Mark as processing
        conn.execute(text("""
            UPDATE zenin_docs.documents
            SET status = 'processing'
            WHERE id = :id
        """), {"id": document_id})
        conn.commit()
        
        # Analyze with ML
        result = self.analyzer.analyze(
            document_id=document_id,
            content_type=content_type,
            normalized_payload=normalized_payload,
        )
        
        # Update with results
        conn.execute(text("""
            UPDATE zenin_docs.documents
            SET status = 'analyzed',
                ml_result = :ml_result,
                conclusion = :conclusion,
                analyzed_at = now()
            WHERE id = :id
        """), {
            "id": document_id,
            "ml_result": json.dumps(result),
            "conclusion": result.get("conclusion", ""),
        })
        
        logger.info(f"[DOCUMENT-PROCESSOR] Document {document_id} analyzed successfully")
    
    def _mark_error(self, conn, document_id: str, error_message: str):
        """Mark document as error.
        
        Args:
            conn: SQLAlchemy connection
            document_id: UUID of document
            error_message: Error message
        """
        conn.execute(text("""
            UPDATE zenin_docs.documents
            SET status = 'error',
                error_message = :error_message
            WHERE id = :id
        """), {
            "id": document_id,
            "error_message": error_message,
        })
        conn.commit()
        logger.warning(f"[DOCUMENT-PROCESSOR] Document {document_id} marked as error: {error_message}")


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = DocumentProcessor(poll_interval_seconds=5)
    
    try:
        processor.start()
    except KeyboardInterrupt:
        processor.stop()


if __name__ == "__main__":
    main()
