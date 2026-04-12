"""Batch operations for Weaviate to reduce HTTP overhead.

Replaces N individual HTTP calls with a single batch insert.
Significantly reduces latency when storing multiple objects.

Usage:
    batch = WeaviateBatch(objects_url, batch_size=100)
    batch.add_object("MLExplanation", properties1)
    batch.add_object("MLAnomaly", properties2)
    batch.flush()  # Sends accumulated objects
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .http_client import post_json

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 100
_BATCH_ENDPOINT = "/v1/batch/objects"


@dataclass
class BatchResult:
    """Result of a batch operation."""
    
    total: int = 0
    successful: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)
    uuids: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.successful / self.total * 100) if self.total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 2),
            "error_count": len(self.errors),
        }


class WeaviateBatch:
    """Accumulates Weaviate objects and sends them in batches.
    
    Reduces HTTP overhead by sending multiple objects in a single request.
    Automatically flushes when batch_size is reached.
    
    Args:
        base_url: Weaviate base URL (e.g. http://localhost:8080)
        batch_size: Number of objects to accumulate before auto-flush
        enabled: Master switch
        dry_run: If True, logs but doesn't send
        timeout: HTTP timeout in seconds
        
    Example:
        batch = WeaviateBatch("http://localhost:8080", batch_size=50)
        for item in items:
            batch.add_object("MLExplanation", item.to_dict())
        batch.flush()  # Send remaining objects
    """
    
    def __init__(
        self,
        base_url: str,
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        enabled: bool = True,
        dry_run: bool = False,
        timeout: int = 10,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._batch_url = f"{self._base_url}{_BATCH_ENDPOINT}"
        self._batch_size = batch_size
        self._enabled = enabled
        self._dry_run = dry_run
        self._timeout = timeout
        
        self._objects: List[Dict[str, Any]] = []
        self._total_sent = 0
        self._total_successful = 0
        self._total_failed = 0
    
    def add_object(
        self,
        class_name: str,
        properties: Dict[str, Any],
        *,
        uuid: Optional[str] = None,
    ) -> None:
        """Add an object to the batch.
        
        Automatically flushes if batch_size is reached.
        
        Args:
            class_name: Weaviate class name
            properties: Object properties
            uuid: Optional UUID (Weaviate will generate if not provided)
        """
        if not self._enabled:
            return
        
        obj = {
            "class": class_name,
            "properties": properties,
        }
        
        if uuid:
            obj["id"] = uuid
        
        self._objects.append(obj)
        
        # Auto-flush when batch size reached
        if len(self._objects) >= self._batch_size:
            self.flush()
    
    def flush(self) -> BatchResult:
        """Send all accumulated objects to Weaviate.
        
        Returns:
            BatchResult with statistics
        """
        if not self._objects:
            return BatchResult()
        
        if not self._enabled:
            self._objects.clear()
            return BatchResult()
        
        batch_count = len(self._objects)
        
        if self._dry_run:
            logger.info(
                "weaviate_batch_dry_run",
                extra={
                    "batch_size": batch_count,
                    "classes": self._get_class_distribution(),
                },
            )
            result = BatchResult(
                total=batch_count,
                successful=batch_count,
                uuids=["dry-run-uuid"] * batch_count,
            )
            self._objects.clear()
            return result
        
        # Send batch request
        payload = {"objects": self._objects}
        
        try:
            resp = post_json(self._batch_url, payload, timeout=self._timeout)
            result = self._process_batch_response(resp, batch_count)
            
            # Update totals
            self._total_sent += result.total
            self._total_successful += result.successful
            self._total_failed += result.failed
            
            logger.info(
                "weaviate_batch_sent",
                extra={
                    **result.to_dict(),
                    "classes": self._get_class_distribution(),
                },
            )
            
            return result
            
        except Exception as exc:
            logger.error(
                "weaviate_batch_error",
                extra={
                    "error": str(exc),
                    "batch_size": batch_count,
                },
            )
            return BatchResult(
                total=batch_count,
                failed=batch_count,
                errors=[str(exc)],
            )
        finally:
            self._objects.clear()
    
    def _process_batch_response(
        self,
        response: Optional[Dict[str, Any]],
        expected_count: int,
    ) -> BatchResult:
        """Process Weaviate batch response.
        
        Args:
            response: Response from Weaviate batch endpoint
            expected_count: Number of objects sent
            
        Returns:
            BatchResult with statistics
        """
        if not response:
            return BatchResult(
                total=expected_count,
                failed=expected_count,
                errors=["Empty response from Weaviate"],
            )
        
        result = BatchResult(total=expected_count)
        
        # Weaviate batch response format:
        # [{"result": {"status": "SUCCESS", ...}, "id": "uuid"}, ...]
        results_list = response if isinstance(response, list) else []
        
        for item in results_list:
            item_result = item.get("result", {})
            status = item_result.get("status", "UNKNOWN")
            uuid = item.get("id")
            
            if status == "SUCCESS" and uuid:
                result.successful += 1
                result.uuids.append(uuid)
            else:
                result.failed += 1
                error_msg = item_result.get("errors", {}).get("error", [{}])[0].get("message", "Unknown error")
                result.errors.append(error_msg)
        
        # Handle mismatch between sent and received
        if result.successful + result.failed != expected_count:
            logger.warning(
                "weaviate_batch_count_mismatch",
                extra={
                    "sent": expected_count,
                    "received": result.successful + result.failed,
                },
            )
        
        return result
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in current batch."""
        distribution: Dict[str, int] = {}
        for obj in self._objects:
            class_name = obj.get("class", "Unknown")
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    @property
    def pending_count(self) -> int:
        """Number of objects waiting to be flushed."""
        return len(self._objects)
    
    @property
    def total_sent(self) -> int:
        """Total objects sent across all flushes."""
        return self._total_sent
    
    @property
    def total_successful(self) -> int:
        """Total successful inserts across all flushes."""
        return self._total_successful
    
    @property
    def total_failed(self) -> int:
        """Total failed inserts across all flushes."""
        return self._total_failed
    
    @property
    def overall_success_rate(self) -> float:
        """Overall success rate across all flushes."""
        return (self._total_successful / self._total_sent * 100) if self._total_sent > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return {
            "pending": self.pending_count,
            "total_sent": self.total_sent,
            "total_successful": self.total_successful,
            "total_failed": self.total_failed,
            "success_rate": round(self.overall_success_rate, 2),
        }
    
    def __enter__(self) -> WeaviateBatch:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - auto-flush remaining objects."""
        if self._objects:
            self.flush()
