"""HTTP client primitives for Weaviate REST API.

Uses stdlib urllib.request — no SDK dependency.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def post_json(
    url: str,
    payload: Dict[str, Any],
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:
    """POST JSON to Weaviate. Returns parsed response or None on error.
    
    Args:
        url: Full URL to POST to
        payload: JSON-serializable dict
        timeout: Request timeout in seconds
        
    Returns:
        Parsed JSON response dict, or None on error
    """
    body = json.dumps(payload, default=str).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        logger.warning(
            "weaviate_http_error",
            extra={"status": exc.code, "url": url, "body": error_body[:500]},
        )
        return None
    except Exception as exc:
        logger.warning(
            "weaviate_request_error",
            extra={"url": url, "error": str(exc)},
        )
        return None
