"""Timeout utilities for pipeline features."""

import concurrent.futures
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def run_with_timeout(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout_seconds: float = 3.0,
    feature_name: str = "feature",
) -> Any:
    """Run a function with a timeout, returning None if timeout exceeded.
    
    Args:
        fn: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        timeout_seconds: Maximum time to wait
        feature_name: Name for logging
        
    Returns:
        Function result or None if timeout/failure
    """
    if kwargs is None:
        kwargs = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.warning(f"feature_timeout: {feature_name} exceeded {timeout_seconds}s")
            return None
        except Exception as e:
            logger.warning(f"feature_error: {feature_name} failed: {e}")
            return None
