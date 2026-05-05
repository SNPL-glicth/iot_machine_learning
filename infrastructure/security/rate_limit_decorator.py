"""Rate Limit Decorator — easy integration for functions and methods.

Provides decorators for applying rate limits to functions without
modifying their signatures.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, TypeVar

from .rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitScope,
    RateLimitExceeded,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


def rate_limit(
    limiter: RateLimiter,
    scope: RateLimitScope,
    identifier_fn: Optional[Callable[..., str]] = None,
    config: Optional[RateLimitConfig] = None,
    raise_on_limit: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to apply rate limiting to a function.
    
    Args:
        limiter: RateLimiter instance
        scope: Rate limit scope
        identifier_fn: Function to extract identifier from args/kwargs
        config: Optional rate limit config
        raise_on_limit: If True, raise RateLimitExceeded; if False, return None
    
    Returns:
        Decorated function
    
    Example:
        >>> limiter = RateLimiter(redis_client, tenant_id="acme")
        >>> 
        >>> @rate_limit(
        ...     limiter=limiter,
        ...     scope=RateLimitScope.ENDPOINT,
        ...     identifier_fn=lambda *args, **kwargs: "predict",
        ...     config=RateLimitConfig(requests_per_second=10, burst_size=20),
        ... )
        >>> def predict(series_id: str, values: list):
        ...     # ... prediction logic
        ...     pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Extract identifier
            if identifier_fn:
                identifier = identifier_fn(*args, **kwargs)
            else:
                # Default: use function name
                identifier = func.__name__
            
            # Check rate limit
            result = limiter.check_rate_limit(
                scope=scope,
                identifier=identifier,
                config=config,
            )
            
            if not result.allowed:
                if raise_on_limit:
                    raise RateLimitExceeded(
                        result=result,
                        scope=scope,
                        identifier=identifier,
                    )
                else:
                    logger.warning(
                        f"rate_limit_exceeded_silent: {scope.value}:{identifier}"
                    )
                    return None
            
            # Execute function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit_tenant(
    limiter: RateLimiter,
    config: Optional[RateLimitConfig] = None,
    raise_on_limit: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for tenant-level rate limiting.
    
    Args:
        limiter: RateLimiter instance
        config: Optional rate limit config
        raise_on_limit: If True, raise RateLimitExceeded
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            result = limiter.check_tenant_limit(config=config)
            
            if not result.allowed:
                if raise_on_limit:
                    raise RateLimitExceeded(
                        result=result,
                        scope=RateLimitScope.TENANT,
                        identifier=limiter._tenant_id,
                    )
                else:
                    return None
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit_series(
    limiter: RateLimiter,
    series_id_arg: str = "series_id",
    config: Optional[RateLimitConfig] = None,
    raise_on_limit: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for series-level rate limiting.
    
    Args:
        limiter: RateLimiter instance
        series_id_arg: Name of series_id argument
        config: Optional rate limit config
        raise_on_limit: If True, raise RateLimitExceeded
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Extract series_id from kwargs or args
            series_id = kwargs.get(series_id_arg)
            
            if series_id is None:
                # Try to get from function signature
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                
                if series_id_arg in params:
                    idx = params.index(series_id_arg)
                    if idx < len(args):
                        series_id = args[idx]
            
            if series_id is None:
                logger.warning(
                    f"rate_limit_series: could not extract {series_id_arg}, skipping"
                )
                return func(*args, **kwargs)
            
            result = limiter.check_series_limit(
                series_id=series_id,
                config=config,
            )
            
            if not result.allowed:
                if raise_on_limit:
                    raise RateLimitExceeded(
                        result=result,
                        scope=RateLimitScope.SERIES,
                        identifier=series_id,
                    )
                else:
                    return None
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit_endpoint(
    limiter: RateLimiter,
    endpoint_name: str,
    config: Optional[RateLimitConfig] = None,
    raise_on_limit: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for endpoint-level rate limiting.
    
    Args:
        limiter: RateLimiter instance
        endpoint_name: Name of endpoint
        config: Optional rate limit config
        raise_on_limit: If True, raise RateLimitExceeded
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            result = limiter.check_endpoint_limit(
                endpoint=endpoint_name,
                config=config,
            )
            
            if not result.allowed:
                if raise_on_limit:
                    raise RateLimitExceeded(
                        result=result,
                        scope=RateLimitScope.ENDPOINT,
                        identifier=endpoint_name,
                    )
                else:
                    return None
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
