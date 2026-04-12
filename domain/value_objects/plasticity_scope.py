"""PlasticityScope — value object for namespaced plasticity isolation.

Prevents cross-contamination between different domains/series types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PlasticityScope:
    """Scope for isolating plasticity weights between different contexts.
    
    Without scope: all series share the same plasticity weights for a regime.
    With scope: weights are namespaced by domain+regime.
    
    Examples:
        >>> scope = PlasticityScope(domain="iot", regime="STABLE")
        >>> scope.redis_key
        'plasticity:iot:STABLE'
        >>> scope.sql_scope
        'iot'
        
        >>> default = PlasticityScope(regime="STABLE")  # no domain
        >>> default.redis_key
        'plasticity:STABLE'
    """
    
    regime: str
    domain: Optional[str] = None
    
    def __post_init__(self) -> None:
        # Validate regime is non-empty
        if not self.regime or not self.regime.strip():
            raise ValueError("regime cannot be empty")
    
    @property
    def redis_key(self) -> str:
        """Redis key for this scoped plasticity.
        
        Format: plasticity:{domain}:{regime} or plasticity:{regime}
        """
        if self.domain:
            return f"plasticity:{self.domain}:{self.regime}"
        return f"plasticity:{self.regime}"
    
    @property
    def sql_scope(self) -> str:
        """SQL scope identifier for filtering queries.
        
        Returns domain if set, otherwise empty string.
        """
        return self.domain or ""
    
    @property
    def is_default(self) -> bool:
        """True if this is the default (unscoped) plasticity."""
        return self.domain is None or self.domain == ""
    
    def with_regime(self, new_regime: str) -> PlasticityScope:
        """Create a new scope with different regime but same domain."""
        return PlasticityScope(regime=new_regime, domain=self.domain)
    
    def __str__(self) -> str:
        if self.domain:
            return f"{self.domain}:{self.regime}"
        return self.regime
