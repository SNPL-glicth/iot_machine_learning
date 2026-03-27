"""DecisionEnginePort — abstract interface for decision strategies.

Domain port defining the contract for all decision engine implementations.
Follows the Strategy pattern: implementations decide how to select actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ...domain.entities.decision import Decision, DecisionContext


class DecisionEnginePort(ABC):
    """Abstract port for decision engines.

    All decision strategies (conservative, aggressive, cost-optimized)
    implement this interface. The domain layer depends on this port,
    not on concrete implementations.

    Implementations must be:
    - Deterministic: same input → same output
    - Stateless: no internal state between calls
    - Fail-safe: never raise exceptions (return Decision.noop() on failure)
    - Pure: no side effects, no I/O

    Usage:
        engine = SimpleDecisionEngine()  # concrete impl
        context = DecisionContext(severity=..., ...)
        decision = engine.decide(context)
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Unique identifier for this decision strategy.

        Returns:
            Strategy name (e.g., "conservative", "aggressive", "simple")
        """
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version of this decision engine implementation.

        Used for audit trails and A/B testing.

        Returns:
            Semantic version string (e.g., "1.0.0")
        """
        ...

    @abstractmethod
    def decide(self, context: DecisionContext) -> Decision:
        """Make a decision based on the provided context.

        Core method of the decision engine. Takes aggregated ML outputs
        and produces a recommended action with full explainability.

        Args:
            context: Aggregated ML outputs (severity, patterns, confidence, etc.)

        Returns:
            Decision object with action, priority, confidence, and reason.
            Never returns None — returns Decision.noop() if unable to decide.
        """
        ...

    @abstractmethod
    def can_decide(self, context: DecisionContext) -> bool:
        """Check if this engine can make a decision for the given context.

        Validates that all required fields are present and within
        acceptable ranges. Used to determine if fallback is needed.

        Args:
            context: Decision context to validate

        Returns:
            True if decide() can be called safely, False otherwise.
        """
        ...

    def decide_safe(
        self,
        context: DecisionContext,
        fallback_reason: str = "Decision engine unable to process context",
    ) -> Decision:
        """Fail-safe wrapper around decide().

        Returns a valid Decision even if decide() fails or context is invalid.
        This is the method called by the application layer.

        Args:
            context: Decision context
            fallback_reason: Reason to use if returning noop decision

        Returns:
            Valid Decision (either from decide() or Decision.noop())
        """
        try:
            if not self.can_decide(context):
                return Decision.noop(
                    series_id=context.series_id,
                    reason=f"{fallback_reason}: context validation failed",
                )
            return self.decide(context)
        except Exception:
            # Fail-safe: return noop decision on any exception
            return Decision.noop(
                series_id=context.series_id,
                reason=fallback_reason,
            )


class NullDecisionEngine(DecisionEnginePort):
    """Null object pattern implementation.

    Returns noop decisions for all contexts. Used when decision engine
    is disabled or as a fallback when no strategy is configured.
    """

    @property
    def strategy_name(self) -> str:
        return "null"

    @property
    def version(self) -> str:
        return "1.0.0"

    def decide(self, context: DecisionContext) -> Decision:
        return Decision.noop(
            series_id=context.series_id,
            reason="Decision engine disabled (NullDecisionEngine)",
        )

    def can_decide(self, context: DecisionContext) -> bool:
        return True  # Always can "decide" (returns noop)


class DecisionEngineRegistry:
    """Registry for decision engine implementations.

    Provides factory access to available strategies without
    hard-coding implementations in domain layer.
    """

    _engines: Dict[str, DecisionEnginePort] = {}

    @classmethod
    def register(cls, engine: DecisionEnginePort) -> None:
        """Register a decision engine implementation."""
        cls._engines[engine.strategy_name] = engine

    @classmethod
    def get(cls, name: str) -> Optional[DecisionEnginePort]:
        """Get a registered engine by name."""
        return cls._engines.get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List available strategy names."""
        return list(cls._engines.keys())

    @classmethod
    def create_default(cls) -> DecisionEnginePort:
        """Get the default engine (null if none registered)."""
        # Prefer "simple" > "conservative" > first available > null
        for preferred in ["simple", "conservative"]:
            if preferred in cls._engines:
                return cls._engines[preferred]
        if cls._engines:
            return next(iter(cls._engines.values()))
        return NullDecisionEngine()
