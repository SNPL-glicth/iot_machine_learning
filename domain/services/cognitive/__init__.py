"""Cognitive domain services (narrative, memory, context)."""
try:
    from .narrative_unifier import NarrativeUnifier
except ImportError:
    NarrativeUnifier = None  # type: ignore[assignment,misc]

try:
    from .memory_recall_enricher import MemoryRecallEnricher
except ImportError:
    MemoryRecallEnricher = None  # type: ignore[assignment,misc]

try:
    from .conclusion_formatter import format_conclusion
except ImportError:
    format_conclusion = None  # type: ignore[assignment,misc]

try:
    from .chat_context_manager import ChatContextManager
except ImportError:
    ChatContextManager = None  # type: ignore[assignment,misc]

try:
    from .interaction_field_service import InteractionFieldService
except ImportError:
    InteractionFieldService = None  # type: ignore[assignment,misc]

try:
    from .situation_vector_builder import SituationVectorBuilder
except ImportError:
    SituationVectorBuilder = None  # type: ignore[assignment,misc]

try:
    from .plasticity_feedback import update_plasticity_from_result
except ImportError:
    update_plasticity_from_result = None  # type: ignore[assignment,misc]

__all__ = [
    "NarrativeUnifier", "MemoryRecallEnricher", "format_conclusion",
    "ChatContextManager", "InteractionFieldService",
    "SituationVectorBuilder", "update_plasticity_from_result",
]
