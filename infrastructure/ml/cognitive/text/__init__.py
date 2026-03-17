"""TextCognitiveEngine — deep analysis engine for text.

Reusable cognitive engine at the same layer as ``MetaCognitiveOrchestrator``.

Subpackage structure:
    types.py                 — TextAnalysisContext, TextAnalysisInput, TextCognitiveResult
    signal_profiler.py       — Text metrics → SignalSnapshot
    perception_collector.py  — Pre-computed scores → EnginePerception[]
    severity_mapper.py       — Urgency/sentiment → SeverityResult
    memory_enricher.py       — CognitiveMemoryPort → TextRecallContext
    explanation_assembler.py — Builds Explanation domain object
    engine.py                — TextCognitiveEngine (main orchestrator)
"""

from .engine import TextCognitiveEngine
from .types import TextAnalysisContext, TextAnalysisInput, TextCognitiveResult

__all__ = [
    "TextAnalysisContext",
    "TextAnalysisInput",
    "TextCognitiveEngine",
    "TextCognitiveResult",
]
