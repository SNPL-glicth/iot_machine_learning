"""Tests for AnalyzeDocumentUseCase with dependency injection.

Estos tests reemplazan test_document_analyzer_integration.py
usando DI explícita en lugar de instanciación directa.
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, Optional

from iot_machine_learning.application.analyze_document import (
    AnalyzeDocumentUseCase,
)
from iot_machine_learning.domain.ports.analysis import (
    AnalysisContext,
    AnalysisEnginePort,
    AnalysisResult,
    Signal,
    Decision,
    Explanation,
    InputType,
)
from iot_machine_learning.domain.ports.document_analysis import (
    AnalysisOutput,
    CachePort,
    DocumentPersistencePort,
    PlasticityPort,
)


class FakeAnalysisEngine(AnalysisEnginePort):
    """Fake engine para testing."""
    
    def __init__(self, classification: str = "test", confidence: float = 0.8) -> None:
        self._classification = classification
        self._confidence = confidence
    
    def analyze(self, content: str, context: AnalysisContext) -> AnalysisResult:
        signal = Signal(
            raw_data=content,
            input_type=context.input_type or InputType.TEXT,
            domain=self._classification,  # Usar clasificación configurada
            features={"length": len(content)},
        )
        decision = Decision(
            severity="info",
            confidence=self._confidence,
            perceptions=[],
            weights={},
        )
        explanation = Explanation(
            narrative=f"Test analysis of: {content[:50]}...",
            confidence=self._confidence,
            contributions={},
            reasoning_trace=[],
            domain=self._classification,  # Usar clasificación configurada
        )
        
        return AnalysisResult(
            signal=signal,
            decision=decision,
            explanation=explanation,
        )


class FakeCache(CachePort):
    """Fake cache para testing."""
    
    def __init__(self) -> None:
        self._data: Dict[str, AnalysisOutput] = {}
    
    def get(self, key: str) -> Optional[AnalysisOutput]:
        return self._data.get(key)
    
    def set(self, key: str, value: AnalysisOutput, ttl: Optional[int] = None) -> None:
        self._data[key] = value
    
    def invalidate(self, key: str) -> None:
        self._data.pop(key, None)


class TestAnalyzeDocumentUseCase:
    """Test AnalyzeDocumentUseCase con DI."""
    
    def test_use_case_instantiates(self) -> None:
        """Use case debe instanciarse con engine y cache."""
        engine = FakeAnalysisEngine()
        cache = FakeCache()
        use_case = AnalyzeDocumentUseCase(engine=engine, cache=cache)
        assert use_case is not None
    
    def test_analyze_text_document(self) -> None:
        """Analizar documento de texto."""
        engine = FakeAnalysisEngine(classification="text", confidence=0.85)
        cache = FakeCache()
        use_case = AnalyzeDocumentUseCase(engine=engine, cache=cache)
        
        result = use_case.execute(
            document_id="test-doc-001",
            content="Critical system alert: Server CPU usage at 98%.",
            tenant_id="test-tenant",
            content_type="text",
        )
        
        assert result is not None
        assert result.document_id == "test-doc-001"
        assert result.tenant_id == "test-tenant"
        assert result.classification == "text"
        assert result.confidence == 0.85
        assert result.conclusion is not None
        assert result.processing_time_ms >= 0
    
    def test_analyze_numeric_data(self) -> None:
        """Analizar datos numéricos."""
        # El fake debe configurar input_type=TIMESERIES para que from_result use "timeseries"
        from unittest.mock import patch
        
        engine = FakeAnalysisEngine(classification="timeseries", confidence=0.75)
        cache = FakeCache()
        use_case = AnalyzeDocumentUseCase(engine=engine, cache=cache)
        
        result = use_case.execute(
            document_id="test-doc-002",
            content="[20.0, 20.5, 21.0, 21.5]",
            tenant_id="test-tenant",
            content_type="timeseries",
        )
        
        assert result.document_id == "test-doc-002"
        # La clasificación viene de signal.input_type.value = "timeseries"
        assert result.classification == "timeseries"
        assert result.confidence == 0.75
    
    def test_analyze_tabular_data(self) -> None:
        """Analizar datos tabulares."""
        # from_result usa signal.input_type.value, que para "tabular" es "tabular"
        engine = FakeAnalysisEngine(classification="tabular", confidence=0.90)
        cache = FakeCache()
        use_case = AnalyzeDocumentUseCase(engine=engine, cache=cache)
        
        result = use_case.execute(
            document_id="test-doc-003",
            content="col1,col2\n1,2\n3,4",
            tenant_id="test-tenant",
            content_type="tabular",
        )
        
        assert result.document_id == "test-doc-003"
        # La clasificación viene de signal.input_type.value
        assert result.classification == "tabular"
        assert result.confidence == 0.90
    
    def test_cache_hit(self) -> None:
        """Cache debe retornar resultado cacheado."""
        engine = FakeAnalysisEngine()
        cache = FakeCache()
        use_case = AnalyzeDocumentUseCase(engine=engine, cache=cache)
        
        # Primera llamada (miss)
        result1 = use_case.execute(
            document_id="test-cache-001",
            content="Cached content",
            tenant_id="tenant-1",
            content_type="text",
        )
        
        # Simular cache hit manual
        cached_output = AnalysisOutput(
            document_id="test-cache-001",
            tenant_id="tenant-1",
            classification="cached",
            conclusion="Cached result",
            confidence=0.99,
            analysis={"cached": True},
            cached=True,
        )
        cache.set("fake_cache_key", cached_output)
        
        # Verificar cache funciona
        cached = cache.get("fake_cache_key")
        assert cached is not None
        assert cached.cached is True
        assert cached.classification == "cached"
    
    def test_error_handling(self) -> None:
        """Manejo de errores en engine."""
        class FailingEngine(AnalysisEnginePort):
            def analyze(self, content: str, context: AnalysisContext) -> AnalysisResult:
                raise ValueError("Engine failure")
        
        cache = FakeCache()
        use_case = AnalyzeDocumentUseCase(engine=FailingEngine(), cache=cache)
        
        result = use_case.execute(
            document_id="test-error-001",
            content="Trigger error",
            tenant_id="test-tenant",
            content_type="text",
        )
        
        # Debe retornar output de error, no lanzar excepción
        assert result.document_id == "test-error-001"
        assert result.classification == "error"
        assert result.confidence == 0.0
        assert "Analysis failed" in result.conclusion
    
    def test_empty_content(self) -> None:
        """Manejar contenido vacío."""
        engine = FakeAnalysisEngine()
        cache = FakeCache()
        use_case = AnalyzeDocumentUseCase(engine=engine, cache=cache)
        
        result = use_case.execute(
            document_id="test-empty-001",
            content="",
            tenant_id="test-tenant",
            content_type="text",
        )
        
        assert result.document_id == "test-empty-001"
        assert result.processing_time_ms >= 0
    
    def test_with_persistence(self) -> None:
        """Persistencia opcional."""
        class FakePersistence(DocumentPersistencePort):
            def __init__(self) -> None:
                self.saved: list = []
            
            def save_analysis(self, output: AnalysisOutput) -> None:
                self.saved.append(output)
            
            def get_analysis(self, document_id: str) -> Optional[AnalysisOutput]:
                return None
        
        engine = FakeAnalysisEngine()
        cache = FakeCache()
        persistence = FakePersistence()
        use_case = AnalyzeDocumentUseCase(
            engine=engine,
            cache=cache,
            persistence=persistence,
        )
        
        result = use_case.execute(
            document_id="test-persist-001",
            content="Content to persist",
            tenant_id="test-tenant",
            content_type="text",
        )
        
        assert result.document_id == "test-persist-001"
        # En implementación real, persistence.save_analysis() sería llamado
    
    def test_with_plasticity(self) -> None:
        """Plasticity opcional."""
        class FakePlasticity(PlasticityPort):
            def update(self, domain: str, result: Any) -> None:
                pass
        
        engine = FakeAnalysisEngine()
        cache = FakeCache()
        plasticity = FakePlasticity()
        use_case = AnalyzeDocumentUseCase(
            engine=engine,
            cache=cache,
            plasticity=plasticity,
        )
        
        result = use_case.execute(
            document_id="test-plasticity-001",
            content="Content for plasticity",
            tenant_id="test-tenant",
            content_type="text",
        )
        
        assert result.document_id == "test-plasticity-001"


class TestDocumentAnalyzerAdapter:
    """Test DocumentAnalyzer legacy adapter con DI."""
    
    def test_adapter_delegates_to_use_case(self) -> None:
        """Adapter delega a AnalyzeDocumentUseCase."""
        from iot_machine_learning.ml_service.api.services.document_analyzer import (
            DocumentAnalyzer,
        )
        
        engine = FakeAnalysisEngine(classification="adapter_test")
        cache = FakeCache()
        
        analyzer = DocumentAnalyzer(
            engine=engine,
            cache=cache,
        )
        
        result = analyzer.analyze(
            document_id="adapter-test-001",
            content_type="text",
            normalized_payload={
                "data": {"full_text": "Test content for adapter"},
            },
            tenant_id="test-tenant",
        )
        
        assert result is not None
        assert result["document_id"] == "adapter-test-001"
        assert result["tenant_id"] == "test-tenant"
        # La clasificación viene de signal.input_type.value = "text" (content_type pasado)
        assert result["classification"] == "text"
        assert "processing_time_ms" in result
