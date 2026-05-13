"""Tests de regresión para COG-CRIT-2, COG-SEV-1, COG-SEV-2, COG-SEV-3.

Tests con carga concurrente para validar comportamiento correcto.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception,
    InhibitionState,
)
from iot_machine_learning.infrastructure.ml.cognitive.decision.contextual_decision_engine import (
    _FLAG_CACHE,
    _get_flags,
)
from iot_machine_learning.infrastructure.ml.cognitive.decision.flag_cache import (
    FlagCache,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.hampel_fallback_strategy import (
    MedianClosestFallbackStrategy,
    BypassAllFallbackStrategy,
)


class TestCOGCRIT2HampelFallback:
    """Test COG-CRIT-2: Hampel fallback con mediana."""
    
    def test_median_closest_fallback_selects_closest_to_median(self):
        """MedianClosestFallbackStrategy selecciona engine más cercano a mediana."""
        strategy = MedianClosestFallbackStrategy()
        
        # Perceptions con valores dispersos
        perceptions = [
            EnginePerception(
                engine_name="engine_1",
                predicted_value=10.0,
                confidence=0.8,
                trend="stable",
                stability=0.1,
                local_fit_error=0.5,
            ),
            EnginePerception(
                engine_name="engine_2",
                predicted_value=50.0,  # Más cercano a mediana 45
                confidence=0.7,
                trend="stable",
                stability=0.2,
                local_fit_error=0.6,
            ),
            EnginePerception(
                engine_name="engine_3",
                predicted_value=100.0,
                confidence=0.9,
                trend="stable",
                stability=0.1,
                local_fit_error=0.4,
            ),
        ]
        
        inhibition_states = [
            InhibitionState(
                engine_name="engine_1",
                base_weight=0.33,
                inhibited_weight=0.33,
                inhibition_reason="none",
                suppression_factor=0.0,
            ),
            InhibitionState(
                engine_name="engine_2",
                base_weight=0.33,
                inhibited_weight=0.33,
                inhibition_reason="none",
                suppression_factor=0.0,
            ),
            InhibitionState(
                engine_name="engine_3",
                base_weight=0.34,
                inhibited_weight=0.34,
                inhibition_reason="none",
                suppression_factor=0.0,
            ),
        ]
        
        median = 45.0  # Mediana aproximada
        
        selected_perceptions, selected_states, reason = strategy.select_fallback(
            perceptions, inhibition_states, median
        )
        
        # Debe seleccionar engine_2 (50.0 es más cercano a 45.0)
        assert len(selected_perceptions) == 1
        assert selected_perceptions[0].engine_name == "engine_2"
        assert selected_perceptions[0].predicted_value == 50.0
        
        assert len(selected_states) == 1
        assert selected_states[0].engine_name == "engine_2"
        
        assert "hampel_all_rejected_using_median" in reason
        assert "engine_2" in reason
    
    def test_median_closest_fallback_concurrent_access(self):
        """MedianClosestFallbackStrategy es thread-safe."""
        strategy = MedianClosestFallbackStrategy()
        
        perceptions = [
            EnginePerception(
                engine_name=f"engine_{i}",
                predicted_value=float(i * 10),
                confidence=0.8,
                trend="stable",
                stability=0.1,
                local_fit_error=0.5,
            )
            for i in range(10)
        ]
        
        inhibition_states = [
            InhibitionState(
                engine_name=f"engine_{i}",
                base_weight=0.1,
                inhibited_weight=0.1,
                inhibition_reason="none",
                suppression_factor=0.0,
            )
            for i in range(10)
        ]
        
        median = 45.0
        results = []
        
        def select():
            result = strategy.select_fallback(perceptions, inhibition_states, median)
            results.append(result)
        
        # Ejecutar 100 veces en paralelo
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(select) for _ in range(100)]
            for future in futures:
                future.result()
        
        # Todos deben seleccionar el mismo engine
        assert len(results) == 100
        selected_engines = {r[0][0].engine_name for r in results}
        assert len(selected_engines) == 1  # Siempre el mismo
    
    def test_bypass_all_fallback_legacy_behavior(self):
        """BypassAllFallbackStrategy retorna todos los perceptions (legacy)."""
        strategy = BypassAllFallbackStrategy()
        
        perceptions = [
            EnginePerception(
                engine_name=f"engine_{i}",
                predicted_value=float(i * 10),
                confidence=0.8,
                trend="stable",
                stability=0.1,
                local_fit_error=0.5,
            )
            for i in range(3)
        ]
        
        inhibition_states = [
            InhibitionState(
                engine_name=f"engine_{i}",
                base_weight=0.33,
                inhibited_weight=0.33,
                inhibition_reason="none",
                suppression_factor=0.0,
            )
            for i in range(3)
        ]
        
        median = 10.0
        
        selected_perceptions, selected_states, reason = strategy.select_fallback(
            perceptions, inhibition_states, median
        )
        
        # Debe retornar TODOS
        assert len(selected_perceptions) == 3
        assert len(selected_states) == 3
        assert "bypassed" in reason


class TestCOGSEV1BudgetCheck:
    """Test COG-SEV-1: Budget check antes de ejecutar engines."""
    
    def test_budget_check_before_engine_execution(self):
        """Budget check ocurre ANTES de collect_perceptions()."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.predict_phase import (
            PredictPhase,
        )
        
        phase = PredictPhase()
        
        # Mock context con budget excedido
        ctx = Mock()
        ctx.series_id = "test_series"
        ctx.profile = None
        ctx.values = [1.0, 2.0, 3.0]
        ctx.timestamps = None
        ctx.orchestrator = Mock()
        ctx.orchestrator._engines = []
        
        # Budget ya excedido
        ctx.timer = Mock()
        ctx.timer.total_ms = 150.0
        ctx.timer.budget_ms = 100.0
        
        # Mock handle_fallback
        with patch(
            "iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.predict_phase.handle_fallback"
        ) as mock_fallback:
            mock_fallback.return_value = (None, None, None, None, None)
            
            result = phase.execute(ctx)
            
            # Debe llamar fallback sin ejecutar engines
            mock_fallback.assert_called_once()
            assert result.is_fallback
            assert result.fallback_reason == "budget_exceeded_before_predict"
            assert result.engine_failures == {}  # No engines ejecutados
    
    def test_budget_check_concurrent_execution(self):
        """Budget check es thread-safe bajo carga concurrente."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.predict_phase import (
            PredictPhase,
        )
        
        phase = PredictPhase()
        results = []
        
        def execute_phase():
            ctx = Mock()
            ctx.series_id = "test_series"
            ctx.profile = None
            ctx.values = [1.0, 2.0, 3.0]
            ctx.timestamps = None
            ctx.orchestrator = Mock()
            ctx.orchestrator._engines = []
            ctx.timer = Mock()
            ctx.timer.total_ms = 150.0
            ctx.timer.budget_ms = 100.0
            
            with patch(
                "iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.predict_phase.handle_fallback"
            ) as mock_fallback:
                mock_fallback.return_value = (None, None, None, None, None)
                result = phase.execute(ctx)
                results.append(result)
        
        # Ejecutar 50 veces en paralelo
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(execute_phase) for _ in range(50)]
            for future in futures:
                future.result()
        
        # Todos deben ser fallback
        assert len(results) == 50
        assert all(r.is_fallback for r in results)
        assert all(r.fallback_reason == "budget_exceeded_before_predict" for r in results)


class TestCOGSEV2SpatialCorrection:
    """Test COG-SEV-2: Spatial correction con validación de correlación."""
    
    def test_validate_correlation_quality_sufficient_samples(self):
        """Correlación significativa con suficientes muestras."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
            _validate_correlation_quality,
        )
        
        # Correlación alta (0.8) con 30 muestras → significativa
        assert _validate_correlation_quality(0.8, 30) is True
        
        # Correlación moderada (0.5) con 30 muestras → no significativa
        assert _validate_correlation_quality(0.5, 30) is False
        
        # Correlación alta (0.8) con 100 muestras → significativa
        assert _validate_correlation_quality(0.8, 100) is True
    
    def test_validate_correlation_quality_insufficient_samples(self):
        """Correlación rechazada si < 20 muestras."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
            _validate_correlation_quality,
        )
        
        # Incluso correlación perfecta rechazada con pocas muestras
        assert _validate_correlation_quality(0.99, 10) is False
        assert _validate_correlation_quality(0.99, 19) is False
        
        # Con 20+ muestras, correlación alta es aceptada
        assert _validate_correlation_quality(0.99, 20) is True
    
    def test_spatial_correction_filters_low_correlation(self):
        """Spatial correction filtra neighbors con correlación < 0.7."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
            _apply_spatial_correction,
        )
        
        base_prediction = 100.0
        
        # Neighbors con correlación baja (< 0.7)
        neighbors = [
            ("neighbor_1", 0.5),  # Rechazado: < 0.7
            ("neighbor_2", 0.6),  # Rechazado: < 0.7
        ]
        
        neighbor_values = {
            "neighbor_1": [90.0, 91.0, 92.0, 93.0, 94.0] * 6,  # 30 muestras
            "neighbor_2": [95.0, 96.0, 97.0, 98.0, 99.0] * 6,
        }
        
        # No debe aplicar corrección (todos rechazados)
        corrected = _apply_spatial_correction(base_prediction, neighbors, neighbor_values)
        assert corrected == base_prediction
    
    def test_spatial_correction_filters_insignificant_correlation(self):
        """Spatial correction filtra correlaciones no significativas."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
            _apply_spatial_correction,
        )
        
        base_prediction = 100.0
        
        # Neighbor con correlación alta pero pocas muestras
        neighbors = [
            ("neighbor_1", 0.8),  # Alta correlación
        ]
        
        neighbor_values = {
            "neighbor_1": [90.0, 91.0, 92.0],  # Solo 3 muestras → rechazado
        }
        
        # No debe aplicar corrección (rechazado por pocas muestras)
        corrected = _apply_spatial_correction(base_prediction, neighbors, neighbor_values)
        assert corrected == base_prediction
    
    def test_spatial_correction_accepts_valid_correlation(self):
        """Spatial correction acepta correlación válida (>0.7, significativa)."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
            _apply_spatial_correction,
        )
        
        base_prediction = 100.0
        
        # Neighbor con correlación alta y suficientes muestras
        neighbors = [
            ("neighbor_1", 0.85),  # Alta correlación
        ]
        
        # 30 muestras con gradiente positivo
        neighbor_values = {
            "neighbor_1": [float(i) for i in range(90, 120)],  # 30 muestras
        }
        
        # Debe aplicar corrección
        corrected = _apply_spatial_correction(base_prediction, neighbors, neighbor_values)
        
        # Corrección debe ser != base (se aplicó spatial correction)
        assert corrected != base_prediction
    
    def test_spatial_correction_concurrent_access(self):
        """Spatial correction es thread-safe."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
            _apply_spatial_correction,
        )
        
        base_prediction = 100.0
        neighbors = [("neighbor_1", 0.85)]
        neighbor_values = {"neighbor_1": [float(i) for i in range(90, 120)]}
        
        results = []
        
        def apply_correction():
            corrected = _apply_spatial_correction(base_prediction, neighbors, neighbor_values)
            results.append(corrected)
        
        # Ejecutar 50 veces en paralelo
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(apply_correction) for _ in range(50)]
            for future in futures:
                future.result()
        
        # Todos deben obtener el mismo resultado
        assert len(results) == 50
        assert len(set(results)) == 1  # Un solo valor único


class TestCOGSEV3FlagCache:
    """Test COG-SEV-3: FlagCache con TTL."""
    
    def test_flag_cache_basic_functionality(self):
        """FlagCache cachea y recarga según TTL."""
        cache = FlagCache(ttl_seconds=0.1)  # 100ms TTL
        
        load_count = [0]
        
        def loader():
            load_count[0] += 1
            return Mock(value=load_count[0])
        
        # Primera carga
        flags1 = cache.get_or_load(loader)
        assert flags1.value == 1
        assert load_count[0] == 1
        
        # Segunda carga (dentro de TTL) - debe usar cache
        flags2 = cache.get_or_load(loader)
        assert flags2.value == 1
        assert load_count[0] == 1  # No recargó
        
        # Esperar que expire TTL
        time.sleep(0.15)
        
        # Tercera carga (TTL expirado) - debe recargar
        flags3 = cache.get_or_load(loader)
        assert flags3.value == 2
        assert load_count[0] == 2  # Recargó
    
    def test_flag_cache_concurrent_access(self):
        """FlagCache es thread-safe bajo carga concurrente."""
        cache = FlagCache(ttl_seconds=1.0)
        
        load_count = [0]
        lock = threading.Lock()
        
        def loader():
            with lock:
                load_count[0] += 1
                return Mock(value=load_count[0])
        
        results = []
        
        def get_flags():
            flags = cache.get_or_load(loader)
            results.append(flags.value)
        
        # Ejecutar 100 veces en paralelo
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_flags) for _ in range(100)]
            for future in futures:
                future.result()
        
        # Todos deben obtener el mismo valor (cache hit)
        assert len(results) == 100
        assert all(v == 1 for v in results)
        assert load_count[0] == 1  # Solo cargó una vez
    
    def test_flag_cache_invalidation(self):
        """FlagCache.invalidate() fuerza recarga."""
        cache = FlagCache(ttl_seconds=10.0)
        
        load_count = [0]
        
        def loader():
            load_count[0] += 1
            return Mock(value=load_count[0])
        
        # Primera carga
        flags1 = cache.get_or_load(loader)
        assert flags1.value == 1
        
        # Invalidar cache
        cache.invalidate()
        
        # Segunda carga (debe recargar aunque TTL no expiró)
        flags2 = cache.get_or_load(loader)
        assert flags2.value == 2
        assert load_count[0] == 2
    
    def test_flag_cache_stale_on_error(self):
        """FlagCache retorna stale cache si reload falla."""
        cache = FlagCache(ttl_seconds=0.1)
        
        load_count = [0]
        
        def loader():
            load_count[0] += 1
            if load_count[0] == 1:
                return Mock(value=1)
            else:
                raise ValueError("Simulated error")
        
        # Primera carga exitosa
        flags1 = cache.get_or_load(loader)
        assert flags1.value == 1
        
        # Esperar que expire TTL
        time.sleep(0.15)
        
        # Segunda carga (falla, debe retornar stale)
        flags2 = cache.get_or_load(loader)
        assert flags2.value == 1  # Stale cache
        assert load_count[0] == 2  # Intentó recargar
    
    def test_global_flag_cache_integration(self):
        """_get_flags() usa cache global correctamente."""
        # Invalidar cache global
        _FLAG_CACHE.invalidate()
        
        # Primera llamada
        flags1 = _get_flags()
        assert flags1 is not None
        
        # Segunda llamada (debe usar cache)
        flags2 = _get_flags()
        assert flags2 is not None
        
        # Deben ser la misma instancia (cache hit)
        assert flags1 is flags2
