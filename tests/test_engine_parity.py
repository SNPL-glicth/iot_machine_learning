"""Script de validación de paridad entre motores."""

from __future__ import annotations

import sys
from pathlib import Path

# Agregar path del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iot_machine_learning.infrastructure.analysis import build_unified_engine
from iot_machine_learning.domain.ports.analysis import AnalysisContext, InputType


def test_text_analysis_parity():
    """Verifica paridad en análisis de texto."""
    print("\n=== TEST 1: Análisis de Texto ===")
    
    # Input de prueba
    test_input = """
    Error crítico detectado en el sistema de trading.
    Precio del activo cayó 15% en los últimos 30 minutos.
    Se requiere atención inmediata del equipo de operaciones.
    """
    
    context = AnalysisContext(
        tenant_id="test_tenant",
        series_id="test_series_001",
        input_type=InputType.TEXT,
        domain_hint="trading",
    )
    
    # Motor unificado
    unified_engine = build_unified_engine()
    result = unified_engine.analyze(test_input, context)
    
    print(f"✓ Motor unificado ejecutado")
    print(f"  - Tipo detectado: {result.signal.input_type.value}")
    print(f"  - Dominio: {result.signal.domain}")
    print(f"  - Severidad: {result.decision.severity}")
    print(f"  - Confianza: {result.decision.confidence:.3f}")
    print(f"  - Percepciones: {len(result.decision.perceptions)}")
    print(f"  - Narrativa: {result.explanation.narrative[:100]}...")
    
    # Validaciones
    assert result.signal.input_type == InputType.TEXT, "Tipo incorrecto"
    assert result.signal.domain in ["trading", "general"], "Dominio inesperado"
    assert result.decision.severity in ["info", "warning", "critical"], "Severidad inválida"
    assert 0.0 <= result.decision.confidence <= 1.0, "Confianza fuera de rango"
    
    print("✓ Validaciones pasadas")
    return result


def test_timeseries_analysis_parity():
    """Verifica paridad en análisis de series temporales."""
    print("\n=== TEST 2: Análisis de Series Temporales ===")
    
    # Input de prueba: serie con tendencia creciente
    test_input = [10.0, 12.0, 15.0, 18.0, 22.0, 28.0, 35.0, 45.0]
    
    context = AnalysisContext(
        tenant_id="test_tenant",
        series_id="test_series_002",
        input_type=InputType.TIMESERIES,
    )
    
    # Motor unificado
    unified_engine = build_unified_engine()
    result = unified_engine.analyze(test_input, context)
    
    print(f"✓ Motor unificado ejecutado")
    print(f"  - Tipo detectado: {result.signal.input_type.value}")
    print(f"  - Dominio: {result.signal.domain}")
    print(f"  - Severidad: {result.decision.severity}")
    print(f"  - Confianza: {result.decision.confidence:.3f}")
    print(f"  - Features: {list(result.signal.features.keys())}")
    
    # Validaciones
    assert result.signal.input_type == InputType.TIMESERIES, "Tipo incorrecto"
    assert "mean" in result.signal.features, "Falta feature mean"
    assert "std" in result.signal.features, "Falta feature std"
    
    print("✓ Validaciones pasadas")
    return result


def test_pipeline_timing():
    """Verifica que el pipeline registra tiempos."""
    print("\n=== TEST 3: Pipeline Timing ===")
    
    test_input = "Texto de prueba para timing"
    context = AnalysisContext(
        tenant_id="test_tenant",
        series_id="test_series_003",
    )
    
    unified_engine = build_unified_engine()
    result = unified_engine.analyze(test_input, context)
    
    print(f"✓ Pipeline timing:")
    for phase, ms in result.pipeline_timing.items():
        print(f"  - {phase}: {ms:.2f}ms")
    
    total_ms = sum(result.pipeline_timing.values())
    print(f"  - TOTAL: {total_ms:.2f}ms")
    
    # Validaciones
    assert "perceive" in result.pipeline_timing, "Falta timing de perceive"
    assert "analyze" in result.pipeline_timing, "Falta timing de analyze"
    assert "reason" in result.pipeline_timing, "Falta timing de reason"
    assert "explain" in result.pipeline_timing, "Falta timing de explain"
    assert total_ms < 2000, "Pipeline excede budget de 2000ms"
    
    print("✓ Validaciones pasadas")
    return result


def test_explanation_structure():
    """Verifica estructura de explicación."""
    print("\n=== TEST 4: Estructura de Explicación ===")
    
    test_input = "Análisis de prueba"
    context = AnalysisContext(
        tenant_id="test_tenant",
        series_id="test_series_004",
    )
    
    unified_engine = build_unified_engine()
    result = unified_engine.analyze(test_input, context)
    
    exp = result.explanation
    
    print(f"✓ Explicación generada:")
    print(f"  - Narrativa: {len(exp.narrative)} caracteres")
    print(f"  - Contribuciones: {len(exp.contributions)}")
    print(f"  - Traza: {len(exp.reasoning_trace)} pasos")
    print(f"  - Dominio: {exp.domain}")
    print(f"  - Severidad: {exp.severity}")
    
    # Validaciones
    assert len(exp.narrative) > 0, "Narrativa vacía"
    assert len(exp.reasoning_trace) > 0, "Traza vacía"
    assert exp.domain != "", "Dominio vacío"
    assert exp.severity in ["info", "warning", "critical"], "Severidad inválida"
    
    print("✓ Validaciones pasadas")
    return result


def main():
    """Ejecuta todos los tests de paridad."""
    print("=" * 60)
    print("VALIDACIÓN DE PARIDAD: Motor Unificado")
    print("=" * 60)
    
    try:
        test_text_analysis_parity()
        test_timeseries_analysis_parity()
        test_pipeline_timing()
        test_explanation_structure()
        
        print("\n" + "=" * 60)
        print("✓ TODOS LOS TESTS DE PARIDAD PASARON")
        print("=" * 60)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ TEST FALLÓ: {e}")
        return 1
    
    except Exception as e:
        print(f"\n✗ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
