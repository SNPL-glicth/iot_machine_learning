"""Test dataset for SemanticEnrichment evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TestCase:
    """Single test case with expected characteristics."""
    
    id: str
    text: str
    category: str  # "industrial", "neutral", "noise"
    expected_entities: List[str]  # Equipment IDs expected
    expected_metrics: List[str]   # Metrics expected
    has_critical: bool          # Should detect critical condition
    description: str


class TestDataset:
    """Curated dataset for A/B testing semantic enrichment."""
    
    INDUSTRIAL_CASES: List[TestCase] = [
        TestCase(
            id="IND-001",
            text="Alerta crítica: Compresor C-12 registró presión de 3401 PSI durante procedimiento de shutdown. Valor normal: 3000-3200 PSI. Operador reporta anomalía en válvula V-23.",
            category="industrial",
            expected_entities=["C-12", "V-23"],
            expected_metrics=["3401 PSI"],
            has_critical=True,
            description="Compressor pressure anomaly with equipment-metric relation",
        ),
        TestCase(
            id="IND-002",
            text="Mantenimiento programado para bomba P-05. Temperatura de succión: 85°C, presión descarga: 4.2 BAR. Motor M-02 funcionando normal a 1750 RPM.",
            category="industrial",
            expected_entities=["P-05", "M-02"],
            expected_metrics=["85°C", "4.2 BAR", "1750 RPM"],
            has_critical=False,
            description="Normal maintenance with multiple equipment-metric pairs",
        ),
        TestCase(
            id="IND-003",
            text="Falla detectada en generador GEN-01. Voltaje de salida: 11.2 kV (esperado: 13.8 kV). Transformador TX-15 muestra temperatura elevada: 95°C. Acción inmediata requerida.",
            category="industrial",
            expected_entities=["GEN-01", "TX-15"],
            expected_metrics=["11.2 kV", "13.8 kV", "95°C"],
            has_critical=True,
            description="Electrical fault with voltage anomaly",
        ),
        TestCase(
            id="IND-004",
            text="Sistema HVAC sector A: Ventilador FAN-03 operando a 1200 RPM. Temperatura ambiente: 24°C. Sin alertas activas. Revisión completada.",
            category="industrial",
            expected_entities=["FAN-03"],
            expected_metrics=["1200 RPM", "24°C"],
            has_critical=False,
            description="Normal HVAC operation",
        ),
        TestCase(
            id="IND-005",
            text="Emergencia: Reactor R-100 presión 450 PSI (límite: 400 PSI). Válvula de alivio VLV-50 activada. Flujo de emergencia: 850 GPM. Cierre de proceso iniciado.",
            category="industrial",
            expected_entities=["R-100", "VLV-50"],
            expected_metrics=["450 PSI", "400 PSI", "850 GPM"],
            has_critical=True,
            description="Reactor emergency with pressure limit exceeded",
        ),
    ]
    
    NEUTRAL_CASES: List[TestCase] = [
        TestCase(
            id="NEU-001",
            text="El sistema presenta un comportamiento estable durante las últimas 24 horas. No se registraron eventos significativos.",
            category="neutral",
            expected_entities=[],
            expected_metrics=[],
            has_critical=False,
            description="No technical content",
        ),
        TestCase(
            id="NEU-002",
            text="La documentación del proceso indica que se deben seguir los procedimientos estándar establecidos en la normativa vigente.",
            category="neutral",
            expected_entities=[],
            expected_metrics=[],
            has_critical=False,
            description="Generic documentation language",
        ),
        TestCase(
            id="NEU-003",
            text="Reunión de coordinación programada para mañana a las 9:00 AM. Asistentes: equipo de operaciones y mantenimiento.",
            category="neutral",
            expected_entities=[],
            expected_metrics=[],
            has_critical=False,
            description="Administrative content",
        ),
    ]
    
    NOISE_CASES: List[TestCase] = [
        TestCase(
            id="NOI-001",
            text="El cielo está despejado y hace 25 grados. Mi perro come 3 veces al día. El coche tiene 4 llantas.",
            category="noise",
            expected_entities=[],
            expected_metrics=[],
            has_critical=False,
            description="Complete nonsense, no industrial context",
        ),
        TestCase(
            id="NOI-002",
            text="C-12 es una vitamina importante. PSI son libras por pulgada cuadrada. V-23 es mi vecino.",
            category="noise",
            expected_entities=[],  # Should NOT detect these as equipment
            expected_metrics=[],
            has_critical=False,
            description="Context confusion - same strings, wrong context",
        ),
    ]
    
    @classmethod
    def all_cases(cls) -> List[TestCase]:
        """Return all test cases."""
        return cls.INDUSTRIAL_CASES + cls.NEUTRAL_CASES + cls.NOISE_CASES
    
    @classmethod
    def get_by_category(cls, category: str) -> List[TestCase]:
        """Get cases by category."""
        return [c for c in cls.all_cases() if c.category == category]
    
    @classmethod
    def get_by_id(cls, case_id: str) -> Optional[TestCase]:
        """Get single case by ID."""
        for case in cls.all_cases():
            if case.id == case_id:
                return case
        return None
