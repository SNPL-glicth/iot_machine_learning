"""
Módulo de Detección de Patrones para ML
========================================

Este módulo identifica y clasifica patrones en series temporales de sensores IoT.
NO genera alertas - solo diagnóstico para entender qué está viendo el modelo.

Patrones detectados:
- STABLE: Valores estables con poca variación
- MICRO_VARIATION: Cambios pequeños que podrían ignorarse
- TREND_UP: Tendencia ascendente suave
- TREND_DOWN: Tendencia descendente suave
- SPIKE: Cambio brusco (delta spike)
- DRIFT: Deriva lenta en una dirección
- NOISE: Ruido aleatorio sin patrón claro
- INSUFFICIENT_DATA: Datos insuficientes para análisis

ISO 27001: Este módulo NO expone datos sensibles, solo métricas agregadas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, stdev
from typing import List, Dict, Optional
from datetime import datetime, timezone


class PatternType(Enum):
    """Tipos de patrones que el ML puede identificar."""
    STABLE = "stable"
    MICRO_VARIATION = "micro_variation"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    SPIKE = "spike"
    DRIFT = "drift"
    NOISE = "noise"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class PatternResult:
    """Resultado de detección de un patrón."""
    pattern_type: PatternType
    confidence: float  # 0.0 - 1.0
    description: str
    criteria: str  # Cómo se identificó
    sample_size: int
    
    def to_dict(self) -> Dict:
        return {
            "pattern_type": self.pattern_type.value,
            "confidence": round(self.confidence, 3),
            "description": self.description,
            "criteria": self.criteria,
            "sample_size": self.sample_size,
        }


@dataclass
class MicroDeltaAnalysis:
    """Análisis de micro-variaciones (cambios pequeños)."""
    total_changes: int
    micro_changes: int  # Cambios < umbral
    ignored_changes: int  # Cambios que el modelo ignora
    micro_change_rate: float  # % de cambios que son micro
    avg_micro_delta: float
    max_micro_delta: float
    sensitivity_threshold: float  # Umbral usado
    
    def to_dict(self) -> Dict:
        return {
            "total_changes": self.total_changes,
            "micro_changes": self.micro_changes,
            "ignored_changes": self.ignored_changes,
            "micro_change_rate": round(self.micro_change_rate, 2),
            "avg_micro_delta": round(self.avg_micro_delta, 4),
            "max_micro_delta": round(self.max_micro_delta, 4),
            "sensitivity_threshold": round(self.sensitivity_threshold, 4),
        }


@dataclass
class ErrorMarginAnalysis:
    """Análisis del margen de error del modelo."""
    estimated_margin: float  # Margen de error estimado
    margin_confidence: float  # Confianza en el margen
    variability: float  # Variabilidad del error
    is_reliable: bool  # Si el margen es confiable
    explanation: str
    
    def to_dict(self) -> Dict:
        return {
            "estimated_margin": round(self.estimated_margin, 4),
            "margin_confidence": round(self.margin_confidence, 3),
            "variability": round(self.variability, 4),
            "is_reliable": self.is_reliable,
            "explanation": self.explanation,
        }


@dataclass
class PatternDiagnostic:
    """Diagnóstico completo de patrones del modelo ML."""
    timestamp: str
    patterns_detected: List[PatternResult]
    dominant_pattern: Optional[PatternResult]
    micro_delta_analysis: MicroDeltaAnalysis
    error_margin: ErrorMarginAnalysis
    total_samples_analyzed: int
    analysis_window_minutes: int
    
    # Conteo de patrones por tipo
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    
    # Lo que el modelo está ignorando
    ignored_reasons: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "patterns_detected": [p.to_dict() for p in self.patterns_detected],
            "dominant_pattern": self.dominant_pattern.to_dict() if self.dominant_pattern else None,
            "micro_delta_analysis": self.micro_delta_analysis.to_dict(),
            "error_margin": self.error_margin.to_dict(),
            "total_samples_analyzed": self.total_samples_analyzed,
            "analysis_window_minutes": self.analysis_window_minutes,
            "pattern_counts": self.pattern_counts,
            "ignored_reasons": self.ignored_reasons,
        }


class PatternDetector:
    """
    Detector de patrones para series temporales de sensores.
    
    Este detector analiza valores históricos y clasifica el comportamiento
    del sensor para diagnóstico del modelo ML.
    """
    
    # Umbrales configurables
    MICRO_DELTA_THRESHOLD = 0.01  # 1% de variación = micro-cambio
    SPIKE_THRESHOLD = 0.10  # 10% de variación = spike
    TREND_MIN_SAMPLES = 5  # Mínimo de muestras para detectar tendencia
    STABILITY_THRESHOLD = 0.005  # 0.5% variación = estable
    
    def __init__(
        self,
        micro_delta_pct: float = 0.01,
        spike_pct: float = 0.10,
        stability_pct: float = 0.005,
    ):
        self.micro_delta_pct = micro_delta_pct
        self.spike_pct = spike_pct
        self.stability_pct = stability_pct
    
    def analyze(
        self,
        values: List[float],
        window_minutes: int = 60,
    ) -> PatternDiagnostic:
        """
        Analiza una serie de valores y genera diagnóstico de patrones.
        
        Args:
            values: Lista de valores del sensor (más reciente al final)
            window_minutes: Ventana de análisis en minutos
            
        Returns:
            PatternDiagnostic con el análisis completo
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        if len(values) < 2:
            return self._insufficient_data_diagnostic(timestamp, len(values), window_minutes)
        
        # Detectar todos los patrones
        patterns = self._detect_all_patterns(values)
        
        # Encontrar patrón dominante
        dominant = self._find_dominant_pattern(patterns)
        
        # Analizar micro-deltas
        micro_analysis = self._analyze_micro_deltas(values)
        
        # Calcular margen de error
        error_margin = self._calculate_error_margin(values)
        
        # Contar patrones
        pattern_counts = {}
        for p in patterns:
            key = p.pattern_type.value
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        
        # Identificar qué está ignorando el modelo
        ignored_reasons = self._identify_ignored_data(values, micro_analysis)
        
        return PatternDiagnostic(
            timestamp=timestamp,
            patterns_detected=patterns,
            dominant_pattern=dominant,
            micro_delta_analysis=micro_analysis,
            error_margin=error_margin,
            total_samples_analyzed=len(values),
            analysis_window_minutes=window_minutes,
            pattern_counts=pattern_counts,
            ignored_reasons=ignored_reasons,
        )
    
    def _insufficient_data_diagnostic(
        self,
        timestamp: str,
        sample_count: int,
        window_minutes: int,
    ) -> PatternDiagnostic:
        """Genera diagnóstico cuando hay datos insuficientes."""
        pattern = PatternResult(
            pattern_type=PatternType.INSUFFICIENT_DATA,
            confidence=0.0,
            description="Datos insuficientes para análisis de patrones",
            criteria=f"Se requieren al menos 2 muestras, se tienen {sample_count}",
            sample_size=sample_count,
        )
        
        return PatternDiagnostic(
            timestamp=timestamp,
            patterns_detected=[pattern],
            dominant_pattern=pattern,
            micro_delta_analysis=MicroDeltaAnalysis(
                total_changes=0,
                micro_changes=0,
                ignored_changes=0,
                micro_change_rate=0.0,
                avg_micro_delta=0.0,
                max_micro_delta=0.0,
                sensitivity_threshold=self.micro_delta_pct,
            ),
            error_margin=ErrorMarginAnalysis(
                estimated_margin=0.0,
                margin_confidence=0.0,
                variability=0.0,
                is_reliable=False,
                explanation="Datos insuficientes para calcular margen de error",
            ),
            total_samples_analyzed=sample_count,
            analysis_window_minutes=window_minutes,
            pattern_counts={PatternType.INSUFFICIENT_DATA.value: 1},
            ignored_reasons=[{"reason": "insufficient_data", "count": 1}],
        )
    
    def _detect_all_patterns(self, values: List[float]) -> List[PatternResult]:
        """Detecta todos los patrones presentes en los datos."""
        patterns = []
        
        # 1. Detectar estabilidad
        stability = self._detect_stability(values)
        if stability:
            patterns.append(stability)
        
        # 2. Detectar micro-variaciones
        micro_var = self._detect_micro_variation(values)
        if micro_var:
            patterns.append(micro_var)
        
        # 3. Detectar tendencias
        trend = self._detect_trend(values)
        if trend:
            patterns.append(trend)
        
        # 4. Detectar spikes
        spike = self._detect_spike(values)
        if spike:
            patterns.append(spike)
        
        # 5. Detectar drift
        drift = self._detect_drift(values)
        if drift:
            patterns.append(drift)
        
        # 6. Detectar ruido
        noise = self._detect_noise(values)
        if noise:
            patterns.append(noise)
        
        # Si no se detectó nada, marcar como datos insuficientes
        if not patterns:
            patterns.append(PatternResult(
                pattern_type=PatternType.NOISE,
                confidence=0.5,
                description="Patrón no clasificable",
                criteria="No se identificó un patrón claro",
                sample_size=len(values),
            ))
        
        return patterns
    
    def _detect_stability(self, values: List[float]) -> Optional[PatternResult]:
        """Detecta si los valores son estables."""
        if len(values) < 3:
            return None
        
        avg = mean(values)
        if avg == 0:
            return None
        
        std = stdev(values) if len(values) > 1 else 0
        cv = std / abs(avg)  # Coeficiente de variación
        
        if cv < self.stability_pct:
            return PatternResult(
                pattern_type=PatternType.STABLE,
                confidence=min(1.0, 1.0 - cv / self.stability_pct),
                description=f"Valores estables (CV={cv:.4f})",
                criteria=f"Coeficiente de variación < {self.stability_pct:.1%}",
                sample_size=len(values),
            )
        return None
    
    def _detect_micro_variation(self, values: List[float]) -> Optional[PatternResult]:
        """Detecta micro-variaciones (cambios pequeños)."""
        if len(values) < 2:
            return None
        
        deltas = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        avg_val = mean(values) if values else 1.0
        if avg_val == 0:
            avg_val = 1.0
        
        # Calcular deltas relativos
        relative_deltas = [d / abs(avg_val) for d in deltas]
        micro_count = sum(1 for d in relative_deltas if d < self.micro_delta_pct)
        
        micro_rate = micro_count / len(deltas) if deltas else 0
        
        if micro_rate > 0.5:  # Más del 50% son micro-cambios
            return PatternResult(
                pattern_type=PatternType.MICRO_VARIATION,
                confidence=micro_rate,
                description=f"Predominan micro-variaciones ({micro_rate:.0%} de cambios)",
                criteria=f"Cambios < {self.micro_delta_pct:.1%} del valor promedio",
                sample_size=len(values),
            )
        return None
    
    def _detect_trend(self, values: List[float]) -> Optional[PatternResult]:
        """Detecta tendencia ascendente o descendente."""
        if len(values) < self.TREND_MIN_SAMPLES:
            return None
        
        # Calcular pendiente simple
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = mean(values)
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return None
        
        slope = numerator / denominator
        
        # Normalizar pendiente
        value_range = max(values) - min(values) if values else 1
        if value_range == 0:
            value_range = abs(y_mean) if y_mean != 0 else 1
        
        normalized_slope = slope / value_range * n
        
        if abs(normalized_slope) > 0.3:  # Tendencia significativa
            if slope > 0:
                return PatternResult(
                    pattern_type=PatternType.TREND_UP,
                    confidence=min(1.0, abs(normalized_slope)),
                    description=f"Tendencia ascendente (pendiente={slope:.4f})",
                    criteria="Pendiente positiva significativa en ventana",
                    sample_size=len(values),
                )
            else:
                return PatternResult(
                    pattern_type=PatternType.TREND_DOWN,
                    confidence=min(1.0, abs(normalized_slope)),
                    description=f"Tendencia descendente (pendiente={slope:.4f})",
                    criteria="Pendiente negativa significativa en ventana",
                    sample_size=len(values),
                )
        return None
    
    def _detect_spike(self, values: List[float]) -> Optional[PatternResult]:
        """Detecta spikes (cambios bruscos)."""
        if len(values) < 2:
            return None
        
        avg_val = mean(values)
        if avg_val == 0:
            avg_val = 1.0
        
        deltas = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        relative_deltas = [d / abs(avg_val) for d in deltas]
        
        spike_count = sum(1 for d in relative_deltas if d > self.spike_pct)
        
        if spike_count > 0:
            max_spike = max(relative_deltas)
            return PatternResult(
                pattern_type=PatternType.SPIKE,
                confidence=min(1.0, max_spike / self.spike_pct),
                description=f"Detectados {spike_count} spikes (máx={max_spike:.1%})",
                criteria=f"Cambios > {self.spike_pct:.0%} del valor promedio",
                sample_size=len(values),
            )
        return None
    
    def _detect_drift(self, values: List[float]) -> Optional[PatternResult]:
        """Detecta deriva lenta (drift)."""
        if len(values) < 10:
            return None
        
        # Comparar primera y última mitad
        mid = len(values) // 2
        first_half_avg = mean(values[:mid])
        second_half_avg = mean(values[mid:])
        
        if first_half_avg == 0:
            return None
        
        drift_pct = abs(second_half_avg - first_half_avg) / abs(first_half_avg)
        
        # Drift es cambio gradual sin spikes
        deltas = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        max_delta = max(deltas) if deltas else 0
        avg_val = mean(values)
        
        has_no_spikes = (max_delta / abs(avg_val) if avg_val != 0 else 0) < self.spike_pct
        
        if drift_pct > 0.05 and has_no_spikes:  # 5% drift sin spikes
            direction = "ascendente" if second_half_avg > first_half_avg else "descendente"
            return PatternResult(
                pattern_type=PatternType.DRIFT,
                confidence=min(1.0, drift_pct * 10),
                description=f"Deriva {direction} de {drift_pct:.1%}",
                criteria="Cambio gradual entre primera y segunda mitad de ventana",
                sample_size=len(values),
            )
        return None
    
    def _detect_noise(self, values: List[float]) -> Optional[PatternResult]:
        """Detecta ruido aleatorio."""
        if len(values) < 5:
            return None
        
        avg = mean(values)
        if avg == 0:
            return None
        
        std = stdev(values)
        cv = std / abs(avg)
        
        # Ruido: alta variabilidad sin tendencia clara
        if cv > 0.1:  # CV > 10%
            # Verificar que no hay tendencia
            trend = self._detect_trend(values)
            if trend is None or trend.confidence < 0.5:
                return PatternResult(
                    pattern_type=PatternType.NOISE,
                    confidence=min(1.0, cv),
                    description=f"Ruido aleatorio detectado (CV={cv:.1%})",
                    criteria="Alta variabilidad sin tendencia clara",
                    sample_size=len(values),
                )
        return None
    
    def _find_dominant_pattern(self, patterns: List[PatternResult]) -> Optional[PatternResult]:
        """Encuentra el patrón dominante (mayor confianza)."""
        if not patterns:
            return None
        return max(patterns, key=lambda p: p.confidence)
    
    def _analyze_micro_deltas(self, values: List[float]) -> MicroDeltaAnalysis:
        """Analiza micro-variaciones en detalle."""
        if len(values) < 2:
            return MicroDeltaAnalysis(
                total_changes=0,
                micro_changes=0,
                ignored_changes=0,
                micro_change_rate=0.0,
                avg_micro_delta=0.0,
                max_micro_delta=0.0,
                sensitivity_threshold=self.micro_delta_pct,
            )
        
        avg_val = mean(values)
        if avg_val == 0:
            avg_val = 1.0
        
        threshold = abs(avg_val) * self.micro_delta_pct
        
        deltas = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        micro_deltas = [d for d in deltas if d < threshold]
        
        return MicroDeltaAnalysis(
            total_changes=len(deltas),
            micro_changes=len(micro_deltas),
            ignored_changes=len(micro_deltas),  # El modelo ignora micro-cambios
            micro_change_rate=len(micro_deltas) / len(deltas) * 100 if deltas else 0,
            avg_micro_delta=mean(micro_deltas) if micro_deltas else 0.0,
            max_micro_delta=max(micro_deltas) if micro_deltas else 0.0,
            sensitivity_threshold=self.micro_delta_pct,
        )
    
    def _calculate_error_margin(self, values: List[float]) -> ErrorMarginAnalysis:
        """Calcula el margen de error estimado del modelo."""
        if len(values) < 3:
            return ErrorMarginAnalysis(
                estimated_margin=0.0,
                margin_confidence=0.0,
                variability=0.0,
                is_reliable=False,
                explanation="Datos insuficientes para calcular margen de error",
            )
        
        avg = mean(values)
        std = stdev(values)
        
        if avg == 0:
            return ErrorMarginAnalysis(
                estimated_margin=std,
                margin_confidence=0.5,
                variability=std,
                is_reliable=False,
                explanation="Valor promedio es cero, margen basado en desviación estándar",
            )
        
        # Margen de error = 2 * desviación estándar (intervalo de confianza ~95%)
        margin = 2 * std
        margin_pct = margin / abs(avg) * 100
        
        # Confianza basada en cantidad de datos
        confidence = min(1.0, len(values) / 30)  # 30 muestras = confianza máxima
        
        # Variabilidad del error (qué tan consistente es)
        deltas = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        variability = stdev(deltas) if len(deltas) > 1 else 0
        
        is_reliable = len(values) >= 10 and confidence >= 0.5
        
        return ErrorMarginAnalysis(
            estimated_margin=margin_pct,
            margin_confidence=confidence,
            variability=variability,
            is_reliable=is_reliable,
            explanation=f"Margen de error ±{margin_pct:.2f}% basado en {len(values)} muestras",
        )
    
    def _identify_ignored_data(
        self,
        values: List[float],
        micro_analysis: MicroDeltaAnalysis,
    ) -> List[Dict]:
        """Identifica qué datos está ignorando el modelo y por qué."""
        ignored = []
        
        if micro_analysis.micro_changes > 0:
            ignored.append({
                "reason": "micro_variation_below_threshold",
                "count": micro_analysis.micro_changes,
                "description": f"Cambios menores a {micro_analysis.sensitivity_threshold:.1%} del valor promedio",
                "impact": "Estos cambios no afectan la predicción",
            })
        
        # Detectar valores repetidos
        if len(values) > 1:
            repeated = sum(1 for i in range(1, len(values)) if values[i] == values[i-1])
            if repeated > 0:
                ignored.append({
                    "reason": "repeated_values",
                    "count": repeated,
                    "description": "Valores idénticos consecutivos",
                    "impact": "No aportan información nueva al modelo",
                })
        
        return ignored


# Instancia global para uso en el batch runner
default_detector = PatternDetector()


def analyze_sensor_patterns(
    values: List[float],
    window_minutes: int = 60,
) -> Dict:
    """
    Función de conveniencia para analizar patrones de un sensor.
    
    Args:
        values: Lista de valores del sensor
        window_minutes: Ventana de análisis
        
    Returns:
        Diccionario con el diagnóstico completo
    """
    diagnostic = default_detector.analyze(values, window_minutes)
    return diagnostic.to_dict()
