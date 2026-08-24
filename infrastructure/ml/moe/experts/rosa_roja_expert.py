"""RosaRojaExpert — Adapter para Rosa Roja Core como experto MoE (Experimento B).

NO tiene: MySQL, Redis, Alpaca, Binance, risk, calibration, evidence gate.
SOLO: SensorWindow → estado S_t → RosaRojaCore → RosaRojaResult → ExpertOutput.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from dataclasses import dataclass

from iot_machine_learning.domain.ports.expert_port import ExpertPort, ExpertOutput, ExpertCapability
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow

# Import condicional - Rosa Roja Core vive en core/orchestration/rosa_roja
try:
    from iot_machine_learning.core.orchestration.rosa_roja.domain.movement import Movement, RhythmSignature
    from iot_machine_learning.core.orchestration.rosa_roja.domain.theta_belief import ThetaBelief
    from iot_machine_learning.core.orchestration.rosa_roja.domain.trajectory import Trajectory, TerminalState
    from iot_machine_learning.core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
    from iot_machine_learning.core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator
    from iot_machine_learning.core.orchestration.rosa_roja.modules.module3_moe_gating import MultiplicativeMoEGating
    from iot_machine_learning.core.orchestration.rosa_roja.ports.expert_jury import ExpertJuryPort
    from iot_machine_learning.core.orchestration.rosa_roja.ports.drift_sensor import DriftSensorPort
    from iot_machine_learning.core.orchestration.rosa_roja.engine import RosaRojaEngine
    ROSA_ROJA_AVAILABLE = True
except ImportError:
    ROSA_ROJA_AVAILABLE = False


@dataclass(frozen=True)
class RosaRojaResult:
    """Resultado rico del core Rosa Roja (no degradado a probability_up)."""
    trajectory: list[str]           # ej: ["expansion", "exhaustion", "pullback", "continuation"]
    trajectory_score: float         # confianza en la trayectoria [0,1]
    rhythm_score: float             # coherencia rítmica [0,1]
    lambda_val: float               # parameter λ
    theta_entropy: float            # entropía Θ
    regime_alert: Optional[str]     # "stable" | "trending" | "volatile" | "noisy" | None
    invalidation_step: Optional[int] # paso donde la trayectoria se invalida
    expected_direction: str         # "up" | "down" | "stable"
    expected_magnitude: float       # magnitud esperada
    confidence: float               # confianza general [0,1]
    evidence: Dict[str, Any]        # evidencia cruda para debugging
    status: str                     # "ok" | "unavailable" | "insufficient_history" | "invalid_state" | "error"


class RosaRojaExpert(ExpertPort):
    """Adapter: Rosa Roja Core → ExpertPort.
    
    Responsabilidad ÚNICA: traducir SensorWindow → estado S_t → RosaRojaEngine → RosaRojaResult → ExpertOutput.
    Fallback limpio: si Rosa Roja no disponible o no puede manejar, devuelve ExpertOutput con confidence=0 y metadata status.
    """

    def __init__(
        self,
        engine: Optional["RosaRojaEngine"] = None,
        min_history_points: int = 50,
        enabled: bool = True,
    ):
        self._enabled = enabled and ROSA_ROJA_AVAILABLE
        self._engine = engine
        self._min_history = min_history_points
        self._capabilities = ExpertCapability(
            regimes=("volatile", "trending", "stable", "noisy"),
            domains=("finance", "iot"),
            min_points=min_history_points,
            max_points=500,
            specialties=("trajectory", "rhythm", "regime_transition", "sequential_reasoning"),
            computational_cost=3.0,
        )
        self._fallback_output = ExpertOutput(
            prediction=0.0,
            confidence=0.0,
            trend="stable",
            metadata={
                "engine_name": "rosa_roja",
                "method": "trajectory_rhythm",
                "rosa_roja_status": "unavailable",
                "reason": "disabled_or_missing_dependency",
            }
        )

    @property
    def name(self) -> str:
        return "rosa_roja"

    @property
    def capabilities(self) -> ExpertCapability:
        return self._capabilities

    def predict(self, window: SensorWindow) -> ExpertOutput:
        if not self._enabled or self._engine is None:
            return self._fallback_output

        if not self.can_handle(window):
            return ExpertOutput(
                prediction=0.0,
                confidence=0.0,
                trend="stable",
                metadata={
                    "engine_name": "rosa_roja",
                    "method": "trajectory_rhythm",
                    "rosa_roja_status": "insufficient_history",
                    "reason": f"need >= {self._min_history} points, got {len(window.readings)}",
                }
            )

        try:
            # 1. SensorWindow → estado S_t (vector + features que Rosa Roja espera)
            s_t = self._window_to_state(window)
            
            # 2. Core Rosa Roja
            rr_result = self._engine.analyze(s_t)
            
            # 3. RosaRojaResult → ExpertOutput (probability_up derivado, metadata rico intacto)
            return ExpertOutput(
                prediction=rr_result.expected_magnitude if rr_result.expected_direction == "up" else -rr_result.expected_magnitude,
                confidence=rr_result.confidence,
                trend=rr_result.expected_direction,
                metadata={
                    "engine_name": "rosa_roja",
                    "method": "trajectory_rhythm",
                    "rosa_roja_status": rr_result.status,
                    "trajectory": rr_result.trajectory,
                    "trajectory_score": rr_result.trajectory_score,
                    "rhythm_score": rr_result.rhythm_score,
                    "lambda": rr_result.lambda_val,
                    "theta_entropy": rr_result.theta_entropy,
                    "regime_alert": rr_result.regime_alert,
                    "invalidation_step": rr_result.invalidation_step,
                    "evidence": rr_result.evidence,
                }
            )
        except Exception as e:
            # Fail-silent: no rompe el MoE
            return ExpertOutput(
                prediction=0.0,
                confidence=0.0,
                trend="stable",
                metadata={
                    "engine_name": "rosa_roja",
                    "method": "trajectory_rhythm",
                    "rosa_roja_status": "error",
                    "reason": str(e),
                }
            )

    def can_handle(self, window: SensorWindow) -> bool:
        if not self._enabled or self._engine is None:
            return False
        return len(window.readings) >= self._min_history

    def estimate_latency_ms(self, n_points: int) -> float:
        return 5.0 + (n_points * 0.02)

    def _window_to_state(self, window: SensorWindow) -> Dict[str, Any]:
        """Convierte SensorWindow → estado vectorial S_t para Rosa Roja Core."""
        values = [r.value for r in window.readings]
        timestamps = [r.timestamp for r in window.readings]
        
        return {
            "values": values,
            "timestamps": timestamps,
            "n_points": len(values),
            "current_regime": self._estimate_regime(values),
            "volatility": self._estimate_volatility(values),
        }

    def _estimate_regime(self, values: list) -> str:
        if len(values) < 10:
            return "stable"
        recent = values[-10:]
        vol = max(recent) - min(recent) if recent else 0
        mean_val = sum(recent) / len(recent) if recent else 1
        cv = vol / mean_val if mean_val != 0 else 0
        if cv > 0.05:
            return "volatile"
        elif cv > 0.02:
            return "trending"
        return "stable"

    def _estimate_volatility(self, values: list) -> float:
        if len(values) < 2:
            return 0.0
        diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        return sum(diffs) / len(diffs) if diffs else 0.0