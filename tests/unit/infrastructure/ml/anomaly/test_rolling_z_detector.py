"""Tests for RollingZScoreDetector with Freeze-on-Alert."""
from __future__ import annotations

import random
import pytest

from iot_machine_learning.infrastructure.ml.anomaly.detectors.rolling_z_detector import RollingZScoreDetector


class TestRollingZScoreDetector:

    def test_handles_insufficient_data(self) -> None:
        det = RollingZScoreDetector(short_window=5, long_window=30)
        # Without training, voting returns None
        assert det.vote(10.0) is None
        assert det.raw_score(10.0) is None
        assert not det.is_trained

    def test_no_false_positive_stable(self) -> None:
        rng = random.Random(42)
        nominal = [20.0 + rng.gauss(0, 0.5) for _ in range(100)]
        det = RollingZScoreDetector(short_window=5, long_window=50, lower=3.0, upper=3.5, hysteresis=2)
        det.train(nominal)
        assert det.is_trained

        votes = []
        for _ in range(20):
            val = 20.0 + rng.gauss(0, 0.5)
            v = det.vote(val)
            votes.append(v)

        assert all(v == 0.0 for v in votes if v is not None)

    def test_detects_clear_step_anomaly(self) -> None:
        rng = random.Random(42)
        nominal = [20.0 + rng.gauss(0, 0.5) for _ in range(100)]
        det = RollingZScoreDetector(short_window=5, long_window=50, lower=3.0, upper=3.5, hysteresis=2)
        det.train(nominal)

        # Inject sudden jump to 35.0
        votes = []
        for _ in range(10):
            v = det.vote(35.0)
            votes.append(v)

        # After hysteresis steps, vote should saturate to 1.0
        assert any(v == 1.0 for v in votes)

    def test_freeze_on_alert_prevents_assimilation(self) -> None:
        """Sustained plateau anomaly: with freeze, score stays high; without freeze, score collapses."""
        rng = random.Random(123)
        nominal = [20.0 + rng.gauss(0, 0.2) for _ in range(80)]

        # 1. Detector CON freeze-on-alert (nuevo comportamiento)
        det_frozen = RollingZScoreDetector(
            short_window=5, long_window=40, lower=3.0, upper=3.5, hysteresis=2, freeze_on_alert=True
        )
        det_frozen.train(nominal)

        # 2. Detector SIN freeze-on-alert (comportamiento defectuoso legacy)
        det_unfrozen = RollingZScoreDetector(
            short_window=5, long_window=40, lower=3.0, upper=3.5, hysteresis=2, freeze_on_alert=False
        )
        det_unfrozen.train(nominal)

        # Inyectar meseta anómala de 50 puntos (mayor que long_window)
        frozen_votes = []
        unfrozen_votes = []
        for _ in range(50):
            v_f = det_frozen.vote(35.0)
            v_u = det_unfrozen.vote(35.0)
            frozen_votes.append(v_f)
            unfrozen_votes.append(v_u)

        # El detector sin freeze asimila la falla: hacia el final de la meseta el voto colapsa a 0.0
        assert unfrozen_votes[-1] == 0.0, "Detector sin freeze debió asimilar la falla y colapsar a 0.0"

        # El detector con freeze preserva la línea base nominal: mantiene la alerta activa durante toda la meseta
        assert det_frozen.is_frozen
        assert frozen_votes[-1] == 1.0, "Detector con freeze debe mantener el voto en 1.0 durante toda la meseta"

    def test_unfreezes_after_nominal_cooldown(self) -> None:
        """Cuando la señal regresa al régimen nominal por N pasos, el detector descongela la ventana."""
        rng = random.Random(42)
        nominal = [20.0 + rng.gauss(0, 0.2) for _ in range(80)]
        det = RollingZScoreDetector(
            short_window=5, long_window=40, lower=3.0, upper=3.5, hysteresis=2, freeze_on_alert=True, cooldown_steps=3
        )
        det.train(nominal)

        # Disparar alerta con salto anómalo
        for _ in range(5):
            det.vote(35.0)
        assert det.is_frozen

        # Regresar a valores nominales (20.0):
        # Primero se vacía la ventana corta (5 pasos) mientras short_mean desciende
        for _ in range(5):
            det.vote(20.0)

        # Ahora el z-score es nominal (< lower) y arranca el cooldown de 3 pasos
        # Tras cooldown_steps pasos adicionales nominales, se descongela
        for _ in range(det._cooldown_steps):
            det.vote(20.0)
        assert not det.is_frozen

    def test_managed_observe_integration(self) -> None:
        """Prueba la interfaz observe() desacoplada de la inferencia."""
        rng = random.Random(42)
        nominal = [20.0 + rng.gauss(0, 0.2) for _ in range(60)]
        det = RollingZScoreDetector(short_window=5, long_window=40, freeze_on_alert=True)
        det.train(nominal)

        # Evaluar raw_score en candidate_value
        raw_s = det.raw_score(35.0)
        assert raw_s is not None
        assert raw_s > 5.0
        # raw_score no debe alterar el estado congelado
        assert not det.is_frozen

        # Notificar mediante observe() de que el ensamble determinó anomalía
        det.observe(35.0, is_anomaly_active=True)
        assert det.is_frozen
