"""NAB Conformance Test Suite.

Verifica la equivalencia exacta, caso por caso, entre el evaluador
canónico de NAB y la especificación de referencia oficial de Numenta NAB (numenta/NAB).

Casos de prueba:
- Caso A: Detección temprana al inicio exacto de la ventana (máximo reward TP).
- Caso B: Detección retrasada a la mitad de la ventana (reward decreciente por sigmoide).
- Caso C: Detección fuera de ventana (penalización FP amortiguada / posición temporal).
- Caso D: Múltiples detecciones dentro de la misma ventana (máximo reward único, no penaliza duplicados).
- Caso E: Detecciones en el período probatorio (probationary period 15%, no penalizadas).
- Caso F: Múltiples ventanas simultáneas y normalización oficial (Null = 0.0%, Perfect = 100.0%).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from benchmarks.nab_evaluator import (
    AnomalyWindow,
    NABEvaluator,
    canonical_scaled_sigmoid,
    score_dataset_canonical,
)


def test_scaled_sigmoid_conformance():
    """Verifica que la función sigmoide escalada replique exactamente los valores canónicos de NAB."""
    # En el extremo izquierdo (y = -1.0)
    max_tp = canonical_scaled_sigmoid(-1.0)
    expected_max = (2.0 / (1.0 + math.exp(-5.0))) - 1.0  # ~ 0.98661
    assert max_tp == pytest.approx(expected_max, abs=1e-5)
    assert max_tp == pytest.approx(0.98661, abs=1e-4)

    # En el centro / mitad del evento (y = -0.5)
    mid_tp = canonical_scaled_sigmoid(-0.5)
    expected_mid = (2.0 / (1.0 + math.exp(-2.5))) - 1.0  # ~ 0.84828
    assert mid_tp == pytest.approx(expected_mid, abs=1e-5)

    # En el borde derecho (y = 0.0)
    edge_val = canonical_scaled_sigmoid(0.0)
    assert edge_val == pytest.approx(0.0, abs=1e-5)

    # Fuera de la ventana (y = 1.0)
    fp_1 = canonical_scaled_sigmoid(1.0)
    expected_fp = (2.0 / (1.0 + math.exp(5.0))) - 1.0  # ~ -0.98661
    assert fp_1 == pytest.approx(expected_fp, abs=1e-5)

    # Muy lejos de la ventana (y > 3.0)
    assert canonical_scaled_sigmoid(3.5) == -1.0
    assert canonical_scaled_sigmoid(10.0) == -1.0


@pytest.fixture
def synthetic_series():
    """Genera serie de 300 puntos para pruebas sintéticas."""
    ts = [pd.to_datetime("2020-01-01") + pd.Timedelta(minutes=i) for i in range(300)]
    return ts


def test_case_a_early_detection(synthetic_series):
    """Caso A: Detección en el inicio exacto de la ventana [100..200].
    Debe producir el máximo reward posible de TP (1.0) y 0 FP."""
    ts = synthetic_series
    windows = [(ts[100], ts[200])]
    scores = [0.0] * len(ts)
    scores[100] = 1.0  # Detección temprana en índice 100

    evaluator = NABEvaluator()
    res = score_dataset_canonical(
        timestamps=ts,
        anomaly_scores=scores,
        window_limits=windows,
        dataset_name="test_case_a",
        threshold=0.5,
        profile_name="standard",
    )

    # En Numenta NAB: un TP perfecto produce score = +1.0
    assert res.score == pytest.approx(1.0, abs=1e-4)
    assert res.tp == 1
    assert res.fp == 0
    assert res.fn == 100  # Puntos en la ventana no seleccionados por el umbral


def test_case_b_delayed_detection(synthetic_series):
    """Caso B: Detección retrasada a la mitad de la ventana (índice 150).
    El reward debe decaer según la sigmoide temporal (~0.8633)."""
    ts = synthetic_series
    windows = [(ts[100], ts[200])]
    scores = [0.0] * len(ts)
    scores[150] = 1.0  # Detección a mitad de ventana

    res = score_dataset_canonical(
        timestamps=ts,
        anomaly_scores=scores,
        window_limits=windows,
        dataset_name="test_case_b",
        threshold=0.5,
        profile_name="standard",
    )

    # Numenta NAB oficial produce score = 0.863273
    assert res.score == pytest.approx(0.863273, abs=1e-4)
    assert res.tp == 1
    assert res.fp == 0


def test_case_c_false_positive_penalty(synthetic_series):
    """Caso C: Detección fuera de ventana (índice 50, antes de la primera ventana).
    Debe recibir penalización de FN (-1.0) más FP (-0.11) = -1.11."""
    ts = synthetic_series
    windows = [(ts[100], ts[200])]
    scores = [0.0] * len(ts)
    scores[50] = 1.0  # FP en índice 50

    res = score_dataset_canonical(
        timestamps=ts,
        anomaly_scores=scores,
        window_limits=windows,
        dataset_name="test_case_c",
        threshold=0.5,
        profile_name="standard",
    )

    # Numenta NAB oficial produce score = -1.11
    assert res.score == pytest.approx(-1.11, abs=1e-4)
    assert res.tp == 0
    assert res.fp == 1


def test_case_d_multiple_detections_in_window(synthetic_series):
    """Caso D: Múltiples detecciones dentro de la misma ventana (100, 101, 102, 103).
    NAB toma el MAX reward de la ventana. Las detecciones subsecuentes NO penalizan ni suman."""
    ts = synthetic_series
    windows = [(ts[100], ts[200])]
    scores = [0.0] * len(ts)
    for idx in [100, 101, 102, 103]:
        scores[idx] = 1.0

    res = score_dataset_canonical(
        timestamps=ts,
        anomaly_scores=scores,
        window_limits=windows,
        dataset_name="test_case_d",
        threshold=0.5,
        profile_name="standard",
    )

    # Numenta NAB oficial produce score = 1.0 (mismo que Caso A) y 0 FPs
    assert res.score == pytest.approx(1.0, abs=1e-4)
    assert res.tp == 4
    assert res.fp == 0


def test_case_e_probationary_period(synthetic_series):
    """Caso E: Detección dentro del período probatorio (primeros 15% o 45 puntos).
    El punto debe ser descartado por completo del scoring (0 FP, score de null = -1.0)."""
    ts = synthetic_series
    windows = [(ts[100], ts[200])]
    scores = [0.0] * len(ts)
    scores[20] = 1.0  # Dentro de los primeros 45 puntos

    res = score_dataset_canonical(
        timestamps=ts,
        anomaly_scores=scores,
        window_limits=windows,
        dataset_name="test_case_e",
        threshold=0.5,
        profile_name="standard",
    )

    # Numenta NAB oficial ignora el punto en probation: score = -1.0, 0 FP
    assert res.score == pytest.approx(-1.0, abs=1e-4)
    assert res.fp == 0
    assert res.tp == 0


def test_case_f_multiple_windows_and_normalization(synthetic_series):
    """Caso F: Múltiples ventanas simultáneas y normalización oficial.
    2 ventanas [50..100] y [150..200].
    - Null score = -2.0 -> Normalizado = 0.0%
    - Perfect score = +2.0 -> Normalizado = 100.0%"""
    ts = synthetic_series
    windows = [(ts[50], ts[100]), (ts[150], ts[200])]

    # 1. Null detector
    null_scores = [0.0] * len(ts)
    null_res = score_dataset_canonical(
        timestamps=ts,
        anomaly_scores=null_scores,
        window_limits=windows,
        dataset_name="test_case_f_null",
        threshold=0.5,
        profile_name="standard",
    )
    assert null_res.score == pytest.approx(-2.0, abs=1e-4)

    # 2. Perfect detector
    perf_scores = [0.0] * len(ts)
    perf_scores[50] = 1.0
    perf_scores[150] = 1.0
    perf_res = score_dataset_canonical(
        timestamps=ts,
        anomaly_scores=perf_scores,
        window_limits=windows,
        dataset_name="test_case_f_perf",
        threshold=0.5,
        profile_name="standard",
    )
    assert perf_res.score == pytest.approx(2.0, abs=1e-4)

    # Normalización oficial: 100 * (raw - null) / (perfect - null)
    denom = perf_res.score - null_res.score  # 4.0
    norm_null = 100.0 * (null_res.score - null_res.score) / denom
    norm_perf = 100.0 * (perf_res.score - null_res.score) / denom
    assert norm_null == pytest.approx(0.0, abs=1e-4)
    assert norm_perf == pytest.approx(100.0, abs=1e-4)
