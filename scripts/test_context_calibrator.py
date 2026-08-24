#!/usr/bin/env python
"""FASE 10.1 — Test Context Calibrator: probar calibración contextual con datos reales.

Uso:
    python scripts/test_context_calibrator.py --symbol BTC-USD
    python scripts/test_context_calibrator.py --symbol NVDA --method platt
    python scripts/test_context_calibrator.py --method bucket
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV, override=True)

import pymysql
from iot_machine_learning.domain.entities.market.calibration import (
    CalibrationMethod,
    ContextCalibrator,
    ContextKey,
    fit_context_calibrator,
)


def load_calibration_data(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    symbol: str | None = None,
) -> list[tuple[ContextKey, float, bool]]:
    """Carga datos de calibración desde la base de datos."""
    
    conn = pymysql.connect(
        host=host, port=port, user=user, password=password, database=database
    )
    cursor = conn.cursor()
    
    where_clauses = ["status = 'rewarded'", "direction_correct IS NOT NULL"]
    params = []
    
    if symbol:
        where_clauses.append("symbol = %s")
        params.append(symbol)
    
    where = " AND ".join(where_clauses)
    
    # Query para obtener datos de calibración
    cursor.execute(f"""
        SELECT 
            COALESCE(strategy, 'baseline') as strategy,
            horizon_seconds,
            COALESCE(regime, 'ALL') as regime,
            probability_up,
            direction_correct
        FROM market_predictions
        WHERE {where}
        ORDER BY emitted_at
    """, params)
    
    data = []
    for row in cursor.fetchall():
        strategy, horizon, regime, prob_up, direction_correct = row
        context = ContextKey(strategy=strategy, horizon_seconds=horizon, regime=regime)
        outcome = bool(direction_correct)
        data.append((context, float(prob_up), outcome))
    
    conn.close()
    return data


def train_test_split(
    data: list[tuple[ContextKey, float, bool]],
    train_ratio: float = 0.7,
) -> tuple[list[tuple[ContextKey, float, bool]], list[tuple[ContextKey, float, bool]]]:
    """Divide datos en train/test temporalmente."""
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def render_calibration_report(
    calibrator: ContextCalibrator,
    train_data: list[tuple[ContextKey, float, bool]],
    test_data: list[tuple[ContextKey, float, bool]],
) -> str:
    """Renderiza reporte de calibración."""
    lines = [
        "CONTEXT CALIBRATION TEST",
        "=" * 25,
        "",
        f"Calibration method: {calibrator.method.value}",
        f"Train samples: {len(train_data):,}",
        f"Test samples: {len(test_data):,}",
        "",
    ]
    
    # Evaluar en test
    test_results = calibrator.evaluate(test_data)
    
    # Agregar métricas globales
    lines.append("GLOBAL CALIBRATION METRICS (TEST)")
    lines.append("-" * 35)
    
    # Calcular métricas globales
    all_raw_probs = [prob for _, prob, _ in test_data]
    all_outcomes = [outcome for _, _, outcome in test_data]
    
    # Calibrar todas las probabilidades
    all_calibrated_probs = []
    for context, prob, _ in test_data:
        result = calibrator.calibrate(context, prob)
        all_calibrated_probs.append(result.prob_calibrated)
    
    # Brier score
    raw_brier = sum((p - (1.0 if o else 0.0)) ** 2 for p, o in zip(all_raw_probs, all_outcomes)) / len(all_raw_probs)
    calibrated_brier = sum((p - (1.0 if o else 0.0)) ** 2 for p, o in zip(all_calibrated_probs, all_outcomes)) / len(all_calibrated_probs)
    
    lines.append(f"Raw Brier:         {raw_brier:.4f}")
    lines.append(f"Calibrated Brier:   {calibrated_brier:.4f}")
    lines.append(f"Improvement:        {raw_brier - calibrated_brier:+.4f}")
    lines.append("")
    
    # Métricas por contexto
    lines.append("CALIBRATION BY CONTEXT (TEST)")
    lines.append("-" * 32)
    lines.append(f"{'Context':<30} {'n':>6} {'Raw Brier':>10} {'Cal Brier':>10} {'Δ':>10}")
    lines.append("-" * 75)
    
    for context, metrics in sorted(test_results.items()):
        n = metrics["n"]
        raw_brier = metrics["raw_brier"]
        cal_brier = metrics["calibrated_brier"]
        improvement = metrics["brier_improvement"]
        
        marker = "✓" if improvement > 0 else "✗"
        lines.append(f"{str(context):<30} {n:>6} {raw_brier:>10.4f} {cal_brier:>10.4f} {improvement:>10.4f} {marker}")
    
    # Parámetros aprendidos
    lines.append("")
    lines.append("LEARNED CALIBRATION PARAMETERS")
    lines.append("-" * 33)
    
    for context, params in sorted(calibrator._params.items()):
        if not params.is_valid:
            continue
        
        if params.method == CalibrationMethod.PLATT:
            a, b = params.params
            lines.append(f"{context}: Platt scaling")
            lines.append(f"  prob_calibrated = sigmoid({a:.4f} * prob_raw + {b:.4f})")
            lines.append(f"  Train samples: {params.n_train}")
            lines.append(f"  Train Brier: {params.train_brier:.4f}")
            lines.append(f"  Train ECE: {params.train_ece:.4f}")
        elif params.method == CalibrationMethod.BUCKET:
            buckets = dict(params.params)
            lines.append(f"{context}: Bucket calibration")
            lines.append(f"  Buckets: {len(buckets)}")
            lines.append(f"  Train samples: {params.n_train}")
            lines.append(f"  Train Brier: {params.train_brier:.4f}")
            lines.append(f"  Train ECE: {params.train_ece:.4f}")
        
        lines.append("")
    
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=None, help="filtrar por símbolo")
    parser.add_argument("--method", default="platt", choices=["platt", "bucket", "none"],
                        help="método de calibración")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="ratio de entrenamiento (default: 0.7)")
    
    args = parser.parse_args()
    
    # Credenciales desde .env
    import os
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "zenin")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "zenin_market")
    
    # Cargar datos
    print(f"Cargando datos desde {database}...")
    data = load_calibration_data(host, port, user, password, database, args.symbol)
    print(f"Total muestras cargadas: {len(data):,}")
    
    if not data:
        print("No hay datos para calibrar.")
        return 1
    
    # Dividir train/test
    train_data, test_data = train_test_split(data, args.train_ratio)
    print(f"Train: {len(train_data):,}, Test: {len(test_data):,}")
    
    # Crear calibrador
    method = CalibrationMethod(args.method)
    calibrator = fit_context_calibrator(train_data, method=method)
    
    # Evaluar
    report = render_calibration_report(calibrator, train_data, test_data)
    print(report)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())