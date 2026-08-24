#!/usr/bin/env python
"""FASE 10.4 — Test Drift Detection: probar detección de drift con datos reales.

Uso:
    python scripts/test_drift_detection.py --symbol BTC-USD
    python scripts/test_drift_detection.py --symbol NVDA --windows 100,500,1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV, override=True)

import pymysql
from iot_machine_learning.domain.entities.market.drift import (
    DriftConfig,
    DriftDetector,
    render_drift_report,
)


def load_drift_data(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    symbol: str | None = None,
    strategy: str | None = None,
) -> list[tuple[float, bool]]:
    """Carga datos temporales para detección de drift."""
    
    conn = pymysql.connect(
        host=host, port=port, user=user, password=password, database=database
    )
    cursor = conn.cursor()
    
    where_clauses = ["status = 'rewarded'", "direction_correct IS NOT NULL"]
    params = []
    
    if symbol:
        where_clauses.append("symbol = %s")
        params.append(symbol)
    if strategy:
        where_clauses.append("strategy = %s")
        params.append(strategy)
    
    where = " AND ".join(where_clauses)
    
    # Query para obtener datos temporales ordenados
    cursor.execute(f"""
        SELECT 
            expected_return,
            direction_correct
        FROM market_predictions
        WHERE {where}
        ORDER BY emitted_at
    """, params)
    
    data = []
    for row in cursor.fetchall():
        expected_return, direction_correct = row
        data.append((float(expected_return) if expected_return else 0.0, bool(direction_correct)))
    
    conn.close()
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=None, help="filtrar por símbolo")
    parser.add_argument("--strategy", default=None, help="filtrar por estrategia")
    parser.add_argument("--windows", default="100,500,1000,5000", 
                        help="ventanas temporales (default: 100,500,1000,5000)")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="mínimo de muestras para evaluar")
    
    args = parser.parse_args()
    
    # Credenciales desde .env
    import os
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "zenin")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "zenin_market")
    
    # Parsear ventanas
    windows = tuple(int(w) for w in args.windows.split(",") if w.strip())
    
    # Configuración personalizada
    config = DriftConfig(
        windows=windows,
        min_samples=args.min_samples,
    )
    
    # Cargar datos
    print(f"Cargando datos desde {database}...")
    data = load_drift_data(host, port, user, password, database, args.symbol, args.strategy)
    print(f"Muestras cargadas: {len(data):,}")
    
    if not data:
        print("No hay datos para detección de drift.")
        return 1
    
    # Crear detector y alimentar datos
    detector = DriftDetector(config=config)
    print("Alimentando detector con datos...")
    
    # Limitar a últimas 2,000 muestras para performance
    data_sample = data[-2000:] if len(data) > 2000 else data
    print(f"Usando muestra de {len(data_sample):,} observaciones")
    
    for i, (value, outcome) in enumerate(data_sample):
        detector.add_observation(value, outcome)
        
        # Detectar drift periódicamente
        if i % 500 == 0 and i > 0:
            alert = detector.detect_drift()
            if alert:
                print(f"\n🚨 DRIFT DETECTADO en muestra {i}: {alert.reason}")
    
    # Detección final
    final_alert = detector.detect_drift()
    if final_alert:
        print(f"\n🚨 DRIFT FINAL: {final_alert.reason}")
    
    # Renderizar reporte
    print("\n" + "=" * 80)
    print("DRIFT DETECTION RESULTS")
    print("=" * 80)
    print(render_drift_report(detector))
    
    # Contexto
    context = f"{args.symbol or 'ALL'}"
    if args.strategy:
        context += f" · {args.strategy}"
    
    print(f"\nContexto: {context}")
    print(f"Ventanas evaluadas: {', '.join(f'last_{w}' for w in windows)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())