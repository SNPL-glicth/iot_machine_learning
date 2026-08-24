#!/usr/bin/env python
"""FASE 10.2 — Test Evidence Engine: probar evaluación de evidencia con datos reales.

Uso:
    python scripts/test_evidence_engine.py --symbol BTC-USD
    python scripts/test_evidence_engine.py --symbol NVDA --min-n 50
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
from iot_machine_learning.domain.entities.market.evidence import (
    EvidenceConfig,
    EvidenceEngine,
    render_evidence_report,
)


def load_evidence_data(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    symbol: str | None = None,
) -> dict[str, dict]:
    """Carga datos de evidencia por contexto desde la base de datos."""
    
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
    
    # Query para obtener datos por contexto
    cursor.execute(f"""
        SELECT 
            COALESCE(strategy, 'baseline') as strategy,
            horizon_seconds,
            COALESCE(regime, 'ALL') as regime,
            COUNT(*) as n,
            MIN(emitted_at) as first_ts,
            MAX(emitted_at) as last_ts,
            AVG(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as accuracy,
            AVG(magnitude_error) as avg_magnitude_error,
            AVG(expected_return) as avg_expected_return,
            AVG(reward_execution_costs) as avg_cost,
            AVG(calibration_error) as avg_calibration_error
        FROM market_predictions
        WHERE {where}
        GROUP BY strategy, horizon_seconds, regime
        ORDER BY n DESC
    """, params)
    
    contexts_data = {}
    for row in cursor.fetchall():
        strategy, horizon, regime, n, first_ts, last_ts, accuracy, magnitude_error, expected_return, cost, calibration_error = row
        
        context = f"{strategy}·{horizon}s·{regime}"
        history_days = (last_ts - first_ts) / 86400 if first_ts and last_ts else 0
        
        # Calcular recency accuracies (simulado con bandas)
        # En producción esto requeriría una query más compleja
        recency_accuracies = [accuracy] * 4  # Placeholder
        
        contexts_data[context] = {
            "n": n,
            "history_days": history_days,
            "accuracy": accuracy,
            "magnitude_errors": [magnitude_error] if magnitude_error else [],
            "expected_returns": [expected_return] if expected_return else [],
            "costs": [cost] if cost else [],
            "recency_accuracies": recency_accuracies,
            "calibration_errors": [calibration_error] if calibration_error else [],
        }
    
    conn.close()
    return contexts_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=None, help="filtrar por símbolo")
    parser.add_argument("--min-n", type=int, default=100, help="n mínimo para evidencia")
    parser.add_argument("--min-accuracy", type=float, default=0.52, help="accuracy mínimo")
    
    args = parser.parse_args()
    
    # Credenciales desde .env
    import os
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "zenin")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "zenin_market")
    
    # Configuración personalizada
    config = EvidenceConfig(
        min_n=args.min_n,
        min_accuracy=args.min_accuracy,
    )
    
    # Cargar datos
    print(f"Cargando datos desde {database}...")
    contexts_data = load_evidence_data(host, port, user, password, database, args.symbol)
    print(f"Contextos cargados: {len(contexts_data)}")
    
    if not contexts_data:
        print("No hay datos para evaluar.")
        return 1
    
    # Crear engine y evaluar
    engine = EvidenceEngine(config=config)
    results = engine.batch_evaluate(contexts_data)
    
    # Renderizar reportes
    print("\n" + "=" * 80)
    print("EVIDENCE ENGINE RESULTS")
    print("=" * 80)
    
    # Resumen
    supported = sum(1 for r in results.values() if r.status.value == "evidence_supported")
    degraded = sum(1 for r in results.values() if r.status.value == "evidence_degraded")
    insufficient = sum(1 for r in results.values() if r.status.value == "insufficient_evidence")
    
    print(f"\nRESUMEN:")
    print(f"  EVIDENCE_SUPPORTED:   {supported}")
    print(f"  EVIDENCE_DEGRADED:    {degraded}")
    print(f"  INSUFFICIENT_EVIDENCE: {insufficient}")
    print(f"  Total:                {len(results)}")
    
    # Reportes detallados por contexto
    for context, verdict in sorted(results.items()):
        print("\n" + "=" * 80)
        print(render_evidence_report(verdict))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())