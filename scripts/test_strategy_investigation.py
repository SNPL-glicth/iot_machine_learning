#!/usr/bin/env python
"""FASE 10.3 — Test Strategy Investigation: probar fichas técnicas con datos reales.

Uso:
    python scripts/test_strategy_investigation.py --symbol BTC-USD
    python scripts/test_strategy_investigation.py --symbol NVDA
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
from iot_machine_learning.domain.entities.market.strategy import (
    StrategyInvestigator,
    render_strategy_card,
    render_strategy_comparison,
)


def load_strategy_data(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    symbol: str | None = None,
) -> dict[str, dict]:
    """Carga datos de estrategias desde la base de datos."""
    
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
    
    # Query para obtener datos por estrategia (sin GROUP_CONCAT para evitar límites)
    cursor.execute(f"""
        SELECT 
            COALESCE(strategy, 'baseline') as strategy,
            direction_correct,
            expected_return,
            reward_execution_costs,
            probability_up,
            emitted_at
        FROM market_predictions
        WHERE {where}
        ORDER BY strategy, emitted_at
    """, params)
    
    # Procesar datos row by row
    strategies_data: dict[str, dict] = defaultdict(lambda: {
        "direction_correct": [],
        "returns": [],
        "costs": [],
        "probabilities": [],
        "timestamps": [],
    })
    
    for row in cursor.fetchall():
        strategy, direction_correct, expected_return, cost, probability, timestamp = row
        
        strategies_data[strategy]["direction_correct"].append(bool(direction_correct))
        strategies_data[strategy]["returns"].append(float(expected_return) if expected_return else 0.0)
        strategies_data[strategy]["costs"].append(float(cost) if cost else 0.0)
        strategies_data[strategy]["probabilities"].append(float(probability) if probability else 0.5)
        strategies_data[strategy]["timestamps"].append(timestamp)
    
    # Procesar metadata
    final_strategies_data = {}
    for strategy, data in strategies_data.items():
        if not data["timestamps"]:
            continue
            
        history_days = (max(data["timestamps"]) - min(data["timestamps"])) / 86400
        
        final_strategies_data[strategy] = {
            "direction_correct": data["direction_correct"],
            "returns": data["returns"],
            "costs": data["costs"],
            "probabilities": data["probabilities"],
            "history_days": history_days,
            "evidence_status": "unknown",
            "evidence_reason": "",
        }
    
    conn.close()
    return final_strategies_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=None, help="filtrar por símbolo")
    
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
    strategies_data = load_strategy_data(host, port, user, password, database, args.symbol)
    print(f"Estrategias cargadas: {len(strategies_data)}")
    
    if not strategies_data:
        print("No hay datos para investigar.")
        return 1
    
    # Investigar estrategias
    investigator = StrategyInvestigator(risk_free_rate=0.0)
    cards = investigator.batch_investigate(strategies_data)
    
    # Renderizar comparación
    print("\n" + "=" * 80)
    print("STRATEGY INVESTIGATION RESULTS")
    print("=" * 80)
    print(render_strategy_comparison(cards))
    
    # Fichas técnicas detalladas
    for strategy, card in sorted(cards.items()):
        print("\n" + "=" * 80)
        print(render_strategy_card(card))
    
    # Resumen de veredictos
    print("\n" + "=" * 80)
    print("VEREDICTS SUMMARY")
    print("=" * 80)
    
    profitable_significant = [s for s, c in cards.items() if c.is_profitable and c.is_significant]
    profitable_not_significant = [s for s, c in cards.items() if c.is_profitable and not c.is_significant]
    significant_not_profitable = [s for s, c in cards.items() if not c.is_profitable and c.is_significant]
    neither = [s for s, c in cards.items() if not c.is_profitable and not c.is_significant]
    
    print(f"PROFITABLE & SIGNIFICANT: {len(profitable_significant)}")
    for s in profitable_significant:
        print(f"  - {s}")
    
    print(f"\nPROFITABLE but NOT SIGNIFICANT: {len(profitable_not_significant)}")
    for s in profitable_not_significant:
        print(f"  - {s}")
    
    print(f"\nSIGNIFICANT but NOT PROFITABLE: {len(significant_not_profitable)}")
    for s in significant_not_profitable:
        print(f"  - {s}")
    
    print(f"\nNEITHER: {len(neither)}")
    for s in neither:
        print(f"  - {s}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())