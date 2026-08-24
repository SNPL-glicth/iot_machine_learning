#!/usr/bin/env python
"""FASE 10.1 — Calibration Investigation: rastrear y separar fuentes de probabilidad.

Objetivo:
- Separar probabilidad del experto vs probabilidad final (actualmente no hay MoE)
- Analizar calibración por horizonte/régimen/estrategia
- Identificar dónde está la sobref confianza
- Preparar para calibración por contexto
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


def _bucket_analysis(probability: float) -> str:
    """Bucket de probabilidad para análisis."""
    return f"{round(probability * 10) / 10:.1f}"


def analyze_calibration(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    symbol: str | None = None,
    strategy: str | None = None,
    horizon: int | None = None,
    regime: str | None = None,
) -> dict:
    """Analiza calibración por múltiples dimensiones."""
    
    conn = pymysql.connect(
        host=host, port=port, user=user, password=password, database=database
    )
    cursor = conn.cursor()
    
    # Query base con filtros
    where_clauses = ["status = 'rewarded'", "direction_correct IS NOT NULL"]
    params = []
    
    if symbol:
        where_clauses.append("symbol = %s")
        params.append(symbol)
    if strategy:
        where_clauses.append("strategy = %s")
        params.append(strategy)
    if horizon:
        where_clauses.append("horizon_seconds = %s")
        params.append(horizon)
    if regime:
        where_clauses.append("regime = %s")
        params.append(regime)
    
    where = " AND ".join(where_clauses)
    
    # Stats globales
    cursor.execute(f"""
        SELECT 
            COUNT(*) as total,
            AVG(probability_up) as avg_prob,
            AVG(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as actual_accuracy,
            AVG(POW(probability_up - IF(direction_correct, 1.0, 0.0), 2)) as brier
        FROM market_predictions
        WHERE {where}
    """, params)
    
    global_stats = cursor.fetchone()
    
    # Calibración por bucket
    cursor.execute(f"""
        SELECT 
            ROUND(probability_up * 10) / 10 as bucket,
            COUNT(*) as n,
            AVG(probability_up) as declared,
            AVG(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as realized,
            AVG(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) - AVG(probability_up) as delta
        FROM market_predictions
        WHERE {where}
        GROUP BY ROUND(probability_up * 10) / 10
        ORDER BY bucket
    """, params)
    
    buckets = cursor.fetchall()
    
    # Por estrategia
    cursor.execute(f"""
        SELECT 
            strategy,
            COUNT(*) as n,
            AVG(probability_up) as avg_prob,
            AVG(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as accuracy,
            AVG(POW(probability_up - IF(direction_correct, 1.0, 0.0), 2)) as brier
        FROM market_predictions
        WHERE {where}
        GROUP BY strategy
        ORDER BY n DESC
    """, params)
    
    by_strategy = cursor.fetchall()
    
    # Por horizonte
    cursor.execute(f"""
        SELECT 
            horizon_seconds,
            COUNT(*) as n,
            AVG(probability_up) as avg_prob,
            AVG(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as accuracy,
            AVG(POW(probability_up - IF(direction_correct, 1.0, 0.0), 2)) as brier
        FROM market_predictions
        WHERE {where}
        GROUP BY horizon_seconds
        ORDER BY horizon_seconds
    """, params)
    
    by_horizon = cursor.fetchall()
    
    # Por régimen
    cursor.execute(f"""
        SELECT 
            COALESCE(regime, 'ALL') as regime,
            COUNT(*) as n,
            AVG(probability_up) as avg_prob,
            AVG(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as accuracy,
            AVG(POW(probability_up - IF(direction_correct, 1.0, 0.0), 2)) as brier
        FROM market_predictions
        WHERE {where}
        GROUP BY COALESCE(regime, 'ALL')
        ORDER BY n DESC
    """, params)
    
    by_regime = cursor.fetchall()
    
    # Calibración cruzada: estrategia × horizonte
    cursor.execute(f"""
        SELECT 
            strategy,
            horizon_seconds,
            COUNT(*) as n,
            AVG(probability_up) as avg_prob,
            AVG(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as accuracy
        FROM market_predictions
        WHERE {where}
        GROUP BY strategy, horizon_seconds
        ORDER BY strategy, horizon_seconds
    """, params)
    
    cross_strategy_horizon = cursor.fetchall()
    
    conn.close()
    
    return {
        "global": global_stats,
        "buckets": buckets,
        "by_strategy": by_strategy,
        "by_horizon": by_horizon,
        "by_regime": by_regime,
        "cross_strategy_horizon": cross_strategy_horizon,
    }


def render_report(analysis: dict, title: str = "CALIBRATION INVESTIGATION") -> str:
    """Renderiza reporte ASCII del análisis."""
    lines = [title, "=" * len(title), ""]
    
    # Global stats
    total, avg_prob, accuracy, brier = analysis["global"]
    lines.append("GLOBAL STATS")
    lines.append("-" * 11)
    lines.append(f"Total predictions: {int(total):,}")
    lines.append(f"Avg probability:   {float(avg_prob):.4f}")
    lines.append(f"Actual accuracy:   {float(accuracy):.4f}")
    lines.append(f"Brier score:       {float(brier):.4f}")
    lines.append(f"Calibration gap:  {abs(float(avg_prob) - float(accuracy)):.4f}")
    lines.append("")
    
    # Calibración por bucket
    lines.append("CALIBRATION BY BUCKET")
    lines.append("-" * 22)
    lines.append(f"{'Bucket':<8} {'Declared':>10} {'Realized':>10} {'Δ':>10} {'n':>8}")
    lines.append("-" * 50)
    
    for bucket, n, declared, realized, delta in analysis["buckets"]:
        marker = "🚨" if abs(float(delta)) > 0.10 else ""
        lines.append(f"{bucket:<8} {float(declared):>10.4f} {float(realized):>10.4f} {float(delta):>10.4f} {n:>8} {marker}")
    lines.append("")
    
    # Por estrategia
    lines.append("BY STRATEGY")
    lines.append("-" * 11)
    lines.append(f"{'Strategy':<15} {'n':>8} {'Avg Prob':>10} {'Accuracy':>10} {'Brier':>10}")
    lines.append("-" * 60)
    
    for strategy, n, avg_prob, accuracy, brier in analysis["by_strategy"]:
        gap = abs(float(avg_prob) - float(accuracy))
        marker = "🚨" if gap > 0.10 else ""
        lines.append(f"{strategy:<15} {n:>8} {float(avg_prob):>10.4f} {float(accuracy):>10.4f} {float(brier):>10.4f} {marker}")
    lines.append("")
    
    # Por horizonte
    lines.append("BY HORIZON")
    lines.append("-" * 10)
    lines.append(f"{'Horizon':<10} {'n':>8} {'Avg Prob':>10} {'Accuracy':>10} {'Brier':>10}")
    lines.append("-" * 55)
    
    for horizon, n, avg_prob, accuracy, brier in analysis["by_horizon"]:
        gap = abs(float(avg_prob) - float(accuracy))
        marker = "🚨" if gap > 0.10 else ""
        lines.append(f"{horizon:<10} {n:>8} {float(avg_prob):>10.4f} {float(accuracy):>10.4f} {float(brier):>10.4f} {marker}")
    lines.append("")
    
    # Por régimen
    lines.append("BY REGIME")
    lines.append("-" * 9)
    lines.append(f"{'Regime':<10} {'n':>8} {'Avg Prob':>10} {'Accuracy':>10} {'Brier':>10}")
    lines.append("-" * 54)
    
    for regime, n, avg_prob, accuracy, brier in analysis["by_regime"]:
        gap = abs(float(avg_prob) - float(accuracy))
        marker = "🚨" if gap > 0.10 else ""
        lines.append(f"{regime:<10} {n:>8} {float(avg_prob):>10.4f} {float(accuracy):>10.4f} {float(brier):>10.4f} {marker}")
    lines.append("")
    
    # Cruzado estrategia × horizonte
    lines.append("CROSS: STRATEGY × HORIZON")
    lines.append("-" * 27)
    lines.append(f"{'Strategy':<15} {'Horizon':<10} {'n':>8} {'Avg Prob':>10} {'Accuracy':>10}")
    lines.append("-" * 60)
    
    for strategy, horizon, n, avg_prob, accuracy in analysis["cross_strategy_horizon"]:
        gap = abs(float(avg_prob) - float(accuracy))
        marker = "🚨" if gap > 0.10 else ""
        lines.append(f"{strategy:<15} {horizon:<10} {n:>8} {float(avg_prob):>10.4f} {float(accuracy):>10.4f} {marker}")
    
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=None, help="filtrar por símbolo")
    parser.add_argument("--strategy", default=None, help="filtrar por estrategia")
    parser.add_argument("--horizon", type=int, default=None, help="filtrar por horizonte (segundos)")
    parser.add_argument("--regime", default=None, help="filtrar por régimen")
    
    args = parser.parse_args()
    
    # Credenciales desde .env
    import os
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "zenin")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "zenin_market")
    
    title = f"CALIBRATION INVESTIGATION"
    if args.symbol:
        title += f" — {args.symbol}"
    if args.strategy:
        title += f" — {args.strategy}"
    if args.horizon:
        title += f" — {args.horizon}s"
    if args.regime:
        title += f" — {args.regime}"
    
    analysis = analyze_calibration(
        host=host, port=port, user=user, password=password, database=database,
        symbol=args.symbol, strategy=args.strategy, horizon=args.horizon, regime=args.regime
    )
    
    print(render_report(analysis, title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())