#!/usr/bin/env python
"""FASE 10.5 — Adaptive Calibrator Training Pipeline.

Pipeline estricto sin leakage:
    TRAIN → calibrator_v1 → VALIDATION → ¿mejoró? → TEST congelado → ACCEPT/REJECT

Condiciones:
1. NO leakage: calibrador entrena SOLO en train, evalúa en val/test
2. Comparación obligatoria raw vs calibrated (Brier/ECE/LogLoss/Wilson/Economic)
3. Versionado real: model/calibrator/strategy/evidence por predicción
4. Sistema de rechazo: calibradores pueden ser rechazados
5. Fallback hierarchy: context→regime→strategy→global→unavailable

Uso:
    python scripts/train_adaptive_calibrator.py --symbol BTC-USD
    python scripts/train_adaptive_calibrator.py --symbol NVDA --min-context-samples 50
    python scripts/train_adaptive_calibrator.py --all-symbols --save-db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV, override=True)

import pymysql
from iot_machine_learning.domain.entities.market.calibration import (
    AdaptiveCalibrator,
    CalibrationMethod,
    CalibrationVerdict,
    ContextKey,
    FallbackLevel,
    compute_economic_edge,
    compute_wilson_lb,
    render_calibration_comparison,
    train_val_test_split,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.calibrator_repository_v2 import (
    CalibratorRepositoryV2,
)


def load_calibration_data(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[tuple[ContextKey, float, bool]]:
    """Carga datos de calibración desde la base de datos.
    
    Solo usa predicciones con outcome conocido (status='rewarded').
    """
    conn = pymysql.connect(
        host=host, port=port, user=user, password=password, database=database
    )
    cursor = conn.cursor()
    
    where_clauses = ["status = 'rewarded'", "direction_correct IS NOT NULL"]
    params = []
    
    if symbol:
        where_clauses.append("symbol = %s")
        params.append(symbol)
    if start_date:
        where_clauses.append("emitted_at >= %s")
        params.append(start_date)
    if end_date:
        where_clauses.append("emitted_at <= %s")
        params.append(end_date)
    
    where = " AND ".join(where_clauses)
    
    cursor.execute(f"""
        SELECT 
            COALESCE(strategy, 'baseline') as strategy,
            horizon_seconds,
            COALESCE(regime, 'ALL') as regime,
            probability_up,
            direction_correct,
            emitted_at
        FROM market_predictions
        WHERE {where}
        ORDER BY emitted_at
    """, params)
    
    data = []
    for row in cursor.fetchall():
        strategy, horizon, regime, prob_up, direction_correct, emitted_at = row
        context = ContextKey(strategy=strategy, horizon_seconds=horizon, regime=regime)
        outcome = bool(direction_correct)
        data.append((context, float(prob_up), outcome))
    
    conn.close()
    return data


def group_by_context(
    data: list[tuple[ContextKey, float, bool]]
) -> dict[ContextKey, list[tuple[float, bool]]]:
    """Agrupa datos por contexto."""
    grouped: dict[ContextKey, list[tuple[float, bool]]] = defaultdict(list)
    for context, prob, outcome in data:
        grouped[context].append((prob, outcome))
    return grouped


def render_summary_report(
    comparisons: dict[str, CalibrationComparison],
    calibrators: dict[FallbackLevel, any],
) -> str:
    """Renderiza reporte resumen de todos los calibradores evaluados."""
    lines = [
        "FASE 10.5 — ADAPTIVE CALIBRATOR TRAINING REPORT",
        "=" * 50,
        "",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Fallback Levels Trained: {', '.join([k.value for k in calibrators.keys()])}",
        "",
        "CALIBRATION COMPARISONS (TEST SET)",
        "-" * 35,
    ]
    
    # Tabla resumen
    lines.append(f"{'Context':<35} {'Verdict':>10} {'Raw Brier':>10} {'Cal Brier':>10} {'Δ Brier':>10} {'Econ Impact':>12}")
    lines.append("-" * 95)
    
    accepted = 0
    rejected = 0
    
    for context, comp in sorted(comparisons.items()):
        verdict_str = "✓ ACCEPTED" if comp.verdict == CalibrationVerdict.ACCEPTED else "✗ REJECTED"
        if comp.verdict == CalibrationVerdict.ACCEPTED:
            accepted += 1
        else:
            rejected += 1
        
        lines.append(
            f"{context:<35} {verdict_str:>10} {comp.raw_brier:>10.4f} "
            f"{comp.calibrated_brier:>10.4f} {comp.brier_improvement:>+10.4f} "
            f"{comp.economic_impact:>+12.4f}"
        )
    
    lines.append("")
    lines.append(f"ACCEPTED: {accepted} | REJECTED: {rejected}")
    lines.append("")
    
    # Detalle de rechazados
    rejected_comps = [c for c in comparisons.values() if c.verdict == CalibrationVerdict.REJECTED]
    if rejected_comps:
        lines.append("REJECTED CALIBRATORS DETAIL")
        lines.append("-" * 27)
        for comp in rejected_comps:
            lines.append(f"  {comp.context}: {comp.rejection_reason}")
        lines.append("")
    
    # Detalle de aceptados con métricas completas
    accepted_comps = [c for c in comparisons.values() if c.verdict == CalibrationVerdict.ACCEPTED]
    if accepted_comps:
        lines.append("ACCEPTED CALIBRATORS — FULL METRICS")
        lines.append("-" * 38)
        for comp in accepted_comps:
            lines.append(render_calibration_comparison(comp))
            lines.append("")
    
    return "\n".join(lines)


def save_to_database(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    calibrators: dict[FallbackLevel, any],
    comparisons: dict[str, CalibrationComparison],
    metadata: dict,
    description: str,
) -> str | None:
    """Guarda calibradores y comparaciones en la base de datos."""
    try:
        conn = pymysql.connect(
            host=host, port=port, user=user, password=password, database=database
        )
        
        repo = CalibratorRepositoryV2(conn)
        
        # Para cada calibrador de fallback level que fue aceptado, guardamos
        # En la práctica, guardaríamos el calibrador CONTEXT como principal
        # y los otros como fallback levels en metadata
        
        # Usamos el calibrador CONTEXT como "principal" si existe
        main_calibrator = calibrators.get(FallbackLevel.CONTEXT)
        if not main_calibrator:
            main_calibrator = calibrators.get(FallbackLevel.REGIME)
        if not main_calibrator:
            main_calibrator = calibrators.get(FallbackLevel.STRATEGY)
        if not main_calibrator:
            main_calibrator = calibrators.get(FallbackLevel.GLOBAL)
        
        if not main_calibrator:
            print("No hay calibrador principal para guardar")
            return None
        
        # Creamos una comparación agregada para el calibrador principal
        # (en realidad cada contexto tiene su propia comparación)
        accepted_comparisons = [c for c in comparisons.values() if c.verdict == CalibrationVerdict.ACCEPTED]
        if not accepted_comparisons:
            print("No hay comparaciones aceptadas para guardar")
            return None
        
        # Usamos la primera comparación aceptada como representativa
        # En producción se guardaría una por contexto
        main_comparison = accepted_comparisons[0]
        
        calibrator_id = repo.save_calibrator_v2(
            calibrator=main_calibrator,
            comparison=main_comparison,
            description=description,
            metadata=metadata,
        )
        
        conn.close()
        return calibrator_id
        
    except Exception as e:
        print(f"Error guardando en BD: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=None, help="Símbolo a calibrar (ej: BTC-USD)")
    parser.add_argument("--all-symbols", action="store_true", help="Calibrar todos los símbolos")
    parser.add_argument("--start-date", default=None, help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument("--method", default="platt", choices=["platt", "bucket", "isotonic"],
                        help="Método de calibración")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Ratio train (default: 0.6)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Ratio validation (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Ratio test (default: 0.2)")
    parser.add_argument("--min-train-samples", type=int, default=100, help="Mínimo muestras train")
    parser.add_argument("--min-val-samples", type=int, default=50, help="Mínimo muestras validation")
    parser.add_argument("--min-test-samples", type=int, default=50, help="Mínimo muestras test")
    parser.add_argument("--min-context-samples", type=int, default=30, help="Mínimo muestras por contexto")
    parser.add_argument("--brier-tolerance", type=float, default=0.0, help="Tolerancia mejora Brier")
    parser.add_argument("--economic-tolerance", type=float, default=-0.001, help="Tolerancia impacto económico")
    parser.add_argument("--save-db", action="store_true", help="Guardar en base de datos")
    parser.add_argument("--output-json", default=None, help="Guardar reporte en JSON")
    
    args = parser.parse_args()
    
    if not args.all_symbols and not args.symbol:
        print("Error: debe especificar --symbol o --all-symbols")
        return 1
    
    # Validar ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("Error: train/val/test ratios deben sumar 1.0")
        return 1
    
    # Credenciales
    import os
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "zenin")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "zenin_market")
    
    # Cargar datos
    print(f"Cargando datos desde {database}...")
    if args.all_symbols:
        data = load_calibration_data(host, port, user, password, database,
                                    start_date=args.start_date, end_date=args.end_date)
    else:
        data = load_calibration_data(host, port, user, password, database,
                                    symbol=args.symbol, start_date=args.start_date, end_date=args.end_date)
    
    print(f"Total muestras cargadas: {len(data):,}")
    
    if len(data) < (args.min_train_samples + args.min_val_samples + args.min_test_samples):
        print(f"Insuficientes datos. Necesario: {args.min_train_samples + args.min_val_samples + args.min_test_samples}")
        return 1
    
    # Estadísticas por contexto
    grouped = group_by_context(data)
    print(f"Contextos únicos: {len(grouped)}")
    for ctx, items in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {ctx}: {len(items)} muestras")
    
    # Crear calibrador adaptativo
    method = CalibrationMethod(args.method)
    calibrator = AdaptiveCalibrator(
        method=method,
        min_train_samples=args.min_train_samples,
        min_val_samples=args.min_val_samples,
        min_test_samples=args.min_test_samples,
        brier_tolerance=args.brier_tolerance,
        economic_tolerance=args.economic_tolerance,
        min_context_samples=args.min_context_samples,
    )
    
    # Entrenar y evaluar
    print("\nEntrenando y evaluando con train/val/test split...")
    trained_calibrators, comparisons = calibrator.train_and_evaluate(data)
    
    if not trained_calibrators:
        print("No se pudo entrenar ningún calibrador (insuficientes datos por nivel de fallback)")
        return 1
    
    # Renderizar reporte
    report = render_summary_report(comparisons, trained_calibrators)
    print(report)
    
    # Guardar en BD si solicitado
    calibrator_id = None
    if args.save_db:
        print("\nGuardando en base de datos...")
        metadata = {
            "symbol": args.symbol,
            "method": args.method,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "min_train_samples": args.min_train_samples,
            "min_val_samples": args.min_val_samples,
            "min_test_samples": args.min_test_samples,
            "min_context_samples": args.min_context_samples,
            "brier_tolerance": args.brier_tolerance,
            "economic_tolerance": args.economic_tolerance,
            "total_samples": len(data),
            "contexts_count": len(grouped),
            "fallback_levels": [k.value for k in trained_calibrators.keys()],
        }
        description = f"FASE 10.5 Adaptive Calibrator {args.method} - {args.symbol or 'ALL'} - {datetime.now().isoformat()}"
        
        calibrator_id = save_to_database(
            host, port, user, password, database,
            trained_calibrators, comparisons, metadata, description
        )
        
        if calibrator_id:
            print(f"Calibrador guardado: {calibrator_id}")
        else:
            print("No se pudo guardar el calibrador")
    
    # Guardar JSON si solicitado
    if args.output_json:
        output = {
            "calibrator_id": calibrator_id,
            "metadata": {
                "symbol": args.symbol,
                "method": args.method,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "total_samples": len(data),
                "contexts": len(grouped),
            },
            "comparisons": {
                ctx: {
                    "verdict": comp.verdict.value,
                    "rejection_reason": comp.rejection_reason,
                    "n_train": comp.n_train,
                    "n_val": comp.n_val,
                    "n_test": comp.n_test,
                    "raw_brier": comp.raw_brier,
                    "calibrated_brier": comp.calibrated_brier,
                    "brier_improvement": comp.brier_improvement,
                    "raw_ece": comp.raw_ece,
                    "calibrated_ece": comp.calibrated_ece,
                    "ece_improvement": comp.ece_improvement,
                    "raw_log_loss": comp.raw_log_loss,
                    "calibrated_log_loss": comp.calibrated_log_loss,
                    "log_loss_improvement": comp.log_loss_improvement,
                    "raw_wilson_lb": comp.raw_wilson_lb,
                    "calibrated_wilson_lb": comp.calibrated_wilson_lb,
                    "wilson_improvement": comp.wilson_improvement,
                    "raw_economic_edge": comp.raw_economic_edge,
                    "calibrated_economic_edge": comp.calibrated_economic_edge,
                    "economic_impact": comp.economic_impact,
                }
                for ctx, comp in comparisons.items()
            },
            "fallback_levels": [k.value for k in trained_calibrators.keys()],
        }
        
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nReporte guardado en: {args.output_json}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())