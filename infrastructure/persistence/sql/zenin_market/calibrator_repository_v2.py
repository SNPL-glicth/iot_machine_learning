"""FASE 10.5 — Calibrator Repository V2: versionado real y sistema de rechazo.

Este repositorio maneja:
- Versionado real: model/calibrator/strategy/evidence versions por predicción
- Sistema de rechazo: registrar calibradores rechazados con razones
- Comparaciones raw vs calibrated con métricas completas
- Tracking de qué calibrador se usó en cada predicción
- Historial completo para auditoría

Reglas estrictas (FASE 10.5):
1. NO leakage: train/val/test split estricto
2. Comparación obligatoria raw vs calibrated
3. Versionado real por predicción
4. Sistema de rechazo: calibradores pueden ser rechazados
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final
from enum import Enum

import pymysql

from iot_machine_learning.domain.entities.market.calibration import (
    CalibrationComparison,
    CalibrationMethod,
    CalibrationVerdict,
    ContextCalibrator,
    ContextKey,
)


__all__ = [
    "CalibratorVersionV2",
    "PredictionVersioning",
    "CalibratorRepositoryV2",
]


@dataclass(frozen=True, slots=True)
class CalibratorVersionV2:
    """Versión de calibrador con metadata completa."""
    
    calibrator_id: str
    method: CalibrationMethod
    created_at: str
    is_active: bool
    description: str | None
    
    # Train/val/test split info
    train_samples: int
    val_samples: int
    test_samples: int
    
    # Métricas raw
    raw_brier: float
    raw_ece: float
    raw_log_loss: float
    raw_wilson_lb: float
    raw_economic_edge: float
    
    # Métricas calibrated
    calibrated_brier: float
    calibrated_ece: float
    calibrated_log_loss: float
    calibrated_wilson_lb: float
    calibrated_economic_edge: float
    
    # Veredicto
    verdict: CalibrationVerdict
    rejection_reason: str | None
    
    # Metadata
    params_json: dict
    metadata: dict


@dataclass(frozen=True, slots=True)
class PredictionVersioning:
    """Versionado completo de una predicción."""
    
    prediction_id: str
    model_version: str
    calibrator_version: str | None
    strategy_version: str
    evidence_policy_version: str
    applied_at: str


class CalibratorRepositoryV2:
    """Repositorio V2 con versionado real y sistema de rechazo."""
    
    def __init__(self, connection: pymysql.Connection) -> None:
        self._conn = connection
    
    def save_calibrator_v2(
        self,
        calibrator: ContextCalibrator,
        comparison: CalibrationComparison,
        description: str,
        metadata: dict,
    ) -> str:
        """Guarda un nuevo calibrador con veredicto y comparación completa."""
        
        # Desactivar calibradores anteriores
        with self._conn.cursor() as cursor:
            cursor.execute("UPDATE calibrators SET is_active = FALSE WHERE is_active = TRUE")
        
        # Generar nueva versión
        with self._conn.cursor() as cursor:
            cursor.execute("SELECT MAX(calibrator_id) FROM calibrators")
            result = cursor.fetchone()
            last_id = result[0] if result and result[0] else None
            
            if last_id and last_id.startswith("calibrator_v"):
                last_num = int(last_id.split("_v")[1])
                new_num = last_num + 1
            else:
                new_num = 1
            
            calibrator_id = f"calibrator_v{new_num}"
        
        # Serializar parámetros
        params_json = {}
        for context, params in calibrator._params.items():
            if params.is_valid:
                params_json[str(context)] = {
                    "method": params.method.value,
                    "params": params.params,
                    "n_train": params.n_train,
                    "train_brier": float(params.train_brier),
                    "train_ece": float(params.train_ece),
                }
        
        # Guardar calibrador
        with self._conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO calibrators (
                    calibrator_id, method, is_active, description,
                    train_samples, train_brier, train_ece,
                    test_samples, test_brier, test_ece,
                    params_json, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                calibrator_id,
                comparison.verdict.value if comparison.verdict == CalibrationVerdict.ACCEPTED else "rejected",
                comparison.verdict == CalibrationVerdict.ACCEPTED,
                description,
                comparison.n_train,
                comparison.raw_brier,
                comparison.raw_ece,
                comparison.n_test,
                comparison.calibrated_brier,
                comparison.calibrated_ece,
                json.dumps(params_json),
                json.dumps(metadata),
            ))
        
        # Guardar comparación raw vs calibrated
        with self._conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO calibration_comparisons (
                    calibrator_id, context_key,
                    raw_brier, raw_ece, raw_log_loss, raw_wilson_lb, raw_economic_edge,
                    calibrated_brier, calibrated_ece, calibrated_log_loss, calibrated_wilson_lb, calibrated_economic_edge,
                    brier_improvement, ece_improvement, log_loss_improvement, wilson_improvement, economic_impact,
                    is_accepted, rejection_reason
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                calibrator_id,
                comparison.context,
                comparison.raw_brier,
                comparison.raw_ece,
                comparison.raw_log_loss,
                comparison.raw_wilson_lb,
                comparison.raw_economic_edge,
                comparison.calibrated_brier,
                comparison.calibrated_ece,
                comparison.calibrated_log_loss,
                comparison.calibrated_wilson_lb,
                comparison.calibrated_economic_edge,
                comparison.brier_improvement,
                comparison.ece_improvement,
                comparison.log_loss_improvement,
                comparison.wilson_improvement,
                comparison.economic_impact,
                comparison.verdict == CalibrationVerdict.ACCEPTED,
                comparison.rejection_reason,
            ))
        
        # Si fue rechazado, registrar en tabla de rechazos
        if comparison.verdict == CalibrationVerdict.REJECTED:
            with self._conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO calibrator_rejections (
                        calibrator_id, rejection_reason,
                        train_brier, test_brier, brier_delta,
                        train_ece, test_ece, ece_delta,
                        economic_impact, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    calibrator_id,
                    comparison.rejection_reason,
                    comparison.raw_brier,
                    comparison.calibrated_brier,
                    comparison.brier_improvement,
                    comparison.raw_ece,
                    comparison.calibrated_ece,
                    comparison.ece_improvement,
                    comparison.economic_impact,
                    json.dumps(metadata),
                ))
        
        self._conn.commit()
        return calibrator_id
    
    def update_prediction_versioning(
        self,
        prediction_id: str,
        model_version: str,
        calibrator_version: str | None,
        strategy_version: str,
        evidence_policy_version: str,
    ) -> None:
        """Actualiza el versionado de una predicción."""
        with self._conn.cursor() as cursor:
            cursor.execute("""
                UPDATE market_predictions
                SET model_version = %s,
                    calibrator_version = %s,
                    strategy_version = %s,
                    evidence_policy_version = %s
                WHERE prediction_id = %s
            """, (model_version, calibrator_version, strategy_version, evidence_policy_version, prediction_id))
        
        self._conn.commit()
    
    def get_active_calibrator_v2(self) -> CalibratorVersionV2 | None:
        """Obtiene el calibrador activo con metadata completa."""
        with self._conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    c.calibrator_id, c.method, c.created_at, c.is_active, c.description,
                    c.train_samples, c.train_brier, c.train_ece,
                    c.test_samples, c.test_brier, c.test_ece,
                    c.params_json, c.metadata,
                    cc.raw_brier, cc.raw_ece, cc.raw_log_loss, cc.raw_wilson_lb, cc.raw_economic_edge,
                    cc.calibrated_brier, cc.calibrated_ece, cc.calibrated_log_loss, cc.calibrated_wilson_lb, cc.calibrated_economic_edge,
                    cc.is_accepted, cc.rejection_reason
                FROM calibrators c
                LEFT JOIN calibration_comparisons cc ON c.calibrator_id = cc.calibrator_id
                WHERE c.is_active = TRUE
                LIMIT 1
            """)
            result = cursor.fetchone()
            
            if not result:
                return None
            
            return CalibratorVersionV2(
                calibrator_id=result[0],
                method=CalibrationMethod(result[1]),
                created_at=str(result[2]),
                is_active=bool(result[3]),
                description=result[4],
                train_samples=result[5],
                val_samples=0,  # No stored directly
                test_samples=result[8],
                raw_brier=float(result[12]) if result[12] else 0.0,
                raw_ece=float(result[13]) if result[13] else 0.0,
                raw_log_loss=float(result[14]) if result[14] else 0.0,
                raw_wilson_lb=float(result[15]) if result[15] else 0.0,
                raw_economic_edge=float(result[16]) if result[16] else 0.0,
                calibrated_brier=float(result[17]) if result[17] else 0.0,
                calibrated_ece=float(result[18]) if result[18] else 0.0,
                calibrated_log_loss=float(result[19]) if result[19] else 0.0,
                calibrated_wilson_lb=float(result[20]) if result[20] else 0.0,
                calibrated_economic_edge=float(result[21]) if result[21] else 0.0,
                verdict=CalibrationVerdict.ACCEPTED if result[22] else CalibrationVerdict.REJECTED,
                rejection_reason=result[23],
                params_json=json.loads(result[10]) if result[10] else {},
                metadata=json.loads(result[11]) if result[11] else {},
            )
    
    def get_rejection_history(self) -> list[dict]:
        """Obtiene historial de rechazos."""
        with self._conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    calibrator_id, rejection_reason,
                    train_brier, test_brier, brier_delta,
                    train_ece, test_ece, ece_delta,
                    economic_impact, rejected_at, metadata
                FROM calibrator_rejections
                ORDER BY rejected_at DESC
            """)
            
            return [
                {
                    "calibrator_id": r[0],
                    "rejection_reason": r[1],
                    "train_brier": float(r[2]) if r[2] else None,
                    "test_brier": float(r[3]) if r[3] else None,
                    "brier_delta": float(r[4]) if r[4] else None,
                    "train_ece": float(r[5]) if r[5] else None,
                    "test_ece": float(r[6]) if r[6] else None,
                    "ece_delta": float(r[7]) if r[7] else None,
                    "economic_impact": float(r[8]) if r[8] else None,
                    "rejected_at": str(r[9]),
                    "metadata": json.loads(r[10]) if r[10] else {},
                }
                for r in cursor.fetchall()
            ]