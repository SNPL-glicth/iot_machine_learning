"""FASE 10.1 — Calibrator Repository: persistencia de calibradores versionados.

Este repositorio maneja:
- Guardar calibradores con versiones (calibrator_v1, calibrator_v2, etc.)
- Tracking de qué calibrador se usó en cada predicción
- Cargar calibradores activos por contexto
- Historial de versiones para auditoría

Reglas:
- Nunca modificar calibradores existentes, crear nueva versión
- Solo un calibrador activo a la vez
- Versionado secuencial: calibrator_v1, calibrator_v2, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final
from enum import Enum

import pymysql

from iot_machine_learning.domain.entities.market.calibration import (
    CalibrationMethod,
    CalibrationParams,
    ContextCalibrator,
    ContextKey,
)


__all__ = [
    "CalibratorVersion",
    "CalibratorMetadata",
    "CalibratorRepository",
]


@dataclass(frozen=True, slots=True)
class CalibratorVersion:
    """Versión de un calibrador."""
    
    calibrator_id: str
    method: CalibrationMethod
    created_at: str
    is_active: bool
    description: str | None
    train_samples: int | None
    train_brier: float | None
    train_ece: float | None
    test_samples: int | None
    test_brier: float | None
    test_ece: float | None
    params_json: dict
    metadata: dict


@dataclass(frozen=True, slots=True)
class CalibratorMetadata:
    """Metadata de un calibrador."""
    
    calibrator_id: str
    method: CalibrationMethod
    description: str
    train_samples: int
    test_samples: int
    symbols: list[str]
    train_ratio: float


class CalibratorRepository:
    """Repositorio para calibradores versionados."""
    
    def __init__(self, connection: pymysql.Connection) -> None:
        self._conn = connection
    
    def save_calibrator(
        self,
        calibrator: ContextCalibrator,
        metadata: CalibratorMetadata,
        description: str,
        test_metrics: dict[str, float] | None = None,
    ) -> str:
        """Guarda un nuevo calibrador con versión.
        
        Genera ID automáticamente: calibrator_v1, calibrator_v2, etc.
        """
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
                metadata.method.value,
                True,
                description,
                metadata.train_samples,
                None,  # train_brier (se calcula del calibrador)
                None,  # train_ece
                metadata.test_samples,
                test_metrics.get("brier") if test_metrics else None,
                test_metrics.get("ece") if test_metrics else None,
                json.dumps(params_json),
                json.dumps({
                    "symbols": metadata.symbols,
                    "train_ratio": metadata.train_ratio,
                }),
            ))
        
        self._conn.commit()
        return calibrator_id
    
    def get_active_calibrator(self) -> CalibratorVersion | None:
        """Obtiene el calibrador activo."""
        with self._conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    calibrator_id, method, created_at, is_active, description,
                    train_samples, train_brier, train_ece,
                    test_samples, test_brier, test_ece,
                    params_json, metadata
                FROM calibrators
                WHERE is_active = TRUE
                LIMIT 1
            """)
            result = cursor.fetchone()
            
            if not result:
                return None
            
            return CalibratorVersion(
                calibrator_id=result[0],
                method=CalibrationMethod(result[1]),
                created_at=str(result[2]),
                is_active=bool(result[3]),
                description=result[4],
                train_samples=result[5],
                train_brier=float(result[6]) if result[6] else None,
                train_ece=float(result[7]) if result[7] else None,
                test_samples=result[8],
                test_brier=float(result[9]) if result[9] else None,
                test_ece=float(result[10]) if result[10] else None,
                params_json=json.loads(result[11]) if result[11] else {},
                metadata=json.loads(result[12]) if result[12] else {},
            )
    
    def get_calibrator(self, calibrator_id: str) -> CalibratorVersion | None:
        """Obtiene un calibrador específico por ID."""
        with self._conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    calibrator_id, method, created_at, is_active, description,
                    train_samples, train_brier, train_ece,
                    test_samples, test_brier, test_ece,
                    params_json, metadata
                FROM calibrators
                WHERE calibrator_id = %s
            """, (calibrator_id,))
            result = cursor.fetchone()
            
            if not result:
                return None
            
            return CalibratorVersion(
                calibrator_id=result[0],
                method=CalibrationMethod(result[1]),
                created_at=str(result[2]),
                is_active=bool(result[3]),
                description=result[4],
                train_samples=result[5],
                train_brier=float(result[6]) if result[6] else None,
                train_ece=float(result[7]) if result[7] else None,
                test_samples=result[8],
                test_brier=float(result[9]) if result[9] else None,
                test_ece=float(result[10]) if result[10] else None,
                params_json=json.loads(result[11]) if result[11] else {},
                metadata=json.loads(result[12]) if result[12] else {},
            )
    
    def list_calibrators(self) -> list[CalibratorVersion]:
        """Lista todos los calibradores."""
        with self._conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    calibrator_id, method, created_at, is_active, description,
                    train_samples, train_brier, train_ece,
                    test_samples, test_brier, test_ece,
                    params_json, metadata
                FROM calibrators
                ORDER BY created_at DESC
            """)
            results = cursor.fetchall()
            
            return [
                CalibratorVersion(
                    calibrator_id=r[0],
                    method=CalibrationMethod(r[1]),
                    created_at=str(r[2]),
                    is_active=bool(r[3]),
                    description=r[4],
                    train_samples=r[5],
                    train_brier=float(r[6]) if r[6] else None,
                    train_ece=float(r[7]) if r[7] else None,
                    test_samples=r[8],
                    test_brier=float(r[9]) if r[9] else None,
                    test_ece=float(r[10]) if r[10] else None,
                    params_json=json.loads(r[11]) if r[11] else {},
                    metadata=json.loads(r[12]) if r[12] else {},
                )
                for r in results
            ]
    
    def track_prediction_calibration(
        self,
        prediction_id: str,
        calibrator_id: str | None,
        prob_raw: float,
        prob_calibrated: float | None,
        context_key: str,
    ) -> None:
        """Registra qué calibración se aplicó a una predicción."""
        with self._conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO prediction_calibrators
                (prediction_id, calibrator_id, prob_raw, prob_calibrated, context_key)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                calibrator_id = VALUES(calibrator_id),
                prob_raw = VALUES(prob_raw),
                prob_calibrated = VALUES(prob_calibrated),
                context_key = VALUES(context_key)
            """, (prediction_id, calibrator_id, prob_raw, prob_calibrated, context_key))
        
        self._conn.commit()
    
    def restore_calibrator(self, version: CalibratorVersion) -> ContextCalibrator:
        """Restaura un calibrador desde una versión guardada."""
        calibrator = ContextCalibrator(method=version.method)
        
        # Reconstruir parámetros
        for context_str, params_data in version.params_json.items():
            # Parse context key: "strategy·horizon·regime"
            parts = context_str.split("·")
            if len(parts) >= 3:
                strategy = parts[0]
                horizon = int(parts[1].replace("s", ""))
                regime = parts[2]
                context = ContextKey(strategy=strategy, horizon_seconds=horizon, regime=regime)
                
                # Reconstruir CalibrationParams
                params = CalibrationParams(
                    method=CalibrationMethod(params_data["method"]),
                    params=tuple(params_data["params"]),
                    n_train=params_data["n_train"],
                    train_brier=params_data["train_brier"],
                    train_ece=params_data["train_ece"],
                )
                calibrator._params[context] = params
        
        return calibrator