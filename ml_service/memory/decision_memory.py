"""Módulo de Memoria de Decisiones para ML.

FALENCIA 5: No hay registro de qué decisiones se tomaron ante eventos similares
y cuál fue el resultado.

Este módulo implementa:
- Registro de decisiones tomadas y sus resultados
- Detección de patrones recurrentes
- Aprendizaje de resoluciones efectivas
- Sugerencias basadas en historial
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Registro de una decisión tomada."""
    
    id: Optional[int]
    
    # Identificación del evento
    sensor_id: int
    device_id: int
    event_type: str
    event_code: str
    
    # Firma del patrón (para matching)
    pattern_signature: str
    
    # Contexto del evento
    sensor_type: str
    severity: str
    trend: str
    anomaly_score: float
    predicted_value: float
    
    # Decisión tomada
    decision_type: str  # 'automated', 'manual', 'escalated'
    actions_taken: list[str]
    
    # Resultado
    resolution_status: str  # 'resolved', 'recurring', 'escalated', 'pending'
    resolution_time_minutes: Optional[int]
    root_cause_identified: Optional[str]
    was_effective: Optional[bool]
    
    # Feedback
    user_feedback: Optional[str]  # 'helpful', 'not_helpful', 'incorrect'
    feedback_notes: Optional[str]
    
    # Timestamps
    created_at: datetime
    resolved_at: Optional[datetime]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sensor_id": self.sensor_id,
            "device_id": self.device_id,
            "event_type": self.event_type,
            "event_code": self.event_code,
            "pattern_signature": self.pattern_signature,
            "sensor_type": self.sensor_type,
            "severity": self.severity,
            "trend": self.trend,
            "anomaly_score": self.anomaly_score,
            "predicted_value": self.predicted_value,
            "decision_type": self.decision_type,
            "actions_taken": self.actions_taken,
            "resolution_status": self.resolution_status,
            "resolution_time_minutes": self.resolution_time_minutes,
            "root_cause_identified": self.root_cause_identified,
            "was_effective": self.was_effective,
            "user_feedback": self.user_feedback,
            "feedback_notes": self.feedback_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class PatternMatch:
    """Coincidencia de patrón encontrada en el historial."""
    
    pattern_signature: str
    match_count: int
    avg_resolution_time_minutes: Optional[float]
    common_root_causes: list[str]
    effective_actions: list[str]
    success_rate: float  # 0.0 - 1.0
    last_occurrence: Optional[datetime]
    
    def to_dict(self) -> dict:
        return {
            "pattern_signature": self.pattern_signature,
            "match_count": self.match_count,
            "avg_resolution_time_minutes": self.avg_resolution_time_minutes,
            "common_root_causes": self.common_root_causes,
            "effective_actions": self.effective_actions,
            "success_rate": self.success_rate,
            "last_occurrence": self.last_occurrence.isoformat() if self.last_occurrence else None,
        }


@dataclass
class HistoricalInsight:
    """Insight derivado del historial de decisiones."""
    
    has_history: bool
    similar_events_count: int
    pattern_match: Optional[PatternMatch]
    
    # Sugerencias basadas en historial
    suggested_actions: list[str]
    suggested_root_cause: Optional[str]
    estimated_resolution_time: Optional[int]
    
    # Alertas
    is_recurring_issue: bool
    recurrence_frequency: Optional[str]  # 'daily', 'weekly', 'monthly'
    escalation_recommended: bool
    
    confidence: float
    
    def to_dict(self) -> dict:
        return {
            "has_history": self.has_history,
            "similar_events_count": self.similar_events_count,
            "pattern_match": self.pattern_match.to_dict() if self.pattern_match else None,
            "suggested_actions": self.suggested_actions,
            "suggested_root_cause": self.suggested_root_cause,
            "estimated_resolution_time": self.estimated_resolution_time,
            "is_recurring_issue": self.is_recurring_issue,
            "recurrence_frequency": self.recurrence_frequency,
            "escalation_recommended": self.escalation_recommended,
            "confidence": self.confidence,
        }


class DecisionMemory:
    """Sistema de memoria de decisiones para aprendizaje continuo.
    
    FUNCIONALIDADES:
    1. Registrar decisiones y sus resultados
    2. Buscar patrones similares en el historial
    3. Sugerir acciones basadas en resoluciones efectivas
    4. Detectar problemas recurrentes
    """
    
    def __init__(self, conn: Connection):
        self._conn = conn
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self) -> None:
        """Verifica que las tablas de memoria existan, las crea si no."""
        try:
            # Verificar si la tabla existe
            exists = self._conn.execute(
                text("""
                    SELECT 1 FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = 'dbo' 
                    AND TABLE_NAME = 'ml_decision_memory'
                """)
            ).fetchone()
            
            if not exists:
                self._create_tables()
        except Exception as e:
            logger.warning("[DECISION_MEMORY] Could not verify/create tables: %s", str(e))
    
    def _create_tables(self) -> None:
        """Crea las tablas de memoria de decisiones."""
        try:
            self._conn.execute(text("""
                IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES 
                               WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'ml_decision_memory')
                BEGIN
                    CREATE TABLE dbo.ml_decision_memory (
                        id BIGINT PRIMARY KEY IDENTITY(1,1),
                        
                        -- Identificación
                        sensor_id BIGINT NOT NULL,
                        device_id BIGINT NOT NULL,
                        event_type VARCHAR(50) NOT NULL,
                        event_code VARCHAR(100) NOT NULL,
                        
                        -- Firma del patrón
                        pattern_signature VARCHAR(64) NOT NULL,
                        
                        -- Contexto
                        sensor_type VARCHAR(100),
                        severity VARCHAR(50),
                        trend VARCHAR(50),
                        anomaly_score DECIMAL(10,4),
                        predicted_value DECIMAL(15,5),
                        
                        -- Decisión
                        decision_type VARCHAR(50),
                        actions_taken NVARCHAR(MAX),
                        
                        -- Resultado
                        resolution_status VARCHAR(50) DEFAULT 'pending',
                        resolution_time_minutes INT,
                        root_cause_identified NVARCHAR(500),
                        was_effective BIT,
                        
                        -- Feedback
                        user_feedback VARCHAR(50),
                        feedback_notes NVARCHAR(MAX),
                        
                        -- Timestamps
                        created_at DATETIME2 NOT NULL DEFAULT GETDATE(),
                        resolved_at DATETIME2,
                        
                        -- Índices
                        INDEX IX_decision_memory_pattern (pattern_signature),
                        INDEX IX_decision_memory_sensor (sensor_id),
                        INDEX IX_decision_memory_device (device_id),
                        INDEX IX_decision_memory_created (created_at)
                    );
                END
            """))
            
            self._conn.execute(text("""
                IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES 
                               WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'ml_decision_patterns')
                BEGIN
                    CREATE TABLE dbo.ml_decision_patterns (
                        id BIGINT PRIMARY KEY IDENTITY(1,1),
                        
                        -- Identificación del patrón
                        pattern_signature VARCHAR(64) NOT NULL UNIQUE,
                        
                        -- Características del patrón
                        sensor_type VARCHAR(100),
                        severity VARCHAR(50),
                        trend VARCHAR(50),
                        
                        -- Estadísticas
                        occurrence_count INT NOT NULL DEFAULT 1,
                        successful_resolutions INT NOT NULL DEFAULT 0,
                        avg_resolution_time_minutes INT,
                        
                        -- Conocimiento acumulado
                        common_root_causes NVARCHAR(MAX),
                        effective_actions NVARCHAR(MAX),
                        
                        -- Timestamps
                        first_seen_at DATETIME2 NOT NULL DEFAULT GETDATE(),
                        last_seen_at DATETIME2 NOT NULL DEFAULT GETDATE(),
                        
                        INDEX IX_decision_patterns_sensor_type (sensor_type),
                        INDEX IX_decision_patterns_last_seen (last_seen_at)
                    );
                END
            """))
            
            logger.info("[DECISION_MEMORY] Tables created successfully")
        except Exception as e:
            logger.error("[DECISION_MEMORY] Failed to create tables: %s", str(e))
    
    def generate_pattern_signature(
        self,
        *,
        sensor_type: str,
        severity: str,
        trend: str,
        event_code: str,
    ) -> str:
        """Genera una firma única para el patrón del evento.
        
        La firma permite identificar eventos similares para aprendizaje.
        """
        pattern_str = f"{sensor_type}|{severity}|{trend}|{event_code}"
        return hashlib.sha256(pattern_str.encode()).hexdigest()[:16]
    
    def record_decision(
        self,
        *,
        sensor_id: int,
        device_id: int,
        event_type: str,
        event_code: str,
        sensor_type: str,
        severity: str,
        trend: str,
        anomaly_score: float,
        predicted_value: float,
        decision_type: str,
        actions_taken: list[str],
    ) -> Optional[int]:
        """Registra una nueva decisión en la memoria."""
        
        pattern_sig = self.generate_pattern_signature(
            sensor_type=sensor_type,
            severity=severity,
            trend=trend,
            event_code=event_code,
        )
        
        try:
            row = self._conn.execute(
                text("""
                    INSERT INTO dbo.ml_decision_memory (
                        sensor_id, device_id, event_type, event_code,
                        pattern_signature, sensor_type, severity, trend,
                        anomaly_score, predicted_value, decision_type, actions_taken
                    )
                    OUTPUT INSERTED.id
                    VALUES (
                        :sensor_id, :device_id, :event_type, :event_code,
                        :pattern_sig, :sensor_type, :severity, :trend,
                        :anomaly_score, :predicted_value, :decision_type, :actions_taken
                    )
                """),
                {
                    "sensor_id": sensor_id,
                    "device_id": device_id,
                    "event_type": event_type,
                    "event_code": event_code,
                    "pattern_sig": pattern_sig,
                    "sensor_type": sensor_type,
                    "severity": severity,
                    "trend": trend,
                    "anomaly_score": anomaly_score,
                    "predicted_value": predicted_value,
                    "decision_type": decision_type,
                    "actions_taken": json.dumps(actions_taken, ensure_ascii=False),
                },
            ).fetchone()
            
            if row:
                decision_id = int(row[0])
                
                # Actualizar patrón
                self._update_pattern(
                    pattern_signature=pattern_sig,
                    sensor_type=sensor_type,
                    severity=severity,
                    trend=trend,
                )
                
                logger.debug(
                    "[DECISION_MEMORY] Recorded decision id=%s pattern=%s",
                    decision_id, pattern_sig
                )
                return decision_id
        
        except Exception as e:
            logger.warning("[DECISION_MEMORY] Failed to record decision: %s", str(e))
        
        return None
    
    def update_resolution(
        self,
        decision_id: int,
        *,
        resolution_status: str,
        resolution_time_minutes: Optional[int] = None,
        root_cause_identified: Optional[str] = None,
        was_effective: Optional[bool] = None,
    ) -> bool:
        """Actualiza el resultado de una decisión."""
        
        try:
            self._conn.execute(
                text("""
                    UPDATE dbo.ml_decision_memory
                    SET 
                        resolution_status = :status,
                        resolution_time_minutes = :time,
                        root_cause_identified = :cause,
                        was_effective = :effective,
                        resolved_at = CASE WHEN :status IN ('resolved', 'escalated') THEN GETDATE() ELSE resolved_at END
                    WHERE id = :id
                """),
                {
                    "id": decision_id,
                    "status": resolution_status,
                    "time": resolution_time_minutes,
                    "cause": root_cause_identified,
                    "effective": 1 if was_effective else (0 if was_effective is False else None),
                },
            )
            
            # Si fue efectivo, actualizar patrón con conocimiento
            if was_effective and root_cause_identified:
                self._update_pattern_knowledge(decision_id, root_cause_identified)
            
            return True
        
        except Exception as e:
            logger.warning("[DECISION_MEMORY] Failed to update resolution: %s", str(e))
            return False
    
    def record_feedback(
        self,
        decision_id: int,
        *,
        feedback: str,
        notes: Optional[str] = None,
    ) -> bool:
        """Registra feedback del usuario sobre una decisión."""
        
        try:
            self._conn.execute(
                text("""
                    UPDATE dbo.ml_decision_memory
                    SET user_feedback = :feedback, feedback_notes = :notes
                    WHERE id = :id
                """),
                {"id": decision_id, "feedback": feedback, "notes": notes},
            )
            return True
        except Exception:
            return False
    
    def get_historical_insight(
        self,
        *,
        sensor_type: str,
        severity: str,
        trend: str,
        event_code: str,
    ) -> HistoricalInsight:
        """Obtiene insights del historial para un patrón dado."""
        
        pattern_sig = self.generate_pattern_signature(
            sensor_type=sensor_type,
            severity=severity,
            trend=trend,
            event_code=event_code,
        )
        
        # Buscar patrón en historial
        pattern_match = self._find_pattern_match(pattern_sig)
        
        # Contar eventos similares
        similar_count = self._count_similar_events(pattern_sig)
        
        # Determinar si es recurrente
        is_recurring, frequency = self._check_recurrence(pattern_sig)
        
        # Generar sugerencias
        suggested_actions = []
        suggested_cause = None
        estimated_time = None
        
        if pattern_match:
            suggested_actions = pattern_match.effective_actions[:3]
            if pattern_match.common_root_causes:
                suggested_cause = pattern_match.common_root_causes[0]
            estimated_time = int(pattern_match.avg_resolution_time_minutes) if pattern_match.avg_resolution_time_minutes else None
        
        # Determinar si se recomienda escalamiento
        escalation_recommended = (
            is_recurring 
            and similar_count >= 5 
            and (not pattern_match or pattern_match.success_rate < 0.5)
        )
        
        # Calcular confianza
        confidence = self._calculate_insight_confidence(pattern_match, similar_count)
        
        return HistoricalInsight(
            has_history=similar_count > 0,
            similar_events_count=similar_count,
            pattern_match=pattern_match,
            suggested_actions=suggested_actions,
            suggested_root_cause=suggested_cause,
            estimated_resolution_time=estimated_time,
            is_recurring_issue=is_recurring,
            recurrence_frequency=frequency,
            escalation_recommended=escalation_recommended,
            confidence=confidence,
        )
    
    def _update_pattern(
        self,
        *,
        pattern_signature: str,
        sensor_type: str,
        severity: str,
        trend: str,
    ) -> None:
        """Actualiza o crea un registro de patrón."""
        try:
            self._conn.execute(
                text("""
                    MERGE dbo.ml_decision_patterns AS target
                    USING (SELECT :sig AS pattern_signature) AS source
                    ON target.pattern_signature = source.pattern_signature
                    WHEN MATCHED THEN
                        UPDATE SET 
                            occurrence_count = occurrence_count + 1,
                            last_seen_at = GETDATE()
                    WHEN NOT MATCHED THEN
                        INSERT (pattern_signature, sensor_type, severity, trend)
                        VALUES (:sig, :sensor_type, :severity, :trend);
                """),
                {
                    "sig": pattern_signature,
                    "sensor_type": sensor_type,
                    "severity": severity,
                    "trend": trend,
                },
            )
        except Exception:
            pass
    
    def _update_pattern_knowledge(
        self,
        decision_id: int,
        root_cause: str,
    ) -> None:
        """Actualiza el conocimiento del patrón con una resolución exitosa."""
        try:
            # Obtener patrón de la decisión
            row = self._conn.execute(
                text("""
                    SELECT pattern_signature, actions_taken
                    FROM dbo.ml_decision_memory
                    WHERE id = :id
                """),
                {"id": decision_id},
            ).fetchone()
            
            if not row:
                return
            
            pattern_sig = row.pattern_signature
            actions = row.actions_taken
            
            # Actualizar patrón con conocimiento
            self._conn.execute(
                text("""
                    UPDATE dbo.ml_decision_patterns
                    SET 
                        successful_resolutions = successful_resolutions + 1,
                        common_root_causes = CASE 
                            WHEN common_root_causes IS NULL THEN :cause
                            WHEN common_root_causes NOT LIKE '%' + :cause + '%' 
                                THEN common_root_causes + '|' + :cause
                            ELSE common_root_causes
                        END,
                        effective_actions = CASE
                            WHEN effective_actions IS NULL THEN :actions
                            ELSE effective_actions
                        END
                    WHERE pattern_signature = :sig
                """),
                {
                    "sig": pattern_sig,
                    "cause": root_cause,
                    "actions": actions,
                },
            )
        except Exception:
            pass
    
    def _find_pattern_match(self, pattern_signature: str) -> Optional[PatternMatch]:
        """Busca un patrón en el historial."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT 
                        pattern_signature,
                        occurrence_count,
                        avg_resolution_time_minutes,
                        common_root_causes,
                        effective_actions,
                        successful_resolutions,
                        last_seen_at
                    FROM dbo.ml_decision_patterns
                    WHERE pattern_signature = :sig
                """),
                {"sig": pattern_signature},
            ).fetchone()
            
            if row:
                # Parsear causas y acciones
                causes = []
                if row.common_root_causes:
                    causes = [c.strip() for c in str(row.common_root_causes).split("|") if c.strip()]
                
                actions = []
                if row.effective_actions:
                    try:
                        actions = json.loads(row.effective_actions)
                    except:
                        actions = [row.effective_actions]
                
                # Calcular tasa de éxito
                success_rate = 0.0
                if row.occurrence_count > 0:
                    success_rate = (row.successful_resolutions or 0) / row.occurrence_count
                
                return PatternMatch(
                    pattern_signature=row.pattern_signature,
                    match_count=int(row.occurrence_count),
                    avg_resolution_time_minutes=float(row.avg_resolution_time_minutes) if row.avg_resolution_time_minutes else None,
                    common_root_causes=causes,
                    effective_actions=actions,
                    success_rate=success_rate,
                    last_occurrence=row.last_seen_at,
                )
        except Exception:
            pass
        
        return None
    
    def _count_similar_events(self, pattern_signature: str, days: int = 30) -> int:
        """Cuenta eventos similares en el período."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT COUNT(*) AS cnt
                    FROM dbo.ml_decision_memory
                    WHERE pattern_signature = :sig
                      AND created_at >= DATEADD(day, -:days, GETDATE())
                """),
                {"sig": pattern_signature, "days": days},
            ).fetchone()
            
            if row:
                return int(row.cnt)
        except Exception:
            pass
        
        return 0
    
    def _check_recurrence(self, pattern_signature: str) -> tuple[bool, Optional[str]]:
        """Verifica si el patrón es recurrente y su frecuencia."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT 
                        COUNT(*) AS total,
                        COUNT(CASE WHEN created_at >= DATEADD(day, -7, GETDATE()) THEN 1 END) AS last_week,
                        COUNT(CASE WHEN created_at >= DATEADD(day, -30, GETDATE()) THEN 1 END) AS last_month
                    FROM dbo.ml_decision_memory
                    WHERE pattern_signature = :sig
                """),
                {"sig": pattern_signature},
            ).fetchone()
            
            if row:
                last_week = int(row.last_week or 0)
                last_month = int(row.last_month or 0)
                
                if last_week >= 7:
                    return True, "daily"
                if last_week >= 3:
                    return True, "weekly"
                if last_month >= 4:
                    return True, "monthly"
                if last_month >= 2:
                    return True, None
        except Exception:
            pass
        
        return False, None
    
    def _calculate_insight_confidence(
        self,
        pattern_match: Optional[PatternMatch],
        similar_count: int,
    ) -> float:
        """Calcula la confianza del insight basado en datos disponibles."""
        
        if not pattern_match or similar_count == 0:
            return 0.0
        
        base = 0.3
        
        # Más ocurrencias = más confianza
        if similar_count >= 10:
            base += 0.3
        elif similar_count >= 5:
            base += 0.2
        elif similar_count >= 2:
            base += 0.1
        
        # Tasa de éxito alta = más confianza
        if pattern_match.success_rate >= 0.8:
            base += 0.2
        elif pattern_match.success_rate >= 0.5:
            base += 0.1
        
        # Tiene causas identificadas = más confianza
        if pattern_match.common_root_causes:
            base += 0.1
        
        return min(0.95, base)


def get_historical_insight_for_event(
    conn: Connection,
    *,
    sensor_type: str,
    severity: str,
    trend: str,
    event_code: str,
) -> HistoricalInsight:
    """Función de conveniencia para obtener insights históricos.
    
    Esta función es el punto de entrada principal para el batch runner.
    """
    memory = DecisionMemory(conn)
    return memory.get_historical_insight(
        sensor_type=sensor_type,
        severity=severity,
        trend=trend,
        event_code=event_code,
    )


def record_ml_decision(
    conn: Connection,
    *,
    sensor_id: int,
    device_id: int,
    event_type: str,
    event_code: str,
    sensor_type: str,
    severity: str,
    trend: str,
    anomaly_score: float,
    predicted_value: float,
    actions_taken: list[str],
) -> Optional[int]:
    """Función de conveniencia para registrar una decisión.
    
    Esta función es el punto de entrada principal para el batch runner.
    """
    memory = DecisionMemory(conn)
    return memory.record_decision(
        sensor_id=sensor_id,
        device_id=device_id,
        event_type=event_type,
        event_code=event_code,
        sensor_type=sensor_type,
        severity=severity,
        trend=trend,
        anomaly_score=anomaly_score,
        predicted_value=predicted_value,
        decision_type="automated",
        actions_taken=actions_taken,
    )
