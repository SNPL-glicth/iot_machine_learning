"""Repository for decision engine outcomes."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
    ZeninDbConnection,
)

logger = logging.getLogger(__name__)


class DecisionOutcomesRepository:
    """Persists and loads decision outcomes for learning."""
    
    def __init__(self, engine: Optional[Engine] = None):
        self._engine = engine or ZeninDbConnection.get_engine()
        self._ensure_table()
    
    def _ensure_table(self) -> None:
        """Create table if not exists."""
        try:
            with self._engine.begin() as conn:
                conn.execute(text("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables 
                        WHERE name = 'decision_outcomes' 
                        AND schema_id = SCHEMA_ID('zenin_ml')
                    )
                    BEGIN
                        CREATE TABLE zenin_ml.decision_outcomes (
                            id INT IDENTITY PRIMARY KEY,
                            decision_id VARCHAR(100) NOT NULL UNIQUE,
                            domain VARCHAR(50) NOT NULL,
                            regime VARCHAR(50) NOT NULL,
                            action_taken VARCHAR(100) NOT NULL,
                            severity_declared VARCHAR(20) NOT NULL,
                            confidence_declared FLOAT NOT NULL,
                            outcome_correct BIT NULL,
                            feedback_at DATETIME NULL,
                            created_at DATETIME DEFAULT GETDATE()
                        )
                        CREATE INDEX IX_decision_outcomes_domain_regime 
                        ON zenin_ml.decision_outcomes(domain, regime)
                    END
                """))
        except Exception as exc:
            logger.warning(
                "decision_outcomes_table_creation_failed",
                extra={"error": str(exc)},
            )
    
    def save_decision(
        self,
        decision_id: str,
        domain: str,
        regime: str,
        action_taken: str,
        severity_declared: str,
        confidence_declared: float,
    ) -> None:
        """Save a decision outcome (outcome_correct=NULL initially).
        
        Args:
            decision_id: Unique decision identifier
            domain: Domain of the decision
            regime: Regime/context
            action_taken: Action decided
            severity_declared: Severity level
            confidence_declared: Confidence score
        """
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO zenin_ml.decision_outcomes 
                        (decision_id, domain, regime, action_taken, severity_declared, confidence_declared)
                        VALUES (:decision_id, :domain, :regime, :action, :severity, :confidence)
                    """),
                    {
                        "decision_id": decision_id,
                        "domain": domain,
                        "regime": regime,
                        "action": action_taken,
                        "severity": severity_declared,
                        "confidence": confidence_declared,
                    },
                )
            
            logger.debug(
                "decision_outcome_saved",
                extra={
                    "decision_id": decision_id,
                    "domain": domain,
                    "action": action_taken,
                },
            )
            
        except Exception as exc:
            logger.warning(
                "decision_outcome_save_failed",
                extra={
                    "decision_id": decision_id,
                    "error": str(exc)},
            )
    
    def record_feedback(
        self,
        decision_id: str,
        was_correct: bool,
    ) -> None:
        """Record feedback for a decision.
        
        Args:
            decision_id: Decision identifier
            was_correct: Whether the decision was correct
        """
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE zenin_ml.decision_outcomes
                        SET outcome_correct = :was_correct,
                            feedback_at = GETDATE()
                        WHERE decision_id = :decision_id
                    """),
                    {
                        "decision_id": decision_id,
                        "was_correct": was_correct,
                    },
                )
            
            logger.info(
                "decision_feedback_recorded",
                extra={
                    "decision_id": decision_id,
                    "was_correct": was_correct,
                },
            )
            
        except Exception as exc:
            logger.warning(
                "decision_feedback_failed",
                extra={
                    "decision_id": decision_id,
                    "error": str(exc)},
            )
    
    def get_recent_precision(
        self,
        domain: str,
        regime: str,
        limit: int = 20,
    ) -> Optional[float]:
        """Get recent precision for a domain/regime.
        
        Args:
            domain: Domain to query
            regime: Regime to query
            limit: Number of recent decisions to consider
            
        Returns:
            Precision (0.0-1.0) or None if insufficient data
        """
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text("""
                        SELECT 
                            COUNT(*) as total,
                            SUM(CASE WHEN outcome_correct = 1 THEN 1 ELSE 0 END) as correct
                        FROM (
                            SELECT TOP (:limit) outcome_correct
                            FROM zenin_ml.decision_outcomes
                            WHERE domain = :domain 
                              AND regime = :regime
                              AND outcome_correct IS NOT NULL
                            ORDER BY created_at DESC
                        ) recent
                    """),
                    {
                        "domain": domain,
                        "regime": regime,
                        "limit": limit,
                    },
                ).fetchone()
            
            if result and result[0] > 0:
                precision = result[1] / result[0]
                logger.debug(
                    "decision_precision_calculated",
                    extra={
                        "domain": domain,
                        "regime": regime,
                        "precision": round(precision, 4),
                        "n_decisions": result[0],
                    },
                )
                return precision
            
            return None
            
        except Exception as exc:
            logger.warning(
                "decision_precision_query_failed",
                extra={
                    "domain": domain,
                    "regime": regime,
                    "error": str(exc)},
            )
            return None
