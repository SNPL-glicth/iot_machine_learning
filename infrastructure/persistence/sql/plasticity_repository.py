"""SQL Server implementation of PlasticityRepositoryPort.

Simple, fail-safe persistence for PlasticityTracker state.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .....domain.ports.plasticity_repository_port import (
    PlasticityRepositoryPort,
    RegimeWeightState,
)

logger = logging.getLogger(__name__)


class SqlPlasticityRepository(PlasticityRepositoryPort):
    """SQL Server persistence for plasticity state.
    
    Stores regime-engine weights in dbo.plasticity_regime_weights.
    Uses MERGE for upsert operations.
    
    Args:
        engine: SQLAlchemy Engine for database connections
        
    Usage:
        engine = create_engine("mssql+pymssql://...")
        repo = SqlPlasticityRepository(engine)
        
        # Load state on startup
        states = repo.load_regime_state("STABLE", ["taylor", "baseline"])
        
        # Save after updates
        repo.save_regime_state([state1, state2])
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def load_regime_state(
        self,
        regime: str,
        engine_names: List[str],
    ) -> Dict[str, RegimeWeightState]:
        """Load saved state for a regime and list of engines.
        
        Returns empty dict if no state found or on error.
        """
        if not engine_names:
            return {}

        try:
            with self._engine.connect() as conn:
                # Use parameterized IN clause (SQL Server 2016+)
                placeholders = ", ".join([f":eng{i}" for i in range(len(engine_names))])
                params = {f"eng{i}": name for i, name in enumerate(engine_names)}
                params["regime"] = regime

                result = conn.execute(
                    text(f"""
                        SELECT 
                            regime,
                            engine_name,
                            accuracy,
                            prior_mu,
                            prior_sigma2,
                            DATEDIFF(SECOND, '1970-01-01', last_access_time) as last_access_ts,
                            DATEDIFF(SECOND, '1970-01-01', last_update_time) as last_update_ts
                        FROM plasticity_regime_weights
                        WHERE regime = :regime
                          AND engine_name IN ({placeholders})
                    """),
                    params,
                ).fetchall()

                states = {}
                for row in result:
                    key = f"{row.regime}|{row.engine_name}"
                    states[key] = RegimeWeightState(
                        regime=row.regime,
                        engine_name=row.engine_name,
                        accuracy=float(row.accuracy),
                        prior_mu=float(row.prior_mu),
                        prior_sigma2=float(row.prior_sigma2),
                        last_access_time=float(row.last_access_ts) if row.last_access_ts else 0.0,
                        last_update_time=float(row.last_update_ts) if row.last_update_ts else 0.0,
                    )

                logger.debug(
                    "plasticity_state_loaded",
                    extra={
                        "regime": regime,
                        "engines_requested": len(engine_names),
                        "engines_found": len(states),
                    },
                )
                return states

        except Exception as e:
            logger.warning(
                "plasticity_load_failed",
                extra={"regime": regime, "error": str(e)},
            )
            return {}

    def save_regime_state(
        self,
        states: List[RegimeWeightState],
    ) -> None:
        """Persist regime-engine states using MERGE (UPSERT)."""
        if not states:
            return

        try:
            with self._engine.begin() as conn:
                for state in states:
                    conn.execute(
                        text("""
                            MERGE plasticity_regime_weights AS target
                            USING (
                                SELECT 
                                    :regime AS regime,
                                    :engine_name AS engine_name
                            ) AS source
                            ON target.regime = source.regime 
                               AND target.engine_name = source.engine_name
                            WHEN MATCHED THEN
                                UPDATE SET
                                    accuracy = :accuracy,
                                    prior_mu = :prior_mu,
                                    prior_sigma2 = :prior_sigma2,
                                    last_access_time = DATEADD(SECOND, :last_access_time, '1970-01-01'),
                                    last_update_time = DATEADD(SECOND, :last_update_time, '1970-01-01'),
                                    updated_at = SYSDATETIME()
                            WHEN NOT MATCHED THEN
                                INSERT (
                                    regime, engine_name, accuracy,
                                    prior_mu, prior_sigma2,
                                    last_access_time, last_update_time,
                                    created_at, updated_at
                                )
                                VALUES (
                                    :regime, :engine_name, :accuracy,
                                    :prior_mu, :prior_sigma2,
                                    DATEADD(SECOND, :last_access_time, '1970-01-01'),
                                    DATEADD(SECOND, :last_update_time, '1970-01-01'),
                                    SYSDATETIME(),
                                    SYSDATETIME()
                                );
                        """),
                        {
                            "regime": state.regime,
                            "engine_name": state.engine_name,
                            "accuracy": state.accuracy,
                            "prior_mu": state.prior_mu,
                            "prior_sigma2": state.prior_sigma2,
                            "last_access_time": int(state.last_access_time),
                            "last_update_time": int(state.last_update_time),
                        },
                    )

            logger.debug(
                "plasticity_state_saved",
                extra={"regimes": len(set(s.regime for s in states)), "total_states": len(states)},
            )

        except Exception as e:
            logger.warning(
                "plasticity_save_failed",
                extra={"states_count": len(states), "error": str(e)},
            )
            # Fail-safe: don't raise, just log

    def list_stored_regimes(self) -> List[str]:
        """Return list of all regimes with stored state."""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT DISTINCT regime
                        FROM plasticity_regime_weights
                        ORDER BY regime
                    """),
                ).fetchall()

                return [row.regime for row in result]

        except Exception as e:
            logger.warning("plasticity_list_regimes_failed", extra={"error": str(e)})
            return []
