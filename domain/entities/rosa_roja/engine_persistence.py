"""State persistence, lifecycle, and registration mixin for RosaRojaEngine."""

from __future__ import annotations

import logging
import time
from typing import Any
from .state_persistence import STATE_SCHEMA_VERSION
from domain.ports.rosa_roja.state_store import MLStateStore
from domain.ports.rosa_roja.expert_jury import ExpertJuryPort
from domain.ports.rosa_roja.drift_sensor import DriftSensorPort

logger = logging.getLogger(__name__)


class RosaRojaPersistenceMixin:
    """Provides snapshot serialization, recovery, and component lifecycle for RosaRojaEngine."""

    _engine_id: str
    _state_store: Any
    _ingestion: Any
    _rhythm: Any
    _state_machine: Any
    _tracker: Any
    _jury: list[Any]
    _sensors: list[Any]
    _consecutive_outliers: int
    _auto_resets: int
    exploration_boost_events: int

    def cancel_active_trajectory(self) -> None:
        if getattr(self, "_tracker", None):
            self._tracker.set_active_trajectory(None)
        if hasattr(getattr(self, "_state_machine", None), "on_trajectory_complete"):
            self._state_machine.on_trajectory_complete()

    def _trigger_auto_regime_reset(self) -> None:
        self._ingestion.reset()
        self._rhythm.boost_exploration(self.exploration_boost_events)
        self._tracker.set_active_trajectory(None)
        self._consecutive_outliers = 0

    def register_expert(self, expert: ExpertJuryPort) -> None:
        if expert not in self._jury:
            self._jury.append(expert)

    def register_drift_sensor(self, sensor: DriftSensorPort) -> None:
        if sensor not in self._sensors:
            self._sensors.append(sensor)

    def get_jury_status(self) -> dict:
        return {e.name: {"is_critical": e.is_critical, "threshold": e.threshold, "weight": e.weight} for e in self._jury}

    def get_drift_status(self) -> dict:
        return {s.name: {"drift_score": s.get_drift_score()} for s in self._sensors}

    def get_state_summary(self) -> dict:
        return dict(self._state_machine.get_state_summary())

    def export_state(self) -> dict[str, Any]:
        """Atomic snapshot of all learning state."""
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "engine_id": self._engine_id,
            "event_watermark": self._state_machine.state.total_events_processed,
            "saved_at": time.time(),
            "components": {
                "ingestion": self._ingestion.export_state(),
                "rhythm_generator": self._rhythm.export_state(),
                "state_machine": self._state_machine.export_state(),
                "jury": [{"name": e.name, "state": e.export_state()} for e in self._jury if hasattr(e, "export_state")],
                "sensors": [{"name": s.name, "state": s.export_state()} for s in self._sensors if hasattr(s, "export_state")],
            },
        }

    def import_state(self, payload: dict[str, Any]) -> None:
        """Restore learning state from snapshot."""
        if not isinstance(payload, dict) or payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported snapshot schema: {payload.get('schema_version') if isinstance(payload, dict) else type(payload)!r}")
        components = payload.get("components")
        if not isinstance(components, dict):
            raise ValueError("Engine snapshot missing 'components'")

        for req in ("ingestion", "rhythm_generator", "state_machine"):
            if req not in components:
                raise ValueError(f"Engine snapshot missing component: {req}")

        self._ingestion.import_state(components["ingestion"])
        self._rhythm.import_state(components["rhythm_generator"])
        self._state_machine.import_state(components["state_machine"])
        self._tracker.set_active_trajectory(None)

        for key, members in (("jury", self._jury), ("sensors", self._sensors)):
            saved = components.get(key, [])
            if not isinstance(saved, list):
                raise ValueError(f"Engine snapshot '{key}' must be a list")
            live_by_name = {m.name: m for m in members if hasattr(m, "import_state")}
            for entry in saved:
                if not isinstance(entry, dict) or "name" not in entry:
                    raise ValueError(f"Malformed entry in engine snapshot '{key}'")
                name = entry["name"]
                member = live_by_name.get(name)
                if member is None:
                    raise ValueError(f"Snapshot '{key}' member '{name}' has no live counterpart")
                if entry.get("state") is not None and hasattr(member, "import_state"):
                    member.import_state(entry["state"])

    def checkpoint(self) -> bool:
        if self._state_store is None:
            return False
        ok = self._state_store.save(self._engine_id, self.export_state())
        if not ok:
            logger.warning("Checkpoint failed for engine %s", self._engine_id)
        return bool(ok)

    def restore(self, state_store: MLStateStore | None = None) -> bool:
        store = state_store if state_store is not None else self._state_store
        if store is None:
            return False
        payload = store.load(self._engine_id)
        if payload is None:
            return False
        try:
            self.import_state(payload)
        except ValueError as exc:
            logger.error("Corrupt ML snapshot for %s (%s)", self._engine_id, exc)
            return False
        return True

    def reset(self) -> None:
        self._ingestion.reset()
        self._rhythm.reset()
        self._tracker.reset()
        self._consecutive_outliers = 0
        self._auto_resets = 0
        for sensor in self._sensors:
            sensor.reset()
        self._state_machine.reset()
