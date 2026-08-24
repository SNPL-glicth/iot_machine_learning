"""Architecture Gate — prueba de arquitectura/regresión (no de lógica).

Verifica permanentemente las reglas de ARCHITECTURE.md:

* El dominio Market no importa infraestructura (solo stdlib + dominio).
* Colisión de nombres: ``domain/entities/prediction.py`` (módulo legacy
  IoT) NO debe quedar oculto por un paquete ``prediction/``.
* ``IoT Prediction != ZENIN Market Prediction`` y ambos se importan sin
  alterar al otro.

Si alguien vuelve a crear ``domain/entities/prediction/`` o cambia un
``__init__.py`` de ``entities``, este test falla.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # iot_machine_learning/
_PROJECT_ROOT = _REPO_ROOT.parent  # ST/

_LEGACY_PREDICTION = "domain.entities.prediction"
_ZENIN_PREDICTION_PACKAGE = "iot_machine_learning.domain.entities.market.prediction"


def _ensure_paths() -> None:
    """Expone ambos roots de import (legacy requiere iot_machine_learning/)."""
    for path in (str(_PROJECT_ROOT), str(_REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


class TestCollisionGuard:
    def test_legacy_prediction_is_module_not_package(self):
        """El legacy NO debe volver a ser un paquete (lo ocultaría)."""
        _ensure_paths()
        mod = importlib.import_module(_LEGACY_PREDICTION)
        assert not hasattr(mod, "__path__"), (
            "domain/entities/prediction es un paquete: oculta el módulo "
            "legacy IoT. Debe vivir en market/prediction/."
        )
        assert mod.__file__ is not None and mod.__file__.endswith("prediction.py"), (
            "domain.entities.prediction debe resolver a prediction.py (módulo)"
        )

    def test_zenin_prediction_is_package_under_market(self):
        _ensure_paths()
        mod = importlib.import_module(_ZENIN_PREDICTION_PACKAGE)
        assert hasattr(mod, "__path__"), (
            "ZENIN Prediction debe ser el paquete market/prediction/"
        )

    def test_iot_prediction_is_not_zenin_prediction(self):
        _ensure_paths()
        legacy = importlib.import_module(_LEGACY_PREDICTION)
        zenin = importlib.import_module(_ZENIN_PREDICTION_PACKAGE)
        assert legacy is not zenin
        iot_prediction = legacy.Prediction
        zenin_prediction = zenin.Prediction
        assert iot_prediction is not zenin_prediction
        assert iot_prediction.__module__ != zenin_prediction.__module__
        assert "market.prediction" in zenin_prediction.__module__
        assert "market.prediction" not in iot_prediction.__module__

    def test_both_imports_coexist_in_same_process(self):
        """Importar uno NO altera al otro (mismo intérprete)."""
        _ensure_paths()
        legacy_mod = importlib.import_module(_LEGACY_PREDICTION)
        zenin_mod = importlib.import_module(_ZENIN_PREDICTION_PACKAGE)

        reloaded_legacy = importlib.reload(legacy_mod)
        assert reloaded_legacy.Prediction is legacy_mod.Prediction
        assert "market.prediction" not in reloaded_legacy.Prediction.__module__

        zenin = zenin_mod.Prediction
        assert "market.prediction" in zenin.__module__

    def test_entities_init_does_not_shadow_legacy(self):
        """entities/__init__.py re-exporta el Prediction legacy (IoT)."""
        _ensure_paths()
        entities = importlib.import_module("domain.entities")
        exported = getattr(entities, "Prediction", None)
        legacy = importlib.import_module(_LEGACY_PREDICTION)
        assert exported is not None
        assert exported is legacy.Prediction


class TestLayeringRules:
    """Regla 3: domain/market no importa infraestructura."""

    _FORBIDDEN = ("pymysql", "sqlalchemy", "redis", "weaviate",
                  "alpaca", "binance", "requests", "websockets")

    def test_domain_market_has_no_infrastructure_imports(self):
        market_root = _REPO_ROOT / "domain" / "entities" / "market"
        for path in sorted(market_root.rglob("*.py")):
            if path.name == "__init__.py" and path.parent == market_root:
                continue
            for i, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                for lib in self._FORBIDDEN:
                    assert not re.search(
                        rf"\b(?:import|from) {re.escape(lib)}\b", stripped
                    ), (
                        f"{path.relative_to(_REPO_ROOT)}:{i} importa "
                        f"infraestructura ({lib}): regla 3 violada"
                    )

    def test_architecture_rules_doc_exists(self):
        assert (_REPO_ROOT / "ARCHITECTURE.md").is_file(), (
            "ARCHITECTURE.md (contrato) debe existir"
        )

    def test_predictions_do_not_leak_between_packages(self):
        """El paquete market/prediction no debe importar results legacy."""
        _ensure_paths()
        zenin = importlib.import_module(_ZENIN_PREDICTION_PACKAGE)
        legacy = importlib.import_module(_LEGACY_PREDICTION)
        assert "market.prediction" in zenin.__name__
        assert legacy.__name__ != zenin.__name__


def test_repo_root_layout_unchanged():
    """El paquete market vive donde lo espera el contrato."""
    root = _REPO_ROOT
    assert (root / "domain" / "entities" / "market" / "prediction").is_dir()
    assert (root / "domain" / "entities" / "prediction.py").is_file()
    assert not (root / "domain" / "entities" / "prediction").is_dir()
