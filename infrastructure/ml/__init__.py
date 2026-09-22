"""Implementaciones ML de infraestructura."""

import sys
import domain.ports.iot_ports as _iot_ports

# Backward compatibility aliases for relocated ports
sys.modules.setdefault("infrastructure.ml.ports", _iot_ports)
sys.modules.setdefault("infrastructure.ml.ports.iot_ports", _iot_ports)
sys.modules.setdefault("iot_machine_learning.infrastructure.ml.ports", _iot_ports)
sys.modules.setdefault("iot_machine_learning.infrastructure.ml.ports.iot_ports", _iot_ports)
