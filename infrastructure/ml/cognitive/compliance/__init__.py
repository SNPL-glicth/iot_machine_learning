"""Compliance export subsystem (IMP-5).

Turns a :class:`PredictionResult` into a canonical, tamper-evident
:class:`ComplianceRecord` and optionally appends it to an NDJSON
audit sink. Opt-in via the ``ML_COMPLIANCE_EXPORT_PATH`` env var
(consumed by :class:`AssemblyPhase` as a post-hook).
"""

from .compliance_exporter import ComplianceExporter
from .compliance_record import ComplianceRecord

__all__ = ["ComplianceExporter", "ComplianceRecord"]
