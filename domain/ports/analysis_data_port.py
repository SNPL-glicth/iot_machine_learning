"""AnalysisDataPort — acceso a resultados de análisis.

Puerto de dominio para consultar documentos analizados.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class AnalysisRecord:
    """Registro de documento analizado."""
    id: str
    filename: str
    severity: str
    domain: str
    urgency: float
    confidence: float
    conclusion: str
    entities: List[str]
    actions: List[str]
    analyzed_at: str


class AnalysisDataPort(ABC):
    """Puerto para acceder a datos de análisis."""
    
    @abstractmethod
    def get_critical(self, tenant_id: str, limit: int = 5) -> List[AnalysisRecord]:
        """Documentos con severity critical o warning."""
    
    @abstractmethod
    def get_recent(self, tenant_id: str, limit: int = 5) -> List[AnalysisRecord]:
        """Documentos más recientes."""
    
    @abstractmethod
    def get_by_domain(self, tenant_id: str, domain: str) -> List[AnalysisRecord]:
        """Documentos por dominio."""
    
    @abstractmethod
    def get_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Resumen: total, por severidad, promedio confianza."""
