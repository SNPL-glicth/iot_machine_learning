"""Port de memoria cognitiva — contrato para almacenamiento semántico.

Define el contrato que la capa de infraestructura debe implementar
para persistir y consultar artefactos cognitivos (explicaciones,
anomalías, patrones, razonamiento de decisiones).

El dominio emite "memorias" y las consulta por similitud semántica.
La infraestructura decide DÓNDE almacenarlas (Weaviate, Elasticsearch,
Pinecone, /dev/null).

Este port es OPCIONAL — si no se vincula una implementación, el sistema
opera idénticamente a antes (modo SQL-only vía ``NullCognitiveAdapter``).

Principios de diseño:
    - Ningún tipo de Weaviate aparece en este archivo.
    - Todos los identificadores usan ``series_id: str`` (UTSAE agnóstico).
    - ``domain_name`` permite multi-dominio sin cambios de código.
    - Los métodos ``remember_*`` son fire-and-forget: un fallo NO debe
      propagarse al flujo transaccional.
    - Los métodos ``recall_*`` devuelven ``List[MemorySearchResult]``
      (vacía si no hay resultados o si el backend no está disponible).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..entities.anomaly import AnomalyResult
from ..entities.memory_search_result import MemorySearchResult
from ..entities.pattern import PatternResult
from ..entities.prediction import Prediction


class CognitiveMemoryPort(ABC):
    """Contrato para persistencia y consulta de memoria cognitiva.

    Operaciones agrupadas en dos categorías:
        - ``remember_*``: Escritura de artefactos cognitivos.
        - ``recall_*``: Búsqueda semántica sobre artefactos almacenados.
    """

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    def remember_explanation(
        self,
        prediction: Prediction,
        source_record_id: int,
        *,
        explanation_text: str = "",
        domain_name: str = "iot",
    ) -> Optional[str]:
        """Almacena una explicación de predicción como memoria cognitiva.

        Args:
            prediction: Entidad ``Prediction`` del dominio.
            source_record_id: ID del registro en ``dbo.predictions``
                (SQL Server).  Permite referencia cruzada.
            explanation_text: Texto de explicación a vectorizar.
                Si vacío, se usa ``prediction.metadata.get("explanation", "")``.
            domain_name: Namespace de dominio (``"iot"``, ``"finance"``, etc.).

        Returns:
            ID opaco del objeto en memoria cognitiva (e.g. UUID de Weaviate),
            o ``None`` si el almacenamiento falló o fue omitido.
        """
        ...

    @abstractmethod
    def remember_anomaly(
        self,
        anomaly: AnomalyResult,
        source_record_id: int,
        *,
        event_code: str = "ANOMALY_DETECTED",
        behavior_pattern: str = "",
        operational_context: str = "",
        domain_name: str = "iot",
    ) -> Optional[str]:
        """Almacena una traza de detección de anomalía.

        Args:
            anomaly: Entidad ``AnomalyResult`` del dominio.
            source_record_id: ID del registro en ``dbo.ml_events``.
            event_code: Código de evento (``"ANOMALY_DETECTED"``,
                ``"PREDICTION_DEVIATION"``, etc.).
            behavior_pattern: Patrón de comportamiento detectado
                (``"stable"``, ``"drifting"``, ``"spike"``, etc.).
            operational_context: Contexto operacional serializado
                (régimen, correlaciones).  Se vectoriza como campo
                secundario.
            domain_name: Namespace de dominio.

        Returns:
            ID opaco o ``None``.
        """
        ...

    @abstractmethod
    def remember_pattern(
        self,
        pattern: PatternResult,
        *,
        source_record_id: Optional[int] = None,
        domain_name: str = "iot",
    ) -> Optional[str]:
        """Almacena un patrón de comportamiento detectado.

        Args:
            pattern: Entidad ``PatternResult`` del dominio.
            source_record_id: ID en ``dbo.ml_events`` si fue persistido
                en SQL.  ``None`` si solo existe en memoria cognitiva.
            domain_name: Namespace de dominio.

        Returns:
            ID opaco o ``None``.
        """
        ...

    @abstractmethod
    def remember_decision(
        self,
        decision_data: Dict[str, object],
        source_record_id: int,
        *,
        domain_name: str = "iot",
    ) -> Optional[str]:
        """Almacena el razonamiento de una decisión del orquestador.

        Args:
            decision_data: Diccionario con los campos de la decisión.
                Debe contener al menos ``"summary"``, ``"explanation"``,
                ``"recommended_actions"``, ``"affected_series_ids"``.
                Se usa ``dict`` en lugar de la entidad ``Decision`` para
                evitar acoplar este port al módulo ``iot_worker``.
            source_record_id: ID en ``dbo.decision_actions``.
            domain_name: Namespace de dominio.

        Returns:
            ID opaco o ``None``.
        """
        ...

    # ------------------------------------------------------------------
    # Read operations (semantic search)
    # ------------------------------------------------------------------

    @abstractmethod
    def recall_similar_explanations(
        self,
        query: str,
        *,
        series_id: Optional[str] = None,
        engine_name: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        """Busca explicaciones de predicción semánticamente similares.

        Args:
            query: Texto de búsqueda en lenguaje natural.
            series_id: Filtrar por serie específica (``None`` = todas).
            engine_name: Filtrar por motor ML (``None`` = todos).
            limit: Máximo de resultados.
            min_certainty: Similitud mínima (0.0–1.0).

        Returns:
            Lista de ``MemorySearchResult`` ordenada por similitud
            descendente.  Vacía si no hay resultados o si el backend
            no está disponible.
        """
        ...

    @abstractmethod
    def recall_similar_anomalies(
        self,
        query: str,
        *,
        series_id: Optional[str] = None,
        severity: Optional[str] = None,
        event_code: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        """Busca anomalías con explicaciones semánticamente similares.

        Args:
            query: Texto de búsqueda.
            series_id: Filtrar por serie (``None`` = todas).
            severity: Filtrar por severidad (``None`` = todas).
            event_code: Filtrar por código de evento (``None`` = todos).
            limit: Máximo de resultados.
            min_certainty: Similitud mínima.

        Returns:
            Lista de ``MemorySearchResult``.
        """
        ...

    @abstractmethod
    def recall_similar_patterns(
        self,
        query: str,
        *,
        series_id: Optional[str] = None,
        pattern_type: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        """Busca patrones con descripciones semánticamente similares.

        Args:
            query: Texto de búsqueda.
            series_id: Filtrar por serie (``None`` = todas).
            pattern_type: Filtrar por tipo de patrón (``None`` = todos).
            limit: Máximo de resultados.
            min_certainty: Similitud mínima.

        Returns:
            Lista de ``MemorySearchResult``.
        """
        ...

    @abstractmethod
    def recall_similar_decisions(
        self,
        query: str,
        *,
        device_id: Optional[int] = None,
        severity: Optional[str] = None,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[MemorySearchResult]:
        """Busca decisiones con razonamiento semánticamente similar.

        Args:
            query: Texto de búsqueda.
            device_id: Filtrar por dispositivo (``None`` = todos).
            severity: Filtrar por severidad (``None`` = todas).
            limit: Máximo de resultados.
            min_certainty: Similitud mínima.

        Returns:
            Lista de ``MemorySearchResult``.
        """
        ...
