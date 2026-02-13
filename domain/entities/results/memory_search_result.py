"""Value Object para resultados de búsqueda en memoria cognitiva.

Representa un resultado de búsqueda semántica devuelto por
``CognitiveMemoryPort``.  Es agnóstico a la implementación de
almacenamiento (Weaviate, Elasticsearch, Pinecone, etc.).

No contiene tipos ni imports de infraestructura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class MemorySearchResult:
    """Resultado de búsqueda en memoria cognitiva.

    Devuelto por los métodos ``recall_*`` de ``CognitiveMemoryPort``.
    El dominio consume este objeto sin conocer qué backend lo produjo.

    Attributes:
        memory_id: Identificador opaco del registro en memoria cognitiva.
            Internamente puede ser un UUID de Weaviate, un doc_id de
            Elasticsearch, etc.  El dominio no interpreta este valor.
        series_id: Identificador UTSAE de la serie asociada.
        text: Texto que coincidió con la búsqueda semántica.
        certainty: Similitud semántica (0.0–1.0).  1.0 = coincidencia
            perfecta.  El umbral mínimo lo decide el consumidor.
        source_record_id: Referencia cruzada al registro en el sistema
            transaccional (e.g. ``predictions.id``, ``ml_events.id``,
            ``decision_actions.id`` en SQL Server).  ``None`` si el
            registro solo existe en memoria cognitiva.
        created_at: Timestamp ISO 8601 de creación del registro original.
        metadata: Propiedades adicionales del registro.  Estructura
            variable según la clase de memoria consultada.
    """

    memory_id: str
    series_id: str
    text: str
    certainty: float
    source_record_id: Optional[int] = None
    created_at: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def has_source_record(self) -> bool:
        """``True`` si existe referencia cruzada al sistema transaccional."""
        return self.source_record_id is not None

    @property
    def is_high_certainty(self) -> bool:
        """``True`` si la similitud semántica es >= 0.85."""
        return self.certainty >= 0.85

    def to_dict(self) -> Dict[str, object]:
        """Serializa para API responses o logging."""
        return {
            "memory_id": self.memory_id,
            "series_id": self.series_id,
            "text": self.text,
            "certainty": self.certainty,
            "source_record_id": self.source_record_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
