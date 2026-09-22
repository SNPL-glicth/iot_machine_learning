"""Value Object para resultados de búsqueda en memoria cognitiva.

Representa un resultado de búsqueda semántica devuelto por
``CognitiveMemoryPort``.  Es agnóstico a la implementación de
almacenamiento (Weaviate, Elasticsearch, Pinecone, etc.).

No contiene tipos ni imports de infraestructura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class MemorySearchResult:
    """Resultado de búsqueda en memoria cognitiva.

    Devuelto por los métodos ``recall_*`` de ``CognitiveMemoryPort``.
    El dominio consume este objeto sin conocer qué backend lo produjo.

    Attributes:
        memory_id: Identificador opaco del registro en memoria cognitiva.
            Internamente puede ser un UUID de Weaviate, un doc_id de
            Elasticsearch o una clave de Redis.  El dominio lo trata como
            cadena opaca.
        series_id: Identificador de la serie temporal a la que pertenece
            el registro original.
        text: Texto asociado al registro (ej. descripción del evento).
        certainty: Puntuación de similitud semántica en el rango [0.0, 1.0].
            1.0 = coincidencia idéntica; 0.0 = sin relación.
        source_record_id: ID del registro transaccional original (ej. ID de
            anomalía o predicción en Postgres), si aplica.
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
    metadata: Mapping[str, object] = field(default_factory=dict)

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
