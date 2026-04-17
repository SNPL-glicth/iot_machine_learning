"""ExpertRegistry — catálogo de expertos MoE.

Implementa SRP: Solo gestiona registro y búsqueda de expertos.
NO ejecuta expertos, NO toma decisiones de routing.

Patrones:
- Registry: Catálogo centralizado de expertos
- SRP: Una sola responsabilidad (catalogación)
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Iterator, Any, Protocol
from dataclasses import dataclass
from collections import OrderedDict

from iot_machine_learning.domain.model.context_vector import ContextVector
from .expert_capability import ExpertCapability


class Expert(Protocol):
    """Protocolo mínimo para expertos registrables.
    
    ExpertRegistry no necesita saber cómo ejecutar expertos,
    solo necesita saber que existen y qué pueden hacer.
    """
    
    @property
    def name(self) -> str:
        """Nombre identificador único."""
        ...
    
    @property
    def capabilities(self) -> ExpertCapability:
        """Capacidades declaradas."""
        ...


@dataclass(frozen=True)
class ExpertEntry:
    """Entrada en el registro.
    
    Combina experto con sus metadatos de registro.
    """
    expert: Expert
    capabilities: ExpertCapability
    registration_order: int


class ExpertRegistry:
    """Catálogo thread-safe de expertos MoE.
    
    Responsabilidades (SRP):
    1. Registrar expertos con sus capacidades
    2. Buscar expertos por criterios (regime, domain, etc.)
    3. Filtrar candidatos para un contexto
    
    NO hace:
    - Ejecutar expertos
    - Decidir qué experto usar (eso es GatingNetwork)
    - Fusionar resultados (eso es FusionLayer)
    
    Example:
        >>> registry = ExpertRegistry()
        >>> registry.register("taylor", taylor_expert, taylor_caps)
        >>> 
        >>> # Buscar por régimen
        >>> volatile_experts = registry.find_by_regime("volatile")
        >>> 
        >>> # Filtrar por contexto completo
        >>> candidates = registry.get_candidates(ContextVector(
        ...     regime="volatile", domain="iot", n_points=10, signal_features={}
        ... ))
    """
    
    def __init__(self, max_entries: int = 100):
        """Inicializa registro vacío.
        
        Args:
            max_entries: Límite de expertos para prevenir crecimiento ilimitado.
        """
        self._max_entries = max_entries
        self._entries: Dict[str, ExpertEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._registration_counter = 0
    
    def register(
        self,
        expert_id: str,
        expert: Expert,
        capabilities: Optional[ExpertCapability] = None,
        force: bool = False
    ) -> None:
        """Registra un experto en el catálogo.
        
        Args:
            expert_id: Identificador único del experto.
            expert: Instancia que implementa Expert.
            capabilities: Capacidades (si None, usa expert.capabilities).
            force: Si True, sobrescribe existente.
            
        Raises:
            ValueError: Si existe y force=False.
            RuntimeError: Si registro lleno.
        """
        with self._lock:
            if expert_id in self._entries and not force:
                raise ValueError(f"Expert '{expert_id}' already registered. Use force=True to overwrite.")
            
            if len(self._entries) >= self._max_entries and expert_id not in self._entries:
                raise RuntimeError(f"Registry full ({self._max_entries} experts).")
            
            caps = capabilities or expert.capabilities
            entry = ExpertEntry(
                expert=expert,
                capabilities=caps,
                registration_order=self._registration_counter
            )
            
            self._entries[expert_id] = entry
            self._registration_counter += 1
            self._entries.move_to_end(expert_id)
    
    def unregister(self, expert_id: str) -> bool:
        """Elimina un experto del registro.
        
        Returns:
            True si se eliminó, False si no existía.
        """
        with self._lock:
            if expert_id in self._entries:
                del self._entries[expert_id]
                return True
            return False
    
    def get(self, expert_id: str) -> Optional[Expert]:
        """Obtiene experto por su ID."""
        with self._lock:
            entry = self._entries.get(expert_id)
            return entry.expert if entry else None
    
    def get_capabilities(self, expert_id: str) -> Optional[ExpertCapability]:
        """Obtiene capacidades de un experto registrado."""
        with self._lock:
            entry = self._entries.get(expert_id)
            return entry.capabilities if entry else None
    
    def list_all(self) -> List[str]:
        """Lista todos los expert IDs en orden de registro."""
        with self._lock:
            return list(self._entries.keys())
    
    def find_by_regime(self, regime: str) -> List[str]:
        """Busca expertos que soportan un régimen.
        
        Args:
            regime: Régimen a buscar.
            
        Returns:
            Lista de expert_ids que soportan el régimen.
        """
        with self._lock:
            return [
                eid for eid, entry in self._entries.items()
                if entry.capabilities.supports_regime(regime)
            ]
    
    def find_by_domain(self, domain: str) -> List[str]:
        """Busca expertos para un dominio."""
        with self._lock:
            return [
                eid for eid, entry in self._entries.items()
                if entry.capabilities.supports_domain(domain)
            ]
    
    def get_candidates(self, context: ContextVector) -> List[str]:
        """Filtra expertos elegibles para un contexto.
        
        Matching criteria:
        - Regime soportado
        - Domain soportado
        - n_points dentro de rango [min_points, max_points]
        
        Args:
            context: Contexto de predicción.
            
        Returns:
            Lista de expert IDs elegibles, ordenados por costo ascendente.
        """
        with self._lock:
            candidates = []
            for eid, entry in self._entries.items():
                caps = entry.capabilities
                
                if not caps.matches_context(
                    regime=context.regime,
                    domain=context.domain,
                    n_points=context.n_points
                ):
                    continue
                
                candidates.append((eid, caps.computational_cost))
            
            # Ordenar por costo (más eficientes primero)
            candidates.sort(key=lambda x: x[1])
            return [eid for eid, _ in candidates]
    
    def __contains__(self, expert_id: str) -> bool:
        """Operador 'in'."""
        with self._lock:
            return expert_id in self._entries
    
    def __len__(self) -> int:
        """Número de expertos registrados."""
        with self._lock:
            return len(self._entries)
    
    def __iter__(self) -> Iterator[str]:
        """Iterador sobre expert IDs."""
        with self._lock:
            return iter(list(self._entries.keys()))
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del registro."""
        with self._lock:
            if not self._entries:
                return {
                    "total_experts": 0,
                    "regimes_covered": [],
                    "domains_covered": [],
                    "avg_cost": 0.0,
                }
            
            regimes = set()
            domains = set()
            total_cost = 0.0
            
            for entry in self._entries.values():
                regimes.update(entry.capabilities.regimes)
                domains.update(entry.capabilities.domains)
                total_cost += entry.capabilities.computational_cost
            
            return {
                "total_experts": len(self._entries),
                "regimes_covered": sorted(regimes),
                "domains_covered": sorted(domains),
                "avg_cost": total_cost / len(self._entries),
                "max_entries": self._max_entries,
            }
