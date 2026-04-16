"""ExpertRegistry — catálogo de expertos para arquitectura MoE.

Implementa el patrón Registry para gestionar expertos disponibles.
Permite búsqueda por capacidades y contexto.

Siguiendo principios SOLID:
- SRP: Solo gestiona registro y búsqueda de expertos
- OCP: Abierto a nuevos expertos, cerrado a modificación
- DIP: Depende de ExpertPort (abstracción), no implementaciones

ISO 42001: Registra capacidades declaradas para trazabilidad.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Iterator, Set
from dataclasses import dataclass
from collections import OrderedDict

from iot_machine_learning.domain.ports.expert_port import ExpertPort, ExpertCapability


@dataclass(frozen=True)
class ExpertEntry:
    """Entrada en el registro con experto y metadatos.
    
    Attributes:
        expert: Instancia del experto (implementa ExpertPort).
        capabilities: Capacidades declaradas del experto.
        registration_order: Orden de registro (para desempate).
    """
    expert: ExpertPort
    capabilities: ExpertCapability
    registration_order: int


class ExpertRegistry:
    """Catálogo thread-safe de expertos disponibles.
    
    Responsabilidades:
    1. Registrar expertos con sus capacidades
    2. Buscar expertos por criterios (regime, domain, etc.)
    3. Encontrar candidatos para un contexto específico
    4. Proveer información de diagnóstico
    
    Thread-safe: Todas las operaciones están protegidas por lock.
    
    Example:
        >>> registry = ExpertRegistry()
        >>> registry.register("taylor", taylor_expert, taylor_caps)
        >>> 
        >>> # Buscar expertos para régimen volatile
        >>> candidates = registry.find_by_regime("volatile")
        >>> print(candidates)  # ["taylor", ...]
        >>> 
        >>> # Obtener experto por nombre
        >>> expert = registry.get("taylor")
    """
    
    def __init__(self, max_entries: int = 100):
        """Inicializa registro vacío.
        
        Args:
            max_entries: Límite de expertos (evita crecimiento ilimitado).
        """
        self._max_entries = max_entries
        self._entries: Dict[str, ExpertEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._registration_counter = 0
    
    def register(
        self,
        expert_id: str,
        expert: ExpertPort,
        capabilities: Optional[ExpertCapability] = None,
        force: bool = False
    ) -> None:
        """Registra un experto en el catálogo.
        
        Args:
            expert_id: Identificador único del experto.
            expert: Instancia que implementa ExpertPort.
            capabilities: Capacidades declaradas (si None, usa expert.capabilities).
            force: Si True, sobrescribe existente. Si False, raise si existe.
            
        Raises:
            ValueError: Si expert_id existe y force=False.
            RuntimeError: Si registro está lleno.
            
        Note:
            Thread-safe. Bloquea durante la operación.
        """
        with self._lock:
            if expert_id in self._entries and not force:
                raise ValueError(
                    f"Experto '{expert_id}' ya registrado. "
                    f"Use force=True para sobrescribir."
                )
            
            if len(self._entries) >= self._max_entries and expert_id not in self._entries:
                raise RuntimeError(
                    f"Registro lleno ({self._max_entries} expertos). "
                    f"Aumente max_entries o elimine expertos."
                )
            
            caps = capabilities or expert.capabilities
            entry = ExpertEntry(
                expert=expert,
                capabilities=caps,
                registration_order=self._registration_counter
            )
            
            self._entries[expert_id] = entry
            self._registration_counter += 1
            
            # Mantener orden LRU: mover al final si existe
            self._entries.move_to_end(expert_id)
    
    def unregister(self, expert_id: str) -> bool:
        """Elimina un experto del registro.
        
        Args:
            expert_id: Identificador del experto a eliminar.
            
        Returns:
            True si se eliminó, False si no existía.
        """
        with self._lock:
            if expert_id in self._entries:
                del self._entries[expert_id]
                return True
            return False
    
    def get(self, expert_id: str) -> Optional[ExpertPort]:
        """Obtiene experto por su identificador.
        
        Args:
            expert_id: Identificador del experto.
            
        Returns:
            Instancia del experto o None si no existe.
        """
        with self._lock:
            entry = self._entries.get(expert_id)
            return entry.expert if entry else None
    
    def get_capabilities(self, expert_id: str) -> Optional[ExpertCapability]:
        """Obtiene capacidades de un experto registrado.
        
        Args:
            expert_id: Identificador del experto.
            
        Returns:
            ExpertCapability o None si no existe.
        """
        with self._lock:
            entry = self._entries.get(expert_id)
            return entry.capabilities if entry else None
    
    def get_entry(self, expert_id: str) -> Optional[ExpertEntry]:
        """Obtiene entrada completa (experto + metadatos).
        
        Args:
            expert_id: Identificador del experto.
            
        Returns:
            ExpertEntry o None si no existe.
        """
        with self._lock:
            return self._entries.get(expert_id)
    
    def list_all(self) -> List[str]:
        """Lista todos los expertos registrados.
        
        Returns:
            Lista de expert_id en orden de registro.
        """
        with self._lock:
            return list(self._entries.keys())
    
    def find_by_regime(self, regime: str) -> List[str]:
        """Busca expertos que manejan un régimen específico.
        
        Args:
            regime: Régimen a buscar (ej: "stable", "volatile").
            
        Returns:
            Lista de expert_id que declaran manejar ese régimen.
        """
        with self._lock:
            return [
                eid for eid, entry in self._entries.items()
                if regime in entry.capabilities.regimes
            ]
    
    def find_by_domain(self, domain: str) -> List[str]:
        """Busca expertos para un dominio específico.
        
        Args:
            domain: Dominio a buscar (ej: "iot", "finance").
            
        Returns:
            Lista de expert_id disponibles para ese dominio.
        """
        with self._lock:
            return [
                eid for eid, entry in self._entries.items()
                if domain in entry.capabilities.domains
            ]
    
    def find_by_specialty(self, specialty: str) -> List[str]:
        """Busca expertos con una especialidad específica.
        
        Args:
            specialty: Especialidad (ej: "seasonality", "anomalies").
            
        Returns:
            Lista de expert_id con esa especialidad.
        """
        with self._lock:
            return [
                eid for eid, entry in self._entries.items()
                if specialty in entry.capabilities.specialties
            ]
    
    def get_candidates(
        self,
        regime: str,
        domain: str,
        n_points: int,
        max_cost: Optional[float] = None
    ) -> List[str]:
        """Obtiene candidatos elegibles para un contexto.
        
        Filtra por:
        - Régimen soportado
        - Dominio soportado
        - Rango de puntos [min_points, max_points]
        - Costo computacional opcional
        
        Args:
            regime: Régimen actual del contexto.
            domain: Dominio del contexto.
            n_points: Número de puntos disponibles.
            max_cost: Costo máximo permitido (opcional).
            
        Returns:
            Lista de expert_id elegibles, ordenados por costo ascendente.
        """
        with self._lock:
            candidates = []
            for eid, entry in self._entries.items():
                caps = entry.capabilities
                
                # Filtro de régimen
                if regime not in caps.regimes:
                    continue
                
                # Filtro de dominio
                if domain not in caps.domains:
                    continue
                
                # Filtro de puntos
                if n_points < caps.min_points:
                    continue
                if caps.max_points > 0 and n_points > caps.max_points:
                    continue
                
                # Filtro de costo
                if max_cost is not None and caps.computational_cost > max_cost:
                    continue
                
                candidates.append((eid, caps.computational_cost))
            
            # Ordenar por costo ascendente
            candidates.sort(key=lambda x: x[1])
            return [eid for eid, _ in candidates]
    
    def __contains__(self, expert_id: str) -> bool:
        """Operador 'in' para verificar existencia."""
        with self._lock:
            return expert_id in self._entries
    
    def __len__(self) -> int:
        """Número de expertos registrados."""
        with self._lock:
            return len(self._entries)
    
    def __iter__(self) -> Iterator[str]:
        """Iterador sobre expert_ids."""
        with self._lock:
            # Retornar copia para evitar modificación durante iteración
            return iter(list(self._entries.keys()))
    
    def get_stats(self) -> Dict[str, any]:
        """Estadísticas del registro para monitoring.
        
        Returns:
            Dict con métricas del registro.
        """
        with self._lock:
            if not self._entries:
                return {
                    "total_experts": 0,
                    "regimes_covered": [],
                    "domains_covered": [],
                    "avg_computational_cost": 0.0,
                }
            
            regimes: Set[str] = set()
            domains: Set[str] = set()
            total_cost = 0.0
            
            for entry in self._entries.values():
                regimes.update(entry.capabilities.regimes)
                domains.update(entry.capabilities.domains)
                total_cost += entry.capabilities.computational_cost
            
            return {
                "total_experts": len(self._entries),
                "regimes_covered": sorted(regimes),
                "domains_covered": sorted(domains),
                "avg_computational_cost": total_cost / len(self._entries),
                "max_entries": self._max_entries,
            }
