"""RegimeBasedGating — estrategia heurística de routing por régimen.

Implementación simple y robusta basada en reglas de negocio.
No requiere entrenamiento, ideal para Fase 1 de adopción MoE.

Siguiendo principio YAGNI: comenzar simple, complejizar solo si necesario.
ISO 42001: Totalmente interpretable (reglas explícitas).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from iot_machine_learning.infrastructure.ml.moe.gating.base import GatingNetwork, GatingProbs, ContextVector


@dataclass(frozen=True)
class RegimeRoutingRule:
    """Regla de routing para un régimen específico.
    
    Define distribución fija de probabilidades por régimen.
    Las probabilidades deben sumar 1.0 (normalizadas automáticamente).
    
    Attributes:
        regime: Régimen al que aplica (ej: "stable").
        expert_weights: Dict {expert_id: peso relativo}.
        rationale: Explicación textual de la regla.
    """
    regime: str
    expert_weights: Dict[str, float]
    rationale: str
    
    def __post_init__(self):
        # Normalizar pesos a probabilidades
        total = sum(self.expert_weights.values())
        if total <= 0:
            raise ValueError(f"Pesos deben ser positivos, sumaron {total}")
    
    def to_probabilities(self, all_experts: List[str]) -> Dict[str, float]:
        """Convierte pesos a probabilidades normalizadas.
        
        Args:
            all_experts: Lista completa de expertos disponibles.
            
        Returns:
            Dict con probabilidad para cada experto (0.0 si no en regla).
        """
        total = sum(self.expert_weights.values())
        
        probs = {}
        for expert_id in all_experts:
            weight = self.expert_weights.get(expert_id, 0.0)
            probs[expert_id] = weight / total if total > 0 else 0.0
        
        return probs


class RegimeBasedGating(GatingNetwork):
    """Gating basado en reglas por régimen (heurístico).
    
    Implementación de Fase 1: simple, interpretable, sin entrenamiento.
    Cada régimen tiene una distribución fija de probabilidades.
    
    Reglas por defecto:
    - stable: Baseline dominante (bajo costo, suficiente)
    - trending: Statistical dominante (maneja tendencias)
    - volatile: Taylor dominante (maneja no-linealidades)
    - noisy: Ensemble dominante (robustez al ruido)
    
    Example:
        >>> gating = RegimeBasedGating.with_default_rules(["baseline", "statistical", "taylor"])
        >>> context = ContextVector(regime="volatile", domain="iot", n_points=10, signal_features={})
        >>> probs = gating.route(context)
        >>> print(probs.top_expert)  # "taylor"
        >>> print(gating.explain(context, probs))  # "Régimen volatile → preferir Taylor"
    """
    
    # Reglas deterministas según especificación ZENIN MoE
    # Suma de pesos = 1.0 para cada régimen
    DEFAULT_RULES = {
        "stable": RegimeRoutingRule(
            regime="stable",
            expert_weights={
                "baseline": 0.85,
                "statistical": 0.10,
                "taylor": 0.05,
            },
            rationale="Régimen estable: Baseline dominante (85%), mínimo costo computacional"
        ),
        "trending": RegimeRoutingRule(
            regime="trending",
            expert_weights={
                "statistical": 0.60,
                "taylor": 0.30,
                "baseline": 0.10,
            },
            rationale="Régimen con tendencia: Statistical lidera (60%), Taylor apoya (30%)"
        ),
        "volatile": RegimeRoutingRule(
            regime="volatile",
            expert_weights={
                "taylor": 0.70,
                "statistical": 0.20,
                "baseline": 0.10,
            },
            rationale="Régimen volátil: Taylor dominante (70%), maneja no-linealidades"
        ),
    }
    
    def __init__(
        self,
        expert_ids: Optional[List[str]] = None,
        rules: Optional[Dict[str, RegimeRoutingRule]] = None,
        default_rule: Optional[RegimeRoutingRule] = None
    ):
        """Inicializa gating con reglas por régimen.
        
        Args:
            expert_ids: Lista de expertos disponibles.
            rules: Dict de reglas por régimen (si None, usa DEFAULT_RULES).
            default_rule: Regla fallback para regimes desconocidos.
        """
        super().__init__(expert_ids)
        self._rules = rules or self.DEFAULT_RULES.copy()
        self._default_rule = default_rule or RegimeRoutingRule(
            regime="unknown",
            expert_weights={"baseline": 0.5, "statistical": 0.5},
            rationale="Régimen desconocido: fallback conservador"
        )
    
    def route(self, context: ContextVector) -> GatingProbs:
        """Rutea según régimen del contexto.
        
        Args:
            context: Vector de contexto con regime definido.
            
        Returns:
            GatingProbs según regla del régimen.
            
        Raises:
            ValueError: Si contexto no tiene regime o expert_ids no definidos.
        """
        if not context.regime:
            raise ValueError("ContextVector debe tener regime definido")
        
        if not self._expert_ids:
            raise ValueError("Expert IDs no configurados. Set expert_ids primero.")
        
        self._routing_count += 1
        
        # Seleccionar regla apropiada
        rule = self._rules.get(context.regime, self._default_rule)
        
        # Calcular probabilidades
        probs = rule.to_probabilities(self._expert_ids)
        
        # Calcular entropía
        import math
        entropy = -sum(
            p * math.log(p + 1e-10) for p in probs.values() if p > 0
        )
        
        # Top experto
        top_expert = max(probs.items(), key=lambda x: x[1])[0]
        
        return GatingProbs(
            probabilities=probs,
            entropy=entropy,
            top_expert=top_expert,
            metadata={
                "rule_used": rule.regime,
                "rationale": rule.rationale,
                "gating_type": "regime_based",
            }
        )
    
    def explain(self, context: ContextVector, probs: GatingProbs) -> str:
        """Explica decisión de routing.
        
        Args:
            context: Contexto usado.
            probs: Probabilidades resultantes.
            
        Returns:
            Explicación textual detallada.
        """
        rule_name = probs.metadata.get("rule_used", "unknown")
        rationale = probs.metadata.get("rationale", "Sin explicación disponible")
        
        # Construir explicación
        lines = [
            f"[RegimeBasedGating] Régimen detectado: '{context.regime}'",
            f"Regla aplicada: {rule_name}",
            f"Razonamiento: {rationale}",
            "Distribución de probabilidades:",
        ]
        
        for expert_id, prob in sorted(probs.probabilities.items(), key=lambda x: -x[1]):
            if prob > 0.01:  # Solo mostrar relevantes
                lines.append(f"  - {expert_id}: {prob:.1%}")
        
        lines.append(f"Entropía (incertidumbre): {probs.entropy:.3f}")
        lines.append(f"Experto seleccionado: {probs.top_expert}")
        
        return "\n".join(lines)
    
    def add_rule(self, rule: RegimeRoutingRule) -> None:
        """Añade o sobrescribe regla para un régimen.
        
        Args:
            rule: Nueva regla de routing.
        """
        self._rules[rule.regime] = rule
    
    def remove_rule(self, regime: str) -> bool:
        """Elimina regla para un régimen.
        
        Args:
            regime: Régimen a eliminar.
            
        Returns:
            True si existía y se eliminó.
        """
        if regime in self._rules:
            del self._rules[regime]
            return True
        return False
    
    def list_rules(self) -> List[str]:
        """Lista regímenes con reglas definidas."""
        return list(self._rules.keys())
    
    def get_rule(self, regime: str) -> Optional[RegimeRoutingRule]:
        """Obtiene regla para un régimen."""
        return self._rules.get(regime)
    
    @classmethod
    def with_default_rules(cls, expert_ids: List[str]) -> "RegimeBasedGating":
        """Factory method para crear con reglas por defecto.
        
        Args:
            expert_ids: Expertos disponibles.
            
        Returns:
            Instancia configurada con DEFAULT_RULES.
        """
        return cls(expert_ids=expert_ids, rules=cls.DEFAULT_RULES.copy())
    
    @classmethod
    def conservative(cls, expert_ids: List[str]) -> "RegimeBasedGating":
        """Factory method para configuración conservadora (baseline-first).
        
        Prioriza baseline para reducir costo computacional.
        
        Args:
            expert_ids: Expertos disponibles.
            
        Returns:
            Instancia con reglas conservadoras.
        """
        rules = {
            "stable": RegimeRoutingRule(
                regime="stable",
                expert_weights={"baseline": 1.0},
                rationale="Máxima eficiencia: solo baseline"
            ),
            "trending": RegimeRoutingRule(
                regime="trending",
                expert_weights={"statistical": 0.7, "baseline": 0.3},
                rationale="Balance precisión/costo"
            ),
            "volatile": RegimeRoutingRule(
                regime="volatile",
                expert_weights={"taylor": 0.6, "statistical": 0.4},
                rationale="Necesario para dinámica compleja"
            ),
            "noisy": RegimeRoutingRule(
                regime="noisy",
                expert_weights={"ensemble": 0.7, "statistical": 0.3},
                rationale="Robustez ante ruido"
            ),
        }
        return cls(expert_ids=expert_ids, rules=rules)
