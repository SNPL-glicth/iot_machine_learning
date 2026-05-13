"""BayesianWeightConfig — configuration for Bayesian weight tracking.

Centralizes all magic numbers and tunable parameters for the Bayesian
weight tracker. Uses frozen dataclass for immutability and validation.

Applies SRP: Configuration is separate from tracking logic.
Applies DIP: Tracker depends on config abstraction, not literals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BayesianWeightConfig:
    """Configuration for Bayesian weight tracking via Gaussian priors.
    
    All tunable parameters for the Bayesian weight tracking system.
    Uses Bayesian inference with Gaussian priors, NOT reinforcement learning.
    
    ALPHA SELECTION METHODOLOGY (FASE-23):
    - alpha=0.15 (base): Punto medio entre respuesta (0.3) y estabilidad (0.05).
      Equivale a memoria efectiva de ~6 observaciones (1/alpha ≈ 6.67).
      Referencias: Hyndman & Athanasopoulos 'Forecasting' cap. 7.
    - Regime alphas en ML_PLASTICITY_REGIME_ALPHAS (cognitive_config.py):
      * STABLE=0.08 (memoria ~12 obs, baja varianza)
      * TRENDING=0.18 (memoria ~6 obs, adaptación moderada)
      * VOLATILE=0.30 (memoria ~3 obs, respuesta rápida)
      * NOISY=0.05 (memoria ~20 obs, filtro agresivo)
    
    Attributes:
        alpha: Learning rate for weight updates.
        min_weight: Minimum weight floor per engine.
        immediate_persist_threshold: Threshold for immediate persistence.
        drift_decay_factor: Decay factor for prior under drift.
        drift_variance_expansion: Variance expansion multiplier under drift.
        sigma2_obs_default: Default observational variance.
        sigma2_obs_min: Minimum observational variance floor.
        prior_variance_scale: Scale of prior relative to empirical variance.
        variance_window: Window size for empirical variance estimation.
        variance_min_samples: Minimum samples for variance estimation.
        convergence_window: Window for convergence evaluation.
        convergence_cv_threshold: CV threshold for convergence.
        weight_history_maxlen: Maximum history size per engine.
        max_regimes: Maximum regimes tracked per series.
        regime_ttl_seconds: TTL for regime entries (seconds).
        regularization_strength: L2 regularization strength.
    """
    
    # ── Learning ──────────────────────────────────────────
    alpha: float = 0.15
    # Tasa de aprendizaje del tracker. Rango válido: (0.0, 1.0).
    # Valores altos = adaptación rápida, valores bajos = estabilidad.
    
    min_weight: float = 0.05
    # Peso mínimo garantizado por engine en tracking histórico.
    # Rango: (0.0, 1.0). Debe ser < 1/n_engines para no sesgar.
    # FASE-21: min_weight=0.05 para Bayesian (confianza acumulada).
    # Inhibition usa min_weight=0.02 (post-supresión, más bajo).
    # Diferencia intencional: Bayesian floor más alto porque representa
    # confianza histórica; Inhibition floor más bajo porque supresión
    # puede ser severa en casos extremos.
    # Ver: infrastructure/ml/cognitive/inhibition/gate.py
    #
    # WEIGHT COLLAPSE RISK (FASE-26 - Phase 9 audit):
    # NO existe max_weight ceiling. RISK: Weight puede colapsar a 1.0
    # en un solo engine (ensemble collapse). PENDING: Agregar max_weight=0.7
    # cuando se calibre con datos históricos de ensemble diversity.
    # Ver: Phase 9 audit Section 4.1 "Ensemble Collapse Risk".
    
    # ── Persistencia ──────────────────────────────────────
    immediate_persist_threshold: float = 0.15
    # Persiste inmediatamente si el cambio de peso supera este valor.
    # Rango: (0.0, 1.0). Mismo valor que alpha por diseño — si cambia
    # alpha, evaluar si este debe cambiar también.
    
    # ── Drift handling ────────────────────────────────────
    drift_decay_factor: float = 0.5
    # Factor de decaimiento del prior ante drift detectado.
    # Rango: (0.0, 1.0). 0.5 = reduce varianza del prior a la mitad.
    
    drift_variance_expansion: float = 2.0
    # Multiplica la varianza del prior cuando hay drift.
    # Rango: > 1.0. Valores altos = más incertidumbre ante drift.
    
    # ── Varianza observacional ────────────────────────────
    sigma2_obs_default: float = 1.0
    # Varianza observacional default cuando no hay historia suficiente.
    # Rango: > 0.0. Calibrar según escala de los errores del dominio.
    
    sigma2_obs_min: float = 0.01
    # Piso de varianza observacional. Evita división por cero.
    # Rango: > 0.0. Debe ser << sigma2_obs_default.
    
    prior_variance_scale: float = 1.0
    # Escala del prior Gaussiano relativa a la varianza de los datos.
    # Rango: > 0.0. 1.0 = prior igual a varianza empírica.
    
    # ── Ventana de varianza ───────────────────────────────
    variance_window: int = 20
    # Tamaño de ventana para estimar varianza empírica de errores.
    # Rango: >= 5. Debe ser >= variance_min_samples.
    
    variance_min_samples: int = 5
    # Mínimo de muestras para activar estimación de varianza.
    # Rango: [2, variance_window]. Debe ser <= variance_window.
    
    # ── Convergencia ──────────────────────────────────────
    convergence_window: int = 10
    # Últimos N pesos para evaluar convergencia.
    # Rango: >= 3. Ventana pequeña = convergencia más rápida.
    
    convergence_cv_threshold: float = 0.05
    # Coeficiente de variación (std/mean) para declarar convergencia.
    # Rango: (0.0, 1.0). 0.05 = 5% de variación aceptable.
    # NOTA: mismo valor que min_weight por coincidencia numérica,
    # son conceptos distintos — no consolidar.
    
    weight_history_maxlen: int = 50
    # Tamaño máximo del deque de historial de pesos por engine.
    # Rango: >= convergence_window.
    
    # ── Regímenes ─────────────────────────────────────────
    max_regimes: int = 10
    # Máximo de regímenes simultáneos tracked por series.
    # Rango: >= 1. Más regímenes = más memoria.
    
    regime_ttl_seconds: float = 86400.0
    # TTL de entrada de régimen en segundos. Default: 1 día.
    # Rango: > 0.0.
    
    # ── Regularización ────────────────────────────────────
    regularization_strength: float = 0.01
    # L2 regularization sobre los pesos. Evita overfitting a engines.
    # Rango: [0.0, 1.0). 0.0 = sin regularización.
    # NOTA: mismo valor que sigma2_obs_min por coincidencia numérica,
    # son conceptos distintos — no consolidar.
    
    def validate(self) -> None:
        """Validate all fields are in valid ranges.
        
        Raises:
            ValueError: with field name and received value if any check fails.
        """
        # alpha en (0.0, 1.0) — excluye extremos
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0), got {self.alpha}")
        
        # min_weight en (0.0, 1.0)
        if not (0.0 < self.min_weight < 1.0):
            raise ValueError(f"min_weight must be in (0.0, 1.0), got {self.min_weight}")
        
        # sigma2_obs_min debe ser menor que sigma2_obs_default
        if self.sigma2_obs_min >= self.sigma2_obs_default:
            raise ValueError(
                f"sigma2_obs_min ({self.sigma2_obs_min}) must be "
                f"< sigma2_obs_default ({self.sigma2_obs_default})"
            )
        
        # variance_min_samples <= variance_window
        if self.variance_min_samples > self.variance_window:
            raise ValueError(
                f"variance_min_samples ({self.variance_min_samples}) must be "
                f"<= variance_window ({self.variance_window})"
            )
        
        # convergence_window <= weight_history_maxlen
        if self.convergence_window > self.weight_history_maxlen:
            raise ValueError(
                f"convergence_window ({self.convergence_window}) must be "
                f"<= weight_history_maxlen ({self.weight_history_maxlen})"
            )
        
        # drift_variance_expansion > 1.0
        if self.drift_variance_expansion <= 1.0:
            raise ValueError(
                f"drift_variance_expansion must be > 1.0, "
                f"got {self.drift_variance_expansion}"
            )
        
        # Todos los floats > 0.0
        positive_floats = {
            "drift_decay_factor": self.drift_decay_factor,
            "prior_variance_scale": self.prior_variance_scale,
            "sigma2_obs_default": self.sigma2_obs_default,
            "sigma2_obs_min": self.sigma2_obs_min,
            "regularization_strength": self.regularization_strength,
            "regime_ttl_seconds": self.regime_ttl_seconds,
            "convergence_cv_threshold": self.convergence_cv_threshold,
            "immediate_persist_threshold": self.immediate_persist_threshold,
        }
        for name, value in positive_floats.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0.0, got {value}")
        
        # Todos los ints >= 1
        positive_ints = {
            "variance_window": self.variance_window,
            "variance_min_samples": self.variance_min_samples,
            "convergence_window": self.convergence_window,
            "weight_history_maxlen": self.weight_history_maxlen,
            "max_regimes": self.max_regimes,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")


# Deprecation notice para WeightTrackerConfig
# WeightTrackerConfig (constants.py) sigue disponible para
# compatibilidad con código externo pero no debe usarse en código nuevo.
# Usar BayesianWeightConfig como source of truth.
