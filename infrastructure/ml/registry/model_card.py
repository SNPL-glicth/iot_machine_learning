"""ModelCard — ISO 22989 compliant model governance artifact.

Provides standardized metadata for ML models in the ZENIN pipeline,
enabling traceability, auditability, and regulatory compliance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
import json


@dataclass(frozen=True)
class ModelMetrics:
    """Performance metrics for the model."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    ece: Optional[float] = None  # Expected Calibration Error
    brier_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    custom: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelInput:
    """Input schema specification."""
    name: str
    type: str  # "tensor", "scalar", "vector", "timeseries"
    shape: Optional[List[int]] = None
    dtype: str = "float32"
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass(frozen=True)
class ModelOutput:
    """Output schema specification."""
    name: str
    type: str  # "tensor", "scalar", "vector", "distribution"
    shape: Optional[List[int]] = None
    dtype: str = "float32"
    description: str = ""
    range_min: Optional[float] = None
    range_max: Optional[float] = None


@dataclass(frozen=True)
class ModelThresholds:
    """Decision and operational thresholds."""
    gamma_exec: float = 0.5          # EXECUTE threshold
    geometric_threshold: float = 0.3 # EMERGENCY_FLUSH threshold
    min_history_points: int = 50     # Minimum data for valid inference
    max_latency_ms: float = 5.0      # SLA for inference
    confidence_floor: float = 0.05   # Minimum confidence
    confidence_ceiling: float = 0.95 # Maximum confidence


@dataclass
class ModelCard:
    """
    ISO 22989 Model Card for ZENIN ML Pipeline.
    
    This class captures all metadata required for model governance:
    - Identity and versioning
    - Intended use and limitations
    - Training and evaluation data
    - Performance metrics
    - Input/output schemas
    - Operational thresholds
    - Ethical and safety considerations
    - Audit trail
    """
    
    # --- Identity ---
    model_id: str
    version: str
    name: str
    description: str = ""
    
    # --- Lifecycle ---
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    status: str = "development"  # development, staging, production, deprecated
    
    # --- Ownership ---
    owner: str = "ZENIN ML Team"
    maintainers: List[str] = field(default_factory=list)
    
    # --- Intended Use ---
    intended_use: str = ""
    domain: str = "iot_timeseries"  # iot_timeseries, finance, healthcare, etc.
    task_type: str = "forecasting"  # forecasting, classification, anomaly_detection
    target_horizon: str = "multi_step"  # single_step, multi_step
    
    # --- Data ---
    training_data_description: str = ""
    training_data_period: Optional[str] = None
    evaluation_data_description: str = ""
    evaluation_data_period: Optional[str] = None
    data_preprocessing: str = ""
    
    # --- Architecture ---
    architecture: str = "MoE_RosaRoja"  # MoE_RosaRoja, Taylor, Kalman, Statistical, etc.
    framework: str = "numpy"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # --- Performance ---
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    benchmark_results: Dict[str, float] = field(default_factory=dict)
    
    # --- I/O Schema ---
    inputs: List[ModelInput] = field(default_factory=list)
    outputs: List[ModelOutput] = field(default_factory=list)
    
    # --- Operational ---
    thresholds: ModelThresholds = field(default_factory=ModelThresholds)
    compute_requirements: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    # --- Governance ---
    license: str = "proprietary"
    regulatory_classification: str = "unclassified"
    ethical_considerations: str = ""
    bias_mitigation: str = ""
    privacy_impact: str = ""
    
    # --- Audit ---
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.model_id:
            raise ValueError("model_id is required")
        if not self.version:
            raise ValueError("version is required")
    
    def add_audit_entry(self, action: str, details: Dict[str, Any], actor: str = "system") -> None:
        """Add an entry to the audit trail."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "actor": actor,
            "details": details,
        }
        self.audit_trail.append(entry)
        self.updated_at = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (JSON-compatible)."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCard":
        """Deserialize from dictionary."""
        # Handle nested dataclasses
        metrics_data = data.pop("metrics", {})
        thresholds_data = data.pop("thresholds", {})
        inputs_data = data.pop("inputs", [])
        outputs_data = data.pop("outputs", [])
        
        metrics = ModelMetrics(**metrics_data) if metrics_data else ModelMetrics()
        thresholds = ModelThresholds(**thresholds_data) if thresholds_data else ModelThresholds()
        inputs = [ModelInput(**i) for i in inputs_data]
        outputs = [ModelOutput(**o) for o in outputs_data]
        
        return cls(
            metrics=metrics,
            thresholds=thresholds,
            inputs=inputs,
            outputs=outputs,
            **data
        )
    
    @classmethod
    def create_rosa_roja_card(
        cls,
        model_id: str = "rosa_roja_moe",
        version: str = "1.0.0",
        **kwargs
    ) -> "ModelCard":
        """Factory for Rosa Roja MoE ModelCard with sensible defaults."""
        return cls(
            model_id=model_id,
            version=version,
            name="Rosa Roja MoE Engine",
            description="Master orchestrator with Mixture of Experts, trajectory reasoning, and Bayesian active inference",
            domain="iot_timeseries",
            task_type="forecasting",
            target_horizon="multi_step",
            architecture="MoE_RosaRoja",
            framework="numpy",
            hyperparameters={
                "variance_penalty": 0.5,
                "theta_alpha": 0.95,
                "min_trajectory_len": 11,
                "max_trajectory_len": 15,
                "top_k": 5,
                "rhythm_weight": 0.5,
                "max_entropy": 1.0,
                "oversample_factor": 3,
                "max_random_walk_steps": 100,
                "invalidation_threshold": 0.5,
            },
            inputs=[
                ModelInput(
                    name="delta_state",
                    type="vector",
                    shape=[-1],
                    dtype="float64",
                    description="State change vector ΔS (multidimensional)",
                ),
                ModelInput(
                    name="delta_time",
                    type="scalar",
                    dtype="float64",
                    description="Time delta Δt",
                ),
            ],
            outputs=[
                ModelOutput(
                    name="action",
                    type="scalar",
                    dtype="string",
                    description="Action: EXECUTE, HOLD, EMERGENCY_FLUSH",
                ),
                ModelOutput(
                    name="phi_moe",
                    type="scalar",
                    dtype="float32",
                    description="Master Equation output Phi_MoE in [0,1]",
                    range_min=0.0,
                    range_max=1.0,
                ),
                ModelOutput(
                    name="trajectory",
                    type="vector",
                    shape=[-1],
                    dtype="float64",
                    description="Predicted state trajectory",
                ),
                ModelOutput(
                    name="decision_trace",
                    type="tensor",
                    description="ISO 22989 decision trace with telemetry hash, lambda_t, phi_ritmo, expert confidences",
                ),
            ],
            thresholds=ModelThresholds(
                gamma_exec=0.5,
                geometric_threshold=0.3,
                min_history_points=50,
                max_latency_ms=5.0,
                confidence_floor=0.05,
                confidence_ceiling=0.95,
            ),
            intended_use="Real-time state estimation and action planning for IoT time series",
            ethical_considerations="No PII processed. Financial/industrial control actions require human oversight.",
            **kwargs
        )


# Backward compatibility exports
__all__ = [
    "ModelCard",
    "ModelMetrics",
    "ModelInput",
    "ModelOutput",
    "ModelThresholds",
]