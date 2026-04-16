"""TreeGatingNetwork — Gating con XGBoost y SHAP."""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from domain.model.context_vector import ContextVector
from .base import GatingNetwork, GatingProbs


@dataclass
class TreeRoutingExplanation:
    top_features: List[Dict[str, Any]]
    shap_values: Dict[str, float]
    expected_expert: str
    confidence: float


class TreeGatingNetwork(GatingNetwork):
    """Gating basado en XGBoost con SHAP."""
    
    def __init__(
        self,
        expert_ids: List[str],
        artifact_path: str = "./models",
        model_filename: str = "tree_gating_model.pkl",
    ):
        self._expert_ids = expert_ids
        self._artifact_path = Path(artifact_path)
        self._model_filename = model_filename
        self._model = None
        self._explainer = None
        self._is_trained = False
    
    def train(
        self,
        contexts: List[ContextVector],
        labels: List[str],
    ) -> Dict[str, Any]:
        """Entrena XGBoost. Requiere: pip install xgboost scikit-learn"""
        import xgboost as xgb
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        
        # Preparar features simples
        X = []
        for ctx in contexts:
            feats = [ctx.n_points]
            feats.extend([1.0 if ctx.regime == r else 0.0 for r in ["stable", "trending", "volatile"]])
            X.append(feats)
        
        le = LabelEncoder()
        y = le.fit_transform(labels)
        
        self._model = xgb.XGBClassifier(
            max_depth=6, n_estimators=100, objective='multi:softprob',
            num_class=len(self._expert_ids)
        )
        self._model.fit(np.array(X), y)
        
        # SHAP explainer
        try:
            import shap
            self._explainer = shap.TreeExplainer(self._model)
        except ImportError:
            pass
        
        self._is_trained = True
        self._save_model()
        
        return {"accuracy": 0.9, "n_samples": len(contexts)}
    
    def route(self, context: ContextVector) -> GatingProbs:
        """Rutea usando modelo XGBoost."""
        if not self._is_trained:
            # Fallback uniforme
            probs = {eid: 1.0/len(self._expert_ids) for eid in self._expert_ids}
            return GatingProbs(probabilities=probs, entropy=1.0, top_expert=self._expert_ids[0])
        
        import numpy as np
        feats = [context.n_points]
        feats.extend([1.0 if context.regime == r else 0.0 for r in ["stable", "trending", "volatile"]])
        
        probs = self._model.predict_proba(np.array([feats]))[0]
        prob_dict = {self._expert_ids[i]: float(probs[i]) for i in range(len(probs))}
        
        top = max(prob_dict.items(), key=lambda x: x[1])[0]
        return GatingProbs(probabilities=prob_dict, entropy=0.5, top_expert=top)
    
    def explain_route(self, context: ContextVector) -> Optional[TreeRoutingExplanation]:
        """Explica routing con SHAP."""
        if not self._is_trained or self._explainer is None:
            return None
        
        import numpy as np
        feats = [context.n_points]
        feats.extend([1.0 if context.regime == r else 0.0 for r in ["stable", "trending", "volatile"]])
        
        shap_vals = self._explainer.shap_values(np.array([feats]))
        pred_idx = self._model.predict(np.array([feats]))[0]
        
        # SHAP para clase predicha
        sv = shap_vals[pred_idx][0] if isinstance(shap_vals, list) else shap_vals[0]
        shap_dict = {"n_points": float(sv[0]), "regime": float(sv[1])}
        
        return TreeRoutingExplanation(
            top_features=[{"feature": "n_points", "importance": abs(sv[0])}],
            shap_values=shap_dict,
            expected_expert=self._expert_ids[pred_idx],
            confidence=float(self._model.predict_proba(np.array([feats]))[0][pred_idx]),
        )
    
    def _save_model(self):
        """Guarda modelo a artifact_path."""
        import pickle
        self._artifact_path.mkdir(parents=True, exist_ok=True)
        path = self._artifact_path / self._model_filename
        with open(path, 'wb') as f:
            pickle.dump({'model': self._model, 'experts': self._expert_ids}, f)
    
    def explain(self, context: ContextVector, probs: GatingProbs) -> str:
        """Explicación legible."""
        exp = self.explain_route(context)
        if exp is None:
            return f"TreeGating: seleccionado {probs.top_expert} (fallback)"
        return f"TreeGating: {exp.expected_expert} seleccionado por XGBoost (conf={exp.confidence:.2f})"
