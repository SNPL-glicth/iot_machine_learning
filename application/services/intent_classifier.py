"""IntentClassifier — clasificación determinista de intenciones.

Orquesta TF-IDF + CharacterEncoder para clasificar texto.
Sin LLMs externos, puramente reglas configurables.

Máximo 140 líneas.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from iot_machine_learning.domain.ports.text_encoder_port import TextEncoderPort
from iot_machine_learning.infrastructure.ml.cognitive.text.embeddings.word_encoder import (
    WordEncoder,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.encoders.character_encoder import (
    CharacterEncoder,
)


@dataclass
class IntentResult:
    """Resultado de clasificación."""
    intent: str  # conversational | analysis | metrics_query | command
    confidence: float
    signals: List[str]  # qué reglas activaron


class IntentClassifier:
    """Clasificador determinista de intenciones de texto."""
    
    def __init__(
        self,
        word_encoder: Optional[TextEncoderPort] = None,
        char_encoder: Optional[TextEncoderPort] = None,
        config_path: str | None = None,
    ) -> None:
        """Inicializa con encoders y config externa."""
        self._word_enc = word_encoder or self._default_word_encoder()
        self._char_enc = char_encoder or CharacterEncoder()
        self._config = self._load_config(config_path)
    
    def _default_word_encoder(self) -> TextEncoderPort:
        """Encoder por defecto."""
        return WordEncoder(vocab={}, output_dim=64)
    
    def _load_config(self, path: str | None) -> Dict:
        """Carga configuración desde JSON."""
        if path is None:
            base = Path(__file__).parent.parent.parent / "ml_service" / "config"
            path = str(base / "intent_keywords.json")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Config por defecto."""
        return {
            "conversational": ["hola", "gracias", "ok", "bien", "ayuda"],
            "analysis": ["incidente", "alerta", "anomalía", "fallo"],
            "metrics": ["último", "ayer", "promedio", "métricas"],
            "command": ["reinicia", "activa", "configura"],
            "thresholds": {"max_short": 120, "min_long": 200}
        }
    
    def classify(self, text: str, context: Optional[Dict] = None) -> IntentResult:
        """Clasifica texto según reglas configurables."""
        text_lower = text.lower().strip()
        signals: List[str] = []
        
        # Regla 1: Longitud corta = conversacional
        cfg_th = self._config.get("thresholds", {})
        if len(text) < cfg_th.get("max_short", 120):
            signals.append("short_length")
        
        # Regla 2: Keywords por categoría
        for intent_type, keywords in self._config.items():
            if intent_type == "thresholds":
                continue
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                signals.append(f"{intent_type}_keywords:{matches}")
        
        # Regla 3: Entidades técnicas (números + unidades)
        units_pattern = r'\d+\s*(?:°c|%|rpm|kpa|hz|volts|amps)'
        if re.search(units_pattern, text_lower):
            signals.append("technical_entities")
        
        # Regla 4: Signos de pregunta
        if text.endswith("?"):
            signals.append("question_mark")
        
        # Decidir intención
        intent = self._resolve_intent(signals, text)
        
        # Calcular confianza
        confidence = min(len(signals) * 0.25, 0.95)
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            signals=signals
        )
    
    def _resolve_intent(self, signals: List[str], text: str) -> str:
        """Resuelve intención final."""
        # Prioridad: command > analysis > metrics > conversational
        if any("command" in s for s in signals):
            return "command"
        if any("analysis" in s for s in signals) or len(text) > 200:
            return "analysis"
        if any("metrics" in s for s in signals):
            return "metrics_query"
        return "conversational"
