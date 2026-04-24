"""TemplateSelector — selecciona plantillas del JSON.

Responsabilidad única: cargar y seleccionar plantillas.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List


class TemplateSelector:
    """Selecciona plantillas aleatoriamente por intención."""
    
    def __init__(self, responses_path: str | None = None) -> None:
        if responses_path is None:
            base = Path(__file__).parent.parent.parent / "ml_service" / "config"
            responses_path = str(base / "chat_responses.json")
        self._responses = self._load(responses_path)
    
    def _load(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "greeting": ["Hola, ¿en qué puedo ayudarte?"],
                "acknowledge": ["Entendido."],
                "no_context": ["No tengo datos previos."],
                "error": ["No pude procesar tu mensaje."],
            }
    
    def select(self, category: str, defaults: List[str] | None = None) -> str:
        """Selecciona plantilla aleatoria de categoría."""
        options = self._responses.get(category, defaults or ["Entendido."])
        return random.choice(options)
