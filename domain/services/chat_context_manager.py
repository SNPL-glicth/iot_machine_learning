"""ChatContextManager — gestión de historial conversacional.

Responsabilidad: mantener contexto de sesiones con ventana deslizante.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class ChatMessage:
    """Mensaje en la conversación."""
    role: str  # "user" | "assistant"
    content: str
    intent: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class ChatContext:
    """Contexto completo de una sesión."""
    session_id: str
    messages: List[ChatMessage]
    last_analysis_result: Optional[Dict]
    last_intent: Optional[str]
    message_count: int
    created_at: float
    last_activity: float


class ChatContextManager:
    """Gestiona contextos conversacionales."""
    
    def __init__(
        self,
        window_size: int = 20,
        ttl_seconds: int = 3600,
        redis_client: Optional[Any] = None,
    ) -> None:
        """Inicializa con configuración."""
        self._window = window_size
        self._ttl = ttl_seconds
        self._redis = redis_client
        self._memory: Dict[str, ChatContext] = {}
    
    def get_context(self, session_id: str) -> ChatContext:
        """Recupera o crea contexto."""
        # Intentar Redis primero
        if self._redis:
            data = self._redis.get(f"chat_ctx:{session_id}")
            if data:
                return self._deserialize(json.loads(data))
        
        # Fallback a memoria
        if session_id in self._memory:
            return self._memory[session_id]
        
        # Crear nuevo
        now = time.time()
        ctx = ChatContext(
            session_id=session_id,
            messages=[],
            last_analysis_result=None,
            last_intent=None,
            message_count=0,
            created_at=now,
            last_activity=now,
        )
        self._memory[session_id] = ctx
        return ctx
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Añade mensaje al contexto."""
        ctx = self.get_context(session_id)
        
        msg = ChatMessage(
            role=role,
            content=content,
            intent=intent,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        ctx.messages.append(msg)
        ctx.message_count += 1
        ctx.last_intent = intent
        ctx.last_activity = time.time()
        
        # Aplicar ventana deslizante
        if len(ctx.messages) > self._window:
            # Conservar primero, eliminar intermedios antiguos
            first = ctx.messages[0]
            ctx.messages = [first] + ctx.messages[-(self._window - 1):]
        
        self._persist(ctx)
    
    def set_analysis_result(self, session_id: str, result: Dict) -> None:
        """Guarda último resultado de análisis."""
        ctx = self.get_context(session_id)
        ctx.last_analysis_result = result
        self._persist(ctx)
    
    def summarize_context(self, session_id: str) -> str:
        """Genera resumen textual del contexto."""
        ctx = self.get_context(session_id)
        
        parts = [f"El usuario ha enviado {ctx.message_count} mensajes."]
        
        if ctx.last_analysis_result:
            count = ctx.last_analysis_result.get("incident_count", 0)
            sev = ctx.last_analysis_result.get("max_severity", "desconocida")
            parts.append(f"El último análisis detectó {count} incidencias con severidad {sev}.")
        
        intents = [m.intent for m in ctx.messages[-5:]]
        if intents:
            unique = set(intents)
            parts.append(f"Temas recientes: {', '.join(unique)}.")
        
        return " ".join(parts)
    
    def _persist(self, ctx: ChatContext) -> None:
        """Persiste contexto."""
        data = self._serialize(ctx)
        
        if self._redis:
            self._redis.setex(
                f"chat_ctx:{ctx.session_id}",
                self._ttl,
                json.dumps(data)
            )
        else:
            self._memory[ctx.session_id] = ctx
    
    def _serialize(self, ctx: ChatContext) -> Dict:
        """Serializa a dict."""
        return {
            "session_id": ctx.session_id,
            "messages": [asdict(m) for m in ctx.messages],
            "last_analysis_result": ctx.last_analysis_result,
            "last_intent": ctx.last_intent,
            "message_count": ctx.message_count,
            "created_at": ctx.created_at,
            "last_activity": ctx.last_activity,
        }
    
    def _deserialize(self, data: Dict) -> ChatContext:
        """Deserializa desde dict."""
        return ChatContext(
            session_id=data["session_id"],
            messages=[ChatMessage(**m) for m in data.get("messages", [])],
            last_analysis_result=data.get("last_analysis_result"),
            last_intent=data.get("last_intent"),
            message_count=data.get("message_count", 0),
            created_at=data.get("created_at", 0),
            last_activity=data.get("last_activity", 0),
        )
