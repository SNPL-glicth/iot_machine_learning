"""Tool abstract base class.

Defines the interface all tools must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .tool_guard import GuardResult
from .tool_models import ToolContext, ToolResult


class Tool(ABC):
    """Abstract base class for all executable tools.
    
    All tools must implement:
    - name: unique identifier
    - description: human-readable purpose
    - parameters: JSON Schema for arguments
    - can_execute(): safety guard evaluation
    - execute(): actual execution logic
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier (snake_case)."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        pass
    
    @property
    def version(self) -> str:
        """Tool version for compatibility tracking."""
        return "1.0.0"
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass
    
    @abstractmethod
    def can_execute(self, context: ToolContext) -> GuardResult:
        """Evaluate if tool can be executed in given context.
        
        This is the SAFETY GATE. It must be deterministic and fast.
        No side effects allowed here.
        """
        pass
    
    @abstractmethod
    def execute(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate parameters against schema (basic implementation)."""
        schema = self.parameters
        if schema.get("type") != "object":
            return True, None
        
        required = schema.get("required", [])
        for key in required:
            if key not in params:
                return False, f"Missing required parameter: {key}"
        
        return True, None


# Import Optional here to avoid circular imports
from typing import Optional
