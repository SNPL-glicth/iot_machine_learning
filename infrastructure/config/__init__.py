"""Configuration factory for MoE integration.

Exports:
- create_moe_gateway: Factory for MoEGateway instantiation
- create_moe_gateway_safe: Safe version that never raises
"""

from .moe_factory import create_moe_gateway, create_moe_gateway_safe

__all__ = ["create_moe_gateway", "create_moe_gateway_safe"]
