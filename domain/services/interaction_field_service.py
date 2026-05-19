"""Backward-compatible shim — real module at cognitive/interaction_field_service.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.cognitive.interaction_field_service', __package__)
