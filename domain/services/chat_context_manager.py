"""Backward-compatible shim — real module at cognitive/chat_context_manager.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.cognitive.chat_context_manager', __package__)
