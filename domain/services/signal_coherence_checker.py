"""Backward-compatible shim — real module at pattern/signal_coherence_checker.py"""
import importlib as _il, sys as _sys
_sys.modules[__name__] = _il.import_module('.pattern.signal_coherence_checker', __package__)
