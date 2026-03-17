"""Metadata-only analyzers for image, audio, and binary files."""

from __future__ import annotations

from typing import Any, Dict


def analyze_image(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze image metadata."""
    data = payload.get("data", {})
    return {
        "analysis": {"metadata": data, "triggers_activated": []},
        "adaptive_thresholds": {},
        "conclusion": (
            f"Imagen {data.get('format', 'unknown')} de "
            f"{data.get('width', 0)}x{data.get('height', 0)} píxeles, "
            f"{data.get('file_size_kb', 0)} KB. "
            f"Sin análisis de contenido visual."
        ),
        "confidence": 1.0,
    }


def analyze_audio(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze audio metadata."""
    data = payload.get("data", {})
    return {
        "analysis": {"metadata": data, "triggers_activated": []},
        "adaptive_thresholds": {},
        "conclusion": (
            f"Audio {data.get('format', 'unknown')} de "
            f"{data.get('duration_seconds', 0)}s, "
            f"{data.get('bitrate_kbps', 0)} kbps. "
            f"Sin análisis de contenido de audio."
        ),
        "confidence": 1.0,
    }


def analyze_binary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze binary file."""
    data = payload.get("data", {})
    return {
        "analysis": {"metadata": data, "triggers_activated": []},
        "adaptive_thresholds": {},
        "conclusion": (
            f"Archivo binario ({data.get('mime_type', 'application/octet-stream')}) "
            f"de {data.get('size_bytes', 0)} bytes. "
            f"No se puede analizar el contenido."
        ),
        "confidence": 1.0,
    }
