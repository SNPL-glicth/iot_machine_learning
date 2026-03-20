"""Document analysis pipeline components.

Modular components for universal and legacy document analysis.
"""

from .result_builder import build_conclusion, build_output_dict
from .universal_bridge import analyze_with_universal, extract_raw_data
from .legacy_pipeline import analyze_with_legacy
from .neural_bridge import (
    analyze_with_neural,
    arbitrate_results,
    extract_analysis_scores,
)

__all__ = [
    "build_conclusion",
    "build_output_dict",
    "analyze_with_universal",
    "extract_raw_data",
    "analyze_with_legacy",
    "analyze_with_neural",
    "arbitrate_results",
    "extract_analysis_scores",
]
