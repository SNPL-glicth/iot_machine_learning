"""Weaviate Schema Creation Script — Cognitive Memory Layer.

Creates the 4 Weaviate classes for the UTSAE Cognitive Memory:
  - MLExplanation:      Prediction reasoning (vectorized explanations)
  - AnomalyMemory:      Anomaly detection traces (vectorized explanations + context)
  - PatternMemory:       Behavioral pattern descriptions (vectorized descriptions)
  - DecisionReasoning:  Decision orchestrator reasoning chains (vectorized summaries)

Usage:
    # Ensure Weaviate is running (docker-compose.cognitive.yml)
    python -m iot_machine_learning.scripts.create_weaviate_schema

    # Custom URL:
    python -m iot_machine_learning.scripts.create_weaviate_schema --url http://localhost:8080

    # Dry-run (print schema JSON without applying):
    python -m iot_machine_learning.scripts.create_weaviate_schema --dry-run

    # Delete existing classes first (for development reset):
    python -m iot_machine_learning.scripts.create_weaviate_schema --recreate

Requirements:
    pip install weaviate-client>=4.0.0

Note: This script delegates to modularized schema modules in
      infrastructure/persistence/vector/schema/
"""

from __future__ import annotations

import argparse
import logging
import sys

from iot_machine_learning.infrastructure.persistence.vector.schema import create_schema

logger = logging.getLogger(__name__)

WEAVIATE_DEFAULT_URL = "http://localhost:8080"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create Weaviate schema for UTSAE Cognitive Memory Layer."
    )
    parser.add_argument(
        "--url",
        default=WEAVIATE_DEFAULT_URL,
        help=f"Weaviate REST API URL (default: {WEAVIATE_DEFAULT_URL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schema JSON without applying to Weaviate.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing classes before creating (development only).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    success = create_schema(
        args.url,
        dry_run=args.dry_run,
        recreate=args.recreate,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
