"""CLI entrypoint — extract training data and write training_data.json.

Usage:
    python -m scripts.extract_training_data
    python scripts/extract_training_data.py

Output:
    scripts/training_data.json
"""

from __future__ import annotations

import json
import logging
import os
import sys

# Allow running as a script from the iot_machine_learning root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.training_extractor import (  # noqa: E402
    build_training_records,
    compute_summary,
    fetch_rows,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "training_data.json")


def _print_summary(summary: dict) -> None:
    print("\n── Domain distribution ──")
    for domain, count in sorted(summary["by_domain"].items()):
        print(f"  {domain:<20} {count}")

    print("\n── Severity distribution ──")
    for severity, count in sorted(summary["by_severity"].items()):
        print(f"  {severity:<20} {count}")


def main() -> int:
    """Extract, filter, save, and print summary. Returns exit code."""
    logger.info("[CLI] Starting training data extraction")

    try:
        rows = fetch_rows()
    except Exception as exc:
        logger.error("[CLI] DB fetch failed: %s", exc)
        return 1

    records = build_training_records(rows)

    if not records:
        logger.warning("[CLI] No valid records found — training_data.json not written")
        return 0

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)

    logger.info("[CLI] Saved %d records to %s", len(records), _OUTPUT_PATH)

    summary = compute_summary(records)
    print(f"\nTotal valid records: {len(records)}")
    _print_summary(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
