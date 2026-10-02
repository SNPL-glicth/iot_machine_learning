"""Wrapper de compatibilidad para ejecutar el benchmark de Kuramoto Consensus Gate."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.adaptive_gate.run_adaptive_gate_benchmark import main

if __name__ == "__main__":
    main()
