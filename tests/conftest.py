"""Root conftest.py for pytest configuration."""

import sys
from pathlib import Path

# Add project root and iot_machine_learning to Python path
project_root = Path(__file__).parent.parent
iot_ml_path = project_root / "iot_machine_learning"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(iot_ml_path))

def pytest_configure(config):
    """Ensure paths are set early."""
    project_root = Path(__file__).parent.parent
    iot_ml_path = project_root / "iot_machine_learning"
    for path in [str(project_root), str(iot_ml_path)]:
        if path not in sys.path:
            sys.path.insert(0, path)