#!/usr/bin/env python
"""Direct validation script for CRIT-1, CRIT-2, CRIT-3 fixes.

This script validates the fixes without requiring pytest infrastructure.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("VALIDATING CRITICAL FIXES")
print("=" * 80)

# CRIT-1: Validate AdaptPhase has instance-scoped cache
print("\n[CRIT-1] Validating AdaptPhase instance-scoped cache...")
try:
    from infrastructure.ml.cognitive.orchestration.phases.adapt_phase import AdaptPhase, WeightCache
    
    # Check that global _weight_cache was removed
    import infrastructure.ml.cognitive.orchestration.phases.adapt_phase as adapt_module
    if hasattr(adapt_module, '_weight_cache'):
        print("  ❌ FAIL: Global _weight_cache still exists!")
        sys.exit(1)
    else:
        print("  ✓ PASS: Global _weight_cache removed")
    
    # Check that AdaptPhase has __init__
    phase = AdaptPhase()
    if hasattr(phase, '_weight_cache'):
        print("  ✓ PASS: AdaptPhase has instance _weight_cache")
    else:
        print("  ❌ FAIL: AdaptPhase missing instance _weight_cache")
        sys.exit(1)
    
    # Check that two instances have different caches
    phase2 = AdaptPhase()
    if phase._weight_cache is not phase2._weight_cache:
        print("  ✓ PASS: Each AdaptPhase instance has its own cache")
    else:
        print("  ❌ FAIL: AdaptPhase instances share same cache")
        sys.exit(1)
    
    print("[CRIT-1] ✓ ALL CHECKS PASSED")
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# CRIT-2: Validate InhibitPhase doesn't reference _weight_mediator
print("\n[CRIT-2] Validating InhibitPhase doesn't reference _weight_mediator...")
try:
    import inspect
    from infrastructure.ml.cognitive.orchestration.phases.inhibit_phase import InhibitPhase
    
    source = inspect.getsource(InhibitPhase.execute)
    if '_weight_mediator' in source:
        print("  ❌ FAIL: InhibitPhase still references _weight_mediator")
        sys.exit(1)
    else:
        print("  ✓ PASS: InhibitPhase doesn't reference _weight_mediator")
    
    if 'plasticity_weights' in source:
        print("  ✓ PASS: InhibitPhase uses plasticity_weights directly")
    else:
        print("  ❌ FAIL: InhibitPhase doesn't use plasticity_weights")
        sys.exit(1)
    
    print("[CRIT-2] ✓ ALL CHECKS PASSED")
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# CRIT-3: Validate WeightResolutionService epsilon default is 0.01
print("\n[CRIT-3] Validating WeightResolutionService epsilon default...")
try:
    from infrastructure.ml.cognitive.orchestration.weight_resolution_service import WeightResolutionService
    
    # Check default epsilon is 0.01
    service = WeightResolutionService(
        base_weights={"engine1": 0.5, "engine2": 0.5}
    )
    if service._epsilon == 0.01:
        print("  ✓ PASS: Default epsilon is 0.01")
    else:
        print(f"  ❌ FAIL: Default epsilon is {service._epsilon}, expected 0.01")
        sys.exit(1)
    
    # Check epsilon=None uses default
    service2 = WeightResolutionService(
        base_weights={"engine1": 0.5, "engine2": 0.5},
        epsilon=None
    )
    if service2._epsilon == 0.01:
        print("  ✓ PASS: epsilon=None uses default 0.01")
    else:
        print(f"  ❌ FAIL: epsilon=None gives {service2._epsilon}, expected 0.01")
        sys.exit(1)
    
    # Check explicit epsilon is respected
    service3 = WeightResolutionService(
        base_weights={"engine1": 0.5, "engine2": 0.5},
        epsilon=0.05
    )
    if service3._epsilon == 0.05:
        print("  ✓ PASS: Explicit epsilon is respected")
    else:
        print(f"  ❌ FAIL: Explicit epsilon is {service3._epsilon}, expected 0.05")
        sys.exit(1)
    
    # Check epsilon validation
    try:
        service4 = WeightResolutionService(
            base_weights={"engine1": 0.5, "engine2": 0.5},
            epsilon=0.0
        )
        print("  ❌ FAIL: epsilon=0.0 should raise ValueError")
        sys.exit(1)
    except ValueError as e:
        if "epsilon must be positive" in str(e):
            print("  ✓ PASS: epsilon=0.0 raises ValueError")
        else:
            print(f"  ❌ FAIL: Wrong error message: {e}")
            sys.exit(1)
    
    print("[CRIT-3] ✓ ALL CHECKS PASSED")
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL CRITICAL FIXES VALIDATED SUCCESSFULLY")
print("=" * 80)
