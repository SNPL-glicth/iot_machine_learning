#!/usr/bin/env python
"""Direct source code validation for CRIT-1, CRIT-2, CRIT-3 fixes.

This script validates the fixes by reading source files directly without importing.
"""

import os
import re

print("=" * 80)
print("VALIDATING CRITICAL FIXES (SOURCE CODE ANALYSIS)")
print("=" * 80)

# CRIT-1: Validate AdaptPhase has instance-scoped cache
print("\n[CRIT-1] Validating AdaptPhase instance-scoped cache...")
adapt_phase_path = "infrastructure/ml/cognitive/orchestration/phases/adapt_phase.py"

if not os.path.exists(adapt_phase_path):
    print(f"  ❌ FAIL: File not found: {adapt_phase_path}")
    exit(1)

with open(adapt_phase_path, 'r') as f:
    adapt_source = f.read()

# Check that global _weight_cache was removed
if re.search(r'^_weight_cache\s*=\s*WeightCache\(\)', adapt_source, re.MULTILINE):
    print("  ❌ FAIL: Global _weight_cache still exists!")
    exit(1)
else:
    print("  ✓ PASS: Global _weight_cache removed")

# Check that AdaptPhase has __init__ with _weight_cache
if 'def __init__(self' in adapt_source and 'self._weight_cache = WeightCache(' in adapt_source:
    print("  ✓ PASS: AdaptPhase.__init__ creates instance _weight_cache")
else:
    print("  ❌ FAIL: AdaptPhase missing instance _weight_cache in __init__")
    exit(1)

# Check for CRIT-1 comment
if 'CRIT-1 FIX' in adapt_source:
    print("  ✓ PASS: CRIT-1 fix comment present")
else:
    print("  ⚠ WARNING: CRIT-1 fix comment not found")

print("[CRIT-1] ✓ ALL CHECKS PASSED")

# CRIT-2: Validate InhibitPhase doesn't reference _weight_mediator
print("\n[CRIT-2] Validating InhibitPhase doesn't reference _weight_mediator...")
inhibit_phase_path = "infrastructure/ml/cognitive/orchestration/phases/inhibit_phase.py"

if not os.path.exists(inhibit_phase_path):
    print(f"  ❌ FAIL: File not found: {inhibit_phase_path}")
    exit(1)

with open(inhibit_phase_path, 'r') as f:
    inhibit_source = f.read()

# Remove comments before checking
lines = inhibit_source.split('\n')
code_lines = []
for line in lines:
    # Remove inline comments
    if '#' in line:
        line = line[:line.index('#')]
    code_lines.append(line)
code_without_comments = '\n'.join(code_lines)

if '_weight_mediator' in code_without_comments:
    print("  ❌ FAIL: InhibitPhase still references _weight_mediator in code")
    exit(1)
else:
    print("  ✓ PASS: InhibitPhase doesn't reference _weight_mediator in code")

if 'mediated_weights = ctx.plasticity_weights' in inhibit_source:
    print("  ✓ PASS: InhibitPhase uses plasticity_weights directly")
else:
    print("  ❌ FAIL: InhibitPhase doesn't use plasticity_weights directly")
    exit(1)

# Check for CRIT-2 comment
if 'CRIT-2 FIX' in inhibit_source:
    print("  ✓ PASS: CRIT-2 fix comment present")
else:
    print("  ⚠ WARNING: CRIT-2 fix comment not found")

print("[CRIT-2] ✓ ALL CHECKS PASSED")

# CRIT-3: Validate WeightResolutionService epsilon default is 0.01
print("\n[CRIT-3] Validating WeightResolutionService epsilon default...")
weight_resolution_path = "infrastructure/ml/cognitive/orchestration/weight_resolution_service.py"

if not os.path.exists(weight_resolution_path):
    print(f"  ❌ FAIL: File not found: {weight_resolution_path}")
    exit(1)

with open(weight_resolution_path, 'r') as f:
    weight_resolution_source = f.read()

# Check for _DEFAULT_EPSILON = 0.01
if '_DEFAULT_EPSILON: float = 0.01' in weight_resolution_source:
    print("  ✓ PASS: _DEFAULT_EPSILON = 0.01 defined")
else:
    print("  ❌ FAIL: _DEFAULT_EPSILON = 0.01 not found")
    exit(1)

# Check that __init__ uses epsilon: Optional[float] = None
if 'epsilon: Optional[float] = None' in weight_resolution_source:
    print("  ✓ PASS: __init__ epsilon parameter is Optional[float] = None")
else:
    print("  ❌ FAIL: __init__ epsilon parameter not Optional[float] = None")
    exit(1)

# Check for validation logic
if 'if epsilon is None:' in weight_resolution_source and 'epsilon = self._DEFAULT_EPSILON' in weight_resolution_source:
    print("  ✓ PASS: Uses _DEFAULT_EPSILON when epsilon is None")
else:
    print("  ❌ FAIL: Doesn't use _DEFAULT_EPSILON when epsilon is None")
    exit(1)

# Check for validation of epsilon > 0
if 'if epsilon <= 0:' in weight_resolution_source and 'ValueError' in weight_resolution_source:
    print("  ✓ PASS: Validates epsilon > 0")
else:
    print("  ❌ FAIL: Doesn't validate epsilon > 0")
    exit(1)

# Check for CRIT-3 comment
if 'CRIT-3 FIX' in weight_resolution_source:
    print("  ✓ PASS: CRIT-3 fix comment present")
else:
    print("  ⚠ WARNING: CRIT-3 fix comment not found")

print("[CRIT-3] ✓ ALL CHECKS PASSED")

print("\n" + "=" * 80)
print("✓ ALL CRITICAL FIXES VALIDATED SUCCESSFULLY")
print("=" * 80)
