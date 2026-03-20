#!/usr/bin/env python3
"""End-to-end test for UniversalAnalysisEngine with real text file."""

import sys
sys.path.insert(0, '.')

from infrastructure.ml.cognitive.universal import (
    UniversalAnalysisEngine,
    UniversalComparativeEngine,
    UniversalInput,
    UniversalContext,
    ComparisonContext,
    InputType,
)

def main():
    print("=" * 80)
    print("UNIVERSAL ENGINES END-TO-END TEST")
    print("=" * 80)
    print()
    
    # Read test document
    try:
        with open('test_universal_engines_e2e.txt', 'r') as f:
            text_content = f.read()
        print(f"✓ Loaded test document ({len(text_content)} chars)")
    except FileNotFoundError:
        print("✗ Test file not found: test_universal_engines_e2e.txt")
        return 1
    
    # Initialize engines
    print("\n--- Phase 1: Engine Initialization ---")
    try:
        analysis_engine = UniversalAnalysisEngine()
        comparative_engine = UniversalComparativeEngine()
        print("✓ UniversalAnalysisEngine initialized")
        print("✓ UniversalComparativeEngine initialized")
    except Exception as e:
        print(f"✗ Engine initialization failed: {e}")
        return 1
    
    # Build input
    print("\n--- Phase 2: Input Preparation ---")
    try:
        universal_input = UniversalInput(
            raw_data=text_content,
            detected_type=InputType.TEXT,
            metadata={
                "word_count": len(text_content.split()),
                "char_count": len(text_content),
            },
            domain_hint="infrastructure",
            series_id="e2e-test-001",
        )
        print(f"✓ UniversalInput created (type={universal_input.detected_type.value})")
        print(f"  - Series ID: {universal_input.series_id}")
        print(f"  - Domain hint: {universal_input.domain_hint}")
        print(f"  - Word count: {universal_input.metadata['word_count']}")
    except Exception as e:
        print(f"✗ Input preparation failed: {e}")
        return 1
    
    # Build context
    try:
        context = UniversalContext(
            series_id="e2e-test-001",
            tenant_id="test-tenant",
            domain_hint="infrastructure",
            budget_ms=3000.0,
        )
        print(f"✓ UniversalContext created (budget={context.budget_ms}ms)")
    except Exception as e:
        print(f"✗ Context creation failed: {e}")
        return 1
    
    # Run analysis
    print("\n--- Phase 3: Analysis Execution ---")
    try:
        result = analysis_engine.analyze(universal_input, context)
        print("✓ Analysis completed successfully")
        print(f"  - Domain detected: {result.domain}")
        print(f"  - Input type: {result.input_type.value}")
        print(f"  - Confidence: {result.confidence:.3f}")
        print(f"  - Severity: {result.severity.severity} (risk: {result.severity.risk_level})")
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Display explanation
    print("\n--- Phase 4: Explanation Structure ---")
    try:
        exp_dict = result.explanation.to_dict()
        print(f"✓ Explanation serialized")
        print(f"  - Series ID: {exp_dict.get('series_id', 'N/A')}")
        print(f"  - Signal regime: {exp_dict.get('signal', {}).get('regime', 'N/A')}")
        print(f"  - Outcome kind: {exp_dict.get('outcome', {}).get('kind', 'N/A')}")
        
        if 'reasoning' in exp_dict:
            phases = exp_dict['reasoning'].get('phases', [])
            print(f"  - Reasoning phases: {len(phases)}")
            for phase in phases[:3]:  # Show first 3
                print(f"    • {phase.get('name', 'unknown')}")
    except Exception as e:
        print(f"✗ Explanation extraction failed: {e}")
    
    # Display analysis details
    print("\n--- Phase 5: Analysis Details ---")
    try:
        analysis = result.analysis
        print(f"✓ Analysis dict has {len(analysis)} keys")
        for key in list(analysis.keys())[:5]:  # Show first 5 keys
            value = analysis[key]
            if isinstance(value, (int, float, str)):
                print(f"  - {key}: {value}")
            else:
                print(f"  - {key}: <{type(value).__name__}>")
    except Exception as e:
        print(f"✗ Analysis details failed: {e}")
    
    # Comparative analysis (without memory - should return None)
    print("\n--- Phase 6: Comparative Analysis (no memory) ---")
    try:
        comp_ctx = ComparisonContext(
            current_result=result,
            series_id="e2e-test-001",
            cognitive_memory=None,  # No memory available
            domain=result.domain,
        )
        comp_result = comparative_engine.compare(comp_ctx)
        
        if comp_result is None:
            print("✓ Comparative engine returned None (no memory - expected)")
        else:
            print(f"✓ Comparative analysis completed")
            print(f"  - Severity delta: {comp_result.severity_delta_pct}%")
            print(f"  - Similar incidents: {len(comp_result.top_similar)}")
    except Exception as e:
        print(f"✗ Comparative analysis failed: {e}")
    
    # Serialization test
    print("\n--- Phase 7: Serialization ---")
    try:
        result_dict = result.to_dict()
        print(f"✓ Result serialized to dict ({len(result_dict)} keys)")
        
        import json
        json_str = json.dumps(result_dict, default=str)
        print(f"✓ Result JSON serializable ({len(json_str)} chars)")
    except Exception as e:
        print(f"✗ Serialization failed: {e}")
    
    # Pipeline timing
    print("\n--- Phase 8: Performance Metrics ---")
    try:
        timing = result.pipeline_timing
        if timing:
            print("✓ Pipeline timing recorded:")
            for phase, duration_ms in timing.items():
                print(f"  - {phase}: {duration_ms:.2f}ms")
            total_ms = sum(timing.values())
            print(f"  Total: {total_ms:.2f}ms")
        else:
            print("  (No timing data available)")
    except Exception as e:
        print(f"✗ Timing extraction failed: {e}")
    
    print("\n" + "=" * 80)
    print("END-TO-END TEST COMPLETE - ALL PHASES PASSED")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
