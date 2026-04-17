"""Validation script for SemanticEnrichment evaluation system."""

from __future__ import annotations

from iot_machine_learning.application.evaluation import (
    EvaluationRunner,
    TestDataset,
    ReportGenerator,
)


def main() -> None:
    """Run validation on evaluation system itself."""
    print("=" * 70)
    print("SEMANTIC ENRICHMENT EVALUATION SYSTEM VALIDATION")
    print("=" * 70)
    
    # 1. Validate dataset
    print("\n1. DATASET VALIDATION")
    print("-" * 40)
    all_cases = TestDataset.all_cases()
    print(f"   Total test cases: {len(all_cases)}")
    
    for cat in ["industrial", "neutral", "noise"]:
        cases = TestDataset.get_by_category(cat)
        print(f"   {cat}: {len(cases)} cases")
        for case in cases:
            print(f"      - {case.id}: {case.description[:40]}...")
    
    # 2. Validate EvaluationRunner (quick smoke test)
    print("\n2. EVALUATION RUNNER SMOKE TEST")
    print("-" * 40)
    runner = EvaluationRunner(budget_ms=1000.0, deterministic_mode=True)
    
    # Test single case
    test_case = TestDataset.get_by_id("IND-001")
    if test_case:
        print(f"   Testing: {test_case.id} - {test_case.description}")
        print(f"   Input text: {test_case.text[:60]}...")
        print(f"   Expected entities: {test_case.expected_entities}")
        print(f"   Expected metrics: {test_case.expected_metrics}")
        print(f"   Has critical: {test_case.has_critical}")
        
        try:
            result = runner.run_single(test_case)
            print(f"\n   ✓ Control severity: {result.control_output.severity}")
            print(f"   ✓ Treatment severity: {result.treatment_output.severity}")
            print(f"   ✓ Entity delta: {result.metrics.entity_count_delta}")
            print(f"   ✓ Quality improvement: {result.quality_report.improvement} pts")
            print(f"   ✓ Assessment: {result.quality_report.assessment}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. Validate ReportGenerator
    print("\n3. REPORT GENERATOR VALIDATION")
    print("-" * 40)
    if runner.results:
        generator = ReportGenerator(runner.results)
        try:
            report = generator.generate_full_report()
            print(f"   ✓ Report generated successfully")
            print(f"   Summary keys: {list(report.get('summary', {}).keys())}")
            print(f"   Categories: {list(report.get('by_category', {}).keys())}")
            print(f"   Documents in report: {len(report.get('documents', []))}")
        except Exception as e:
            print(f"   ✗ Report generation failed: {e}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
