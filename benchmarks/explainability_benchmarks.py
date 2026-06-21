"""
Latency benchmarks for contextual explainability components.

Measures latency for:
- ContextualConfidenceCalculator.calculate()
- HistoricalContextAggregator.aggregate()
- RecommendationGenerator.generate()
- ContextualExplainabilityEngine.generate_explanation()
- OperationalSummaryBuilder.build()
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import statistics
from typing import Dict, Any
from unittest.mock import Mock

from infrastructure.ml.cognitive.explainability.contextual_confidence_calculator import ContextualConfidenceCalculator
from infrastructure.ml.cognitive.explainability.historical_context_aggregator import HistoricalContextAggregator
from infrastructure.ml.cognitive.explainability.recommendation_generator import RecommendationGenerator
from infrastructure.ml.cognitive.explainability.contextual_explainability_engine import ContextualExplainabilityEngine
from infrastructure.ml.cognitive.explainability.operational_summary_builder import OperationalSummaryBuilder
from infrastructure.ml.cognitive.memory.historical_similarity_retriever import HistoricalSimilarityRetriever
from infrastructure.ml.cognitive.memory.cognitive_memory_registry import CognitiveMemoryRegistry
from domain.entities.memory import MemoryEvent
from domain.entities.explainability import ContextualExplanation


class ExplainabilityLatencyBenchmark:
    """Benchmark latency for explainability operations."""
    
    def __init__(self, iterations: int = 100):
        """
        Initialize benchmark.
        
        Args:
            iterations: Number of iterations for each benchmark
        """
        self.iterations = iterations
    
    def benchmark_confidence_calculator(self) -> Dict[str, float]:
        """Benchmark ContextualConfidenceCalculator.calculate()."""
        calculator = ContextualConfidenceCalculator()
        
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            calculator.calculate(
                anomaly_score=0.85,
                retrieval_similarity=0.8,
                regime_confidence=0.9,
                feature_stability=0.7,
            )
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": statistics.quantiles(latencies, n=100)[94],
            "p99_ms": statistics.quantiles(latencies, n=100)[98],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }
    
    def benchmark_historical_aggregator(self) -> Dict[str, float]:
        """Benchmark HistoricalContextAggregator.aggregate()."""
        aggregator = HistoricalContextAggregator()
        
        # Create sample events
        events = [
            MemoryEvent(
                sensor_id=12345,
                sensor_type="TEMPERATURE",
                timestamp=1234567890.0 + i,
                event_type="ANOMALY_CONFIRMED",
                semantic_text="Sensor 12345...",
                regime="STARTUP" if i % 2 == 0 else "STABLE_NORMAL",
                anomaly_score=0.8 + i * 0.02,
                dynamic_features={"derivative": 2.0 + i * 0.1},
                metadata={"value": 85.0 + i},
            )
            for i in range(10)
        ]
        
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            aggregator.aggregate(events)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)
        
        return {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": statistics.quantiles(latencies, n=100)[94],
            "p99_ms": statistics.quantiles(latencies, n=100)[98],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }
    
    def benchmark_recommendation_generator(self) -> Dict[str, float]:
        """Benchmark RecommendationGenerator.generate()."""
        generator = RecommendationGenerator()
        
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            generator.generate(
                regime="STARTUP",
                anomaly_score=0.85,
                dynamic_features={"derivative": 2.5, "rolling_std_1h": 8.5},
                historical_patterns=["STARTUP"],
            )
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)
        
        return {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": statistics.quantiles(latencies, n=100)[94],
            "p99_ms": statistics.quantiles(latencies, n=100)[98],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }
    
    def benchmark_explainability_engine(self) -> Dict[str, float]:
        """Benchmark ContextualExplainabilityEngine.generate_explanation()."""
        mock_retriever = Mock(spec=HistoricalSimilarityRetriever)
        mock_retriever.retrieve.return_value = []
        registry = CognitiveMemoryRegistry()
        
        engine = ContextualExplainabilityEngine(
            similarity_retriever=mock_retriever,
            registry=registry,
        )
        
        ml_features = {
            "timestamp": 1234567890.0,
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "trend": "increasing",
            "stability": 0.5,
            "dynamic_features": {
                "derivative": 2.5,
                "rolling_std_1h": 8.5,
            },
        }
        
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            engine.generate_explanation(
                sensor_id=12345,
                sensor_type="TEMPERATURE",
                ml_features=ml_features,
                regime="STARTUP",
                anomaly_score=0.85,
                regime_confidence=0.9,
            )
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)
        
        return {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": statistics.quantiles(latencies, n=100)[94],
            "p99_ms": statistics.quantiles(latencies, n=100)[98],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }
    
    def benchmark_summary_builder(self) -> Dict[str, float]:
        """Benchmark OperationalSummaryBuilder.build()."""
        builder = OperationalSummaryBuilder()
        
        explanation = ContextualExplanation(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=1234567890.0,
            current_regime="STARTUP",
            anomaly_score=0.85,
            primary_drivers=["Desviación Z-score (3.20σ)", "Tasa de cambio (2.50)"],
            dynamic_context={"current_value": 85.2, "baseline": 45.0},
            similar_event_count=3,
            historical_context="3 eventos similares encontrados.",
            historical_patterns=["Patrones recurrentes en régimen STARTUP"],
            operational_confidence=0.82,
            suggested_actions=["Monitorear ramp-up", "Verificar estabilidad"],
        )
        
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            builder.build(explanation)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)
        
        return {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": statistics.quantiles(latencies, n=100)[94],
            "p99_ms": statistics.quantiles(latencies, n=100)[98],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }
    
    def run_all_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Run all benchmarks and return results."""
        results = {}
        
        print("Running ContextualConfidenceCalculator benchmark...")
        results["contextual_confidence_calculator"] = self.benchmark_confidence_calculator()
        
        print("Running HistoricalContextAggregator benchmark...")
        results["historical_context_aggregator"] = self.benchmark_historical_aggregator()
        
        print("Running RecommendationGenerator benchmark...")
        results["recommendation_generator"] = self.benchmark_recommendation_generator()
        
        print("Running ContextualExplainabilityEngine benchmark...")
        results["contextual_explainability_engine"] = self.benchmark_explainability_engine()
        
        print("Running OperationalSummaryBuilder benchmark...")
        results["operational_summary_builder"] = self.benchmark_summary_builder()
        
        return results
    
    def print_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Print benchmark results."""
        print("\n" + "=" * 80)
        print("EXPLAINABILITY LATENCY BENCHMARK RESULTS")
        print("=" * 80)
        
        for component, metrics in results.items():
            print(f"\n{component}:")
            print(f"  Mean:   {metrics['mean_ms']:.3f} ms")
            print(f"  Median: {metrics['median_ms']:.3f} ms")
            print(f"  P95:    {metrics['p95_ms']:.3f} ms")
            print(f"  P99:    {metrics['p99_ms']:.3f} ms")
            print(f"  Min:    {metrics['min_ms']:.3f} ms")
            print(f"  Max:    {metrics['max_ms']:.3f} ms")
        
        print("\n" + "=" * 80)
        print("TARGET: Latencia explainability <50ms")
        print("=" * 80)


if __name__ == "__main__":
    benchmark = ExplainabilityLatencyBenchmark(iterations=100)
    results = benchmark.run_all_benchmarks()
    benchmark.print_results(results)
