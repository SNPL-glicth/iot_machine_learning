"""
Latency benchmarks for operational memory components.

Measures latency for:
- SemanticEventBuilder.build()
- AnomalyMemoryStore.store()
- AnomalyMemoryStore.retrieve_similar()
- OperationalMemoryPipeline.process_event()
- HistoricalSimilarityRetriever.retrieve()
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import statistics
from typing import List, Dict, Any
from unittest.mock import Mock

from infrastructure.ml.cognitive.memory.semantic_event_builder import SemanticEventBuilder
from infrastructure.ml.cognitive.memory.anomaly_memory_store import AnomalyMemoryStore
from infrastructure.ml.cognitive.memory.operational_memory_pipeline import OperationalMemoryPipeline
from infrastructure.ml.cognitive.memory.historical_similarity_retriever import HistoricalSimilarityRetriever
from infrastructure.ml.cognitive.memory.cognitive_memory_registry import CognitiveMemoryRegistry


class LatencyBenchmark:
    """Benchmark latency for memory operations."""
    
    def __init__(self, iterations: int = 100):
        """
        Initialize benchmark.
        
        Args:
            iterations: Number of iterations for each benchmark
        """
        self.iterations = iterations
    
    def benchmark_semantic_event_builder(self) -> Dict[str, float]:
        """Benchmark SemanticEventBuilder.build()."""
        builder = SemanticEventBuilder()
        
        ml_features = {
            "timestamp": time.time(),
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "trend": "increasing",
            "stability": 0.5,
            "model_version": "2.0.0",
            "dynamic_features": {
                "derivative": 2.5,
                "rolling_std_1h": 8.5,
            },
        }
        
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            event = builder.build(
                sensor_id=12345,
                sensor_type="TEMPERATURE",
                ml_features=ml_features,
                regime="STARTUP",
                anomaly_score=0.85,
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
    
    def benchmark_anomaly_memory_store_store(self) -> Dict[str, float]:
        """Benchmark AnomalyMemoryStore.store()."""
        mock_client = Mock()
        mock_client.data_object.create.return_value = "test-id"
        
        store = AnomalyMemoryStore(weaviate_client=mock_client)
        
        from domain.entities.memory import MemoryEvent
        
        event = MemoryEvent(
            sensor_id=12345,
            sensor_type="TEMPERATURE",
            timestamp=time.time(),
            event_type="ANOMALY_CONFIRMED",
            semantic_text="Sensor 12345 (TEMPERATURE) en régimen STARTUP...",
            regime="STARTUP",
            anomaly_score=0.85,
            dynamic_features={"derivative": 2.5, "rolling_std_1h": 8.5},
            metadata={"value": 85.2, "baseline": 45.0, "z_score": 3.2},
        )
        
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            store.store(event, ttl=86400)
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
    
    def benchmark_anomaly_memory_store_retrieve(self) -> Dict[str, float]:
        """Benchmark AnomalyMemoryStore.retrieve_similar()."""
        mock_client = Mock()
        mock_client.query.get.return_value = []
        
        store = AnomalyMemoryStore(weaviate_client=mock_client)
        
        query_embedding = [0.1] * 1536
        
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            store.retrieve_similar(
                query_embedding=query_embedding,
                sensor_id=12345,
                regime="STARTUP",
                top_k=5,
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
    
    def benchmark_operational_memory_pipeline(self) -> Dict[str, float]:
        """Benchmark OperationalMemoryPipeline.process_event()."""
        event_builder = SemanticEventBuilder()
        mock_store = Mock()
        registry = CognitiveMemoryRegistry()
        
        pipeline = OperationalMemoryPipeline(
            event_builder=event_builder,
            memory_store=mock_store,
            registry=registry,
        )
        
        ml_features = {
            "timestamp": time.time(),
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "trend": "increasing",
            "stability": 0.5,
            "model_version": "2.0.0",
            "dynamic_features": {
                "derivative": 2.5,
                "rolling_std_1h": 8.5,
            },
        }
        
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            pipeline.process_event(
                sensor_id=12345,
                sensor_type="TEMPERATURE",
                ml_features=ml_features,
                regime="STARTUP",
                anomaly_score=0.85,
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
    
    def benchmark_historical_similarity_retriever(self) -> Dict[str, float]:
        """Benchmark HistoricalSimilarityRetriever.retrieve()."""
        mock_store = Mock()
        mock_store.retrieve_similar.return_value = []
        
        retriever = HistoricalSimilarityRetriever(
            memory_store=mock_store,
            min_similarity_threshold=0.7,
        )
        
        ml_features = {
            "current_value": 85.2,
            "baseline": 45.0,
            "z_score": 3.2,
            "dynamic_features": {
                "derivative": 2.5,
                "rolling_std_1h": 8.5,
            },
        }
        
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            retriever.retrieve(
                sensor_id=12345,
                ml_features=ml_features,
                regime="STARTUP",
                top_k=5,
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
    
    def run_all_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Run all benchmarks and return results."""
        results = {}
        
        print("Running SemanticEventBuilder benchmark...")
        results["semantic_event_builder"] = self.benchmark_semantic_event_builder()
        
        print("Running AnomalyMemoryStore.store benchmark...")
        results["anomaly_memory_store_store"] = self.benchmark_anomaly_memory_store_store()
        
        print("Running AnomalyMemoryStore.retrieve benchmark...")
        results["anomaly_memory_store_retrieve"] = self.benchmark_anomaly_memory_store_retrieve()
        
        print("Running OperationalMemoryPipeline benchmark...")
        results["operational_memory_pipeline"] = self.benchmark_operational_memory_pipeline()
        
        print("Running HistoricalSimilarityRetriever benchmark...")
        results["historical_similarity_retriever"] = self.benchmark_historical_similarity_retriever()
        
        return results
    
    def print_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Print benchmark results."""
        print("\n" + "=" * 80)
        print("LATENCY BENCHMARK RESULTS")
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
        print("TARGET: Latency adicional <100ms")
        print("=" * 80)


if __name__ == "__main__":
    benchmark = LatencyBenchmark(iterations=100)
    results = benchmark.run_all_benchmarks()
    benchmark.print_results(results)
