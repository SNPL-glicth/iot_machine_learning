"""Tests de regresión para SlidingWindowCache canónico."""

from __future__ import annotations

import pytest
import threading
import time
from infrastructure.persistence.sliding_window import SlidingWindowCache


class TestSlidingWindowCache:
    """Tests básicos de SlidingWindowCache."""
    
    def test_append_and_get(self):
        """Agregar puntos y recuperarlos."""
        cache = SlidingWindowCache(window_size=5)
        cache.append("s1", 1.0, 1000.0)
        cache.append("s1", 2.0, 1001.0)
        result = cache.get("s1")
        assert result is not None
        assert len(result) == 2
        assert result[0] == (1000.0, 1.0)
        assert result[1] == (1001.0, 2.0)
    
    def test_window_size_limit(self):
        """Ventana no supera window_size."""
        cache = SlidingWindowCache(window_size=3)
        for i in range(10):
            cache.append("s1", float(i), float(i))
        result = cache.get_values("s1")
        assert len(result) == 3
        assert result == [7.0, 8.0, 9.0]  # Los últimos 3
    
    def test_lru_eviction(self):
        """LRU elimina la serie menos usada."""
        cache = SlidingWindowCache(window_size=5, max_series=2)
        cache.append("s1", 1.0, 1.0)
        cache.append("s2", 2.0, 2.0)
        cache.append("s3", 3.0, 3.0)  # Debe evictar s1
        assert cache.get("s1") is None
        assert cache.get("s2") is not None
        assert cache.get("s3") is not None
    
    def test_ttl_expiry(self):
        """TTL expira entradas antiguas."""
        cache = SlidingWindowCache(window_size=5, ttl_seconds=1)
        cache.append("s1", 1.0, 1.0)
        time.sleep(1.1)
        cache.append("s2", 2.0, 2.0)  # Trigger eviction
        assert cache.get("s1") is None
    
    def test_thread_safety(self):
        """Escrituras concurrentes no corrompen el estado."""
        cache = SlidingWindowCache(window_size=100)
        errors = []
        
        def writer(series_id: str):
            try:
                for i in range(50):
                    cache.append(series_id, float(i), float(i))
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=writer, args=(f"s{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Errores de concurrencia: {errors}"
    
    def test_get_nonexistent(self):
        """Serie que no existe retorna None."""
        cache = SlidingWindowCache(window_size=5)
        assert cache.get("nonexistent") is None
        assert cache.get_values("nonexistent") is None
        assert cache.size("nonexistent") == 0
    
    def test_clear(self):
        """Clear elimina la ventana."""
        cache = SlidingWindowCache(window_size=5)
        cache.append("s1", 1.0, 1.0)
        cache.append("s1", 2.0, 2.0)
        assert cache.size("s1") == 2
        cache.clear("s1")
        assert cache.get("s1") is None
    
    def test_series_ids(self):
        """series_ids retorna lista de series activas."""
        cache = SlidingWindowCache(window_size=5)
        cache.append("s1", 1.0, 1.0)
        cache.append("s2", 2.0, 2.0)
        ids = cache.series_ids()
        assert set(ids) == {"s1", "s2"}
    
    def test_get_metrics(self):
        """get_metrics retorna métricas correctas."""
        cache = SlidingWindowCache(window_size=10, max_series=100, ttl_seconds=300)
        cache.append("s1", 1.0, 1.0)
        metrics = cache.get_metrics()
        assert metrics["active_series"] == 1
        assert metrics["max_series"] == 100
        assert metrics["window_size"] == 10
        assert metrics["ttl_seconds"] == 300
    
    def test_no_ttl(self):
        """TTL None significa sin expiración."""
        cache = SlidingWindowCache(window_size=5, ttl_seconds=None)
        cache.append("s1", 1.0, 1.0)
        time.sleep(0.1)
        cache.append("s2", 2.0, 2.0)
        assert cache.get("s1") is not None  # No debe expirar


class TestLegacyAdapters:
    """Tests de compatibilidad con adapters legacy."""
    
    def test_sliding_window_store_adapter(self):
        """SlidingWindowStore adapter funciona."""
        from infrastructure.adapters.legacy.sliding_window_store_adapter import (
            SlidingWindowStore,
        )
        from domain.entities.iot.sensor_reading import Reading
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            store = SlidingWindowStore(max_size=5)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
        
        # Test basic functionality
        reading = Reading(series_id="s1", value=1.0, timestamp=1000.0)
        store.append(reading)
        window = store.get_window("s1")
        assert len(window) == 1
        assert window[0].value == 1.0
    
    def test_sliding_window_buffer_adapter(self):
        """SlidingWindowBuffer adapter funciona."""
        from infrastructure.adapters.legacy.sliding_window_buffer_adapter import (
            SlidingWindowBuffer,
        )
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            buf = SlidingWindowBuffer(max_horizon_seconds=10.0)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
        
        # Test basic functionality
        stats = buf.add_reading(sensor_id=1, value=10.0, timestamp=1000.0, windows=[1.0])
        assert "w1" in stats
    
    def test_in_memory_sliding_window_adapter(self):
        """InMemorySlidingWindowStore adapter funciona."""
        from infrastructure.adapters.legacy.in_memory_sliding_window_adapter import (
            InMemorySlidingWindowStore,
        )
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            store = InMemorySlidingWindowStore()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
        
        # Test basic functionality
        store.append(sensor_id=1, item="test", timestamp=1000.0)
        window = store.get_window(sensor_id=1)
        assert len(window) == 1
        assert window[0] == "test"
