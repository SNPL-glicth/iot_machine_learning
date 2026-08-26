"""Tests para OrderBookL2 — sincronización snapshot + deltas."""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.adapters.market.binance.order_book_state import OrderBookL2, OrderBookMetrics


class TestOrderBookL2:
    """Tests de sincronización y aplicación de deltas."""
    
    @pytest.fixture
    def order_book(self):
        return OrderBookL2("BTCUSDT", max_levels=20)
    
    @pytest.fixture
    def sample_snapshot(self):
        """Snapshot típico de Binance REST."""
        return {
            "lastUpdateId": 1000000,
            "bids": [
                ["50000.00", "1.5"],
                ["49999.00", "2.0"],
                ["49998.00", "1.0"],
            ],
            "asks": [
                ["50001.00", "1.2"],
                ["50002.00", "1.8"],
                ["50003.00", "0.5"],
            ],
        }
    
    def test_snapshot_initialization(self, order_book, sample_snapshot):
        """Snapshot inicializa correctamente bids y asks."""
        order_book.start_snapshot_sync()
        success = order_book.apply_snapshot(sample_snapshot)
        
        assert success is True
        assert order_book.is_initialized is True
        assert order_book.last_update_id == 1000000
        assert order_book.best_bid == 50000.0
        assert order_book.best_ask == 50001.0
        assert order_book.mid_price == 50000.5
        assert order_book.spread == 1.0
    
    def test_bid_ask_ordering(self, order_book, sample_snapshot):
        """Bids ordenados descendente, asks ascendente."""
        order_book.start_snapshot_sync()
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [["50000.00", "1.0"], ["49999.00", "2.0"], ["49998.00", "1.0"]],
            "asks": [["50001.00", "1.0"], ["50002.00", "2.0"], ["50003.00", "1.0"]],
        })
        
        # Bids descendente (mayor primero)
        assert order_book._bid_prices == [50000.0, 49999.0, 49998.0]
        # Asks ascendente (menor primero)
        assert order_book._ask_prices == [50001.0, 50002.0, 50003.0]
    
    def test_delta_updates_bid(self, order_book):
        """Delta actualiza bid existente."""
        order_book.start_snapshot_sync()
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        })
        
        # Delta que actualiza cantidad
        delta = {
            "e": "depthUpdate",
            "E": 123456789,
            "s": "BTCUSDT",
            "U": 1000001,
            "u": 1000005,
            "b": [["50000.00", "2.5"]],  # Nueva cantidad
            "a": [],
        }
        
        success = order_book.apply_delta(delta)
        assert success is True
        assert order_book.last_update_id == 1000005
        assert order_book._bids[50000.0] == 2.5
    
    def test_delta_removes_level(self, order_book):
        """Delta con qty=0 elimina nivel."""
        order_book.start_snapshot_sync()
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [["50000.00", "1.0"], ["49999.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        })
        
        # Eliminar bid 50000
        delta = {
            "e": "depthUpdate",
            "U": 1000001,
            "u": 1000002,
            "b": [["50000.00", "0.00"]],
            "a": [],
        }
        
        success = order_book.apply_delta(delta)
        assert success is True
        assert 50000.0 not in order_book._bids
        assert order_book.best_bid == 49999.0
    
    def test_delta_new_level(self, order_book):
        """Delta inserta nuevo nivel."""
        order_book.start_snapshot_sync()
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        })
        
        # Nuevo bid más alto
        delta = {
            "U": 1000001,
            "u": 1000002,
            "b": [["50002.00", "3.0"]],
            "a": [],
        }
        
        success = order_book.apply_delta(delta)
        assert success is True
        assert order_book.best_bid == 50002.0
        assert order_book._bids[50002.0] == 3.0
        # Orden descendente mantenido
        assert order_book._bid_prices == [50002.0, 50000.0]
    
    def test_delta_obsolete_discarded(self, order_book):
        """Delta con u < lastUpdateId+1 se descarta."""
        order_book.start_snapshot_sync()
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        })
        
        # Delta obsoleto (u = 1000000, last_update_id = 1000000, necesita >= 1000001)
        delta = {
            "U": 1000001,
            "u": 1000000,  # Obsoleto
            "b": [["50000.00", "999.0"]],
            "a": [],
        }
        
        success = order_book.apply_delta(delta)
        # Debe retornar True (no error) pero no aplicar
        assert success is True
        assert order_book._bids[50000.0] == 1.0  # Sin cambios
    
    def test_gap_detection(self, order_book):
        """Gap en U > lastUpdateId+1 detectado."""
        order_book.start_snapshot_sync()
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        })
        
        # Gap: U = 1000005, pero lastUpdateId = 1000000, se esperaba U <= 1000001
        delta = {
            "U": 1000005,  # Gap!
            "u": 1000010,
            "b": [["50000.00", "2.0"]],
            "a": [],
        }
        
        success = order_book.apply_delta(delta)
        assert success is False  # Gap detectado
    
    def test_metrics_computation(self, order_book):
        """Métricas se calculan correctamente."""
        order_book.start_snapshot_sync()
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [["50000.00", "1.5"], ["49999.00", "1.0"]],
            "asks": [["50001.00", "1.2"], ["50002.00", "0.8"]],
        })
        
        metrics = order_book.metrics
        assert metrics is not None
        assert metrics.mid_price == 50000.5
        assert metrics.spread == 1.0
        assert metrics.spread_bps > 0
        assert metrics.bid_volume == 2.5  # 1.5 + 1.0
        assert metrics.ask_volume == 2.0  # 1.2 + 0.8
        assert -1.0 <= metrics.volume_imbalance <= 1.0
        assert metrics.microprice is not None
    
    def test_buffer_during_snapshot(self, order_book):
        """Deltas durante snapshot se bufferizan y aplican después."""
        order_book.start_snapshot_sync()
        
        # Buffer delta que llega durante snapshot
        delta_during = {
            "U": 1000001,
            "u": 1000002,
            "b": [["50000.00", "2.0"]],  # Actualización durante snapshot
            "a": [],
        }
        order_book.apply_delta(delta_during)  # Se bufferiza
        
        # Aplicar snapshot
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        })
        
        # El delta bufferizado (u=1000002 >= 1000000+1) debe aplicarse
        assert order_book._bids[50000.0] == 2.0
    
    def test_delta_obsolete_during_snapshot_discarded(self, order_book):
        """Delta obsoleto durante snapshot se descarta."""
        order_book.start_snapshot_sync()
        
        # Delta obsoleto (u=1000000, snapshot será lastUpdateId=1000000, min_u=1000001)
        delta_obsolete = {
            "U": 1000000,
            "u": 1000000,
            "b": [["50000.00", "999.0"]],
            "a": [],
        }
        order_book.apply_delta(delta_obsolete)
        
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        })
        
        # Delta obsoleto debe descartarse
        assert order_book._bids[50000.0] == 1.0
    
    def test_max_levels_truncation(self, order_book):
        """Truncación a max_levels funciona."""
        order_book = OrderBookL2("TEST", max_levels=2)
        
        order_book.start_snapshot_sync()
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [
                ["50000.00", "1.0"],
                ["49999.00", "1.0"],
                ["49998.00", "1.0"],  # Debe truncarse
            ],
            "asks": [["50001.00", "1.0"], ["50002.00", "1.0"]],
        })
        
        assert len(order_book._bid_prices) == 2
        assert order_book._bid_prices == [50000.0, 49999.0]
        assert 49998.0 not in order_book._bids
    
    def test_get_top_levels(self, order_book, sample_snapshot):
        """get_top_levels retorna formato correcto."""
        order_book.start_snapshot_sync()
        order_book.apply_snapshot(sample_snapshot)
        
        bids, asks = order_book.get_top_levels(2)
        
        assert len(bids) == 2
        assert len(asks) == 2
        assert all(isinstance(b, tuple) and len(b) == 2 for b in bids)
        assert all(isinstance(a, tuple) and len(a) == 2 for a in asks)
    
    def test_state_snapshot_roundtrip(self, order_book, sample_snapshot):
        """Serialización y deserialización de estado."""
        order_book.start_snapshot_sync()
        order_book.apply_snapshot(sample_snapshot)
        
        # Serializar
        state = order_book.get_state_snapshot()
        
        # Crear nuevo libro y cargar
        new_book = OrderBookL2("BTCUSDT", max_levels=20)
        success = new_book.load_state_snapshot(state)
        
        assert success is True
        assert new_book.last_update_id == 1000000
        assert new_book.best_bid == 50000.0
        assert new_book.best_ask == 50001.0
        assert new_book._bids == order_book._bids
    
    def test_volume_at_price(self, order_book, sample_snapshot):
        """Volumen acumulado alrededor de precio."""
        order_book.start_snapshot_sync()
        order_book.apply_snapshot(sample_snapshot)
        
        # Volumen en best bid ±1 nivel
        vol = order_book.volume_at_price(50000.0, "bid", levels=1)
        assert vol == 1.5 + 2.0  # 50000 + 49999
        
        # Precio inexistente
        vol = order_book.volume_at_price(49990.0, "bid", levels=1)
        assert vol == 0.0
    
    def test_needs_resync(self, order_book):
        """Detección de necesidad de resync."""
        assert order_book.needs_resync() is False  # No inicializado
        
        order_book.start_snapshot_sync()
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        })
        
        # Inmediatamente después, no necesita resync
        assert order_book.needs_resync() is False
    
    def test_metrics_volume_imbalance(self, order_book):
        """Volume imbalance en rango [-1, 1]."""
        order_book.start_snapshot_sync()
        
        # Más bids que asks
        order_book.apply_snapshot({
            "lastUpdateId": 1000000,
            "bids": [["50000.00", "10.0"]],
            "asks": [["50001.00", "1.0"]],
        })
        
        metrics = order_book.metrics
        assert metrics.volume_imbalance > 0.5  # Mayoritariamente bids
        assert -1.0 <= metrics.volume_imbalance <= 1.0
        
        # Más asks que bids
        order_book.apply_snapshot({
            "lastUpdateId": 1000001,
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "10.0"]],
        })
        
        metrics = order_book.metrics
        assert metrics.volume_imbalance < -0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])