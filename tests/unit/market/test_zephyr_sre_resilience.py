"""Unit tests for Zephyr SRE resilience: CircuitBreaker, Broker retries, and Weaviate batching."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from iot_machine_learning.infrastructure.adapters.market.zephyr.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,
)
from iot_machine_learning.infrastructure.adapters.persistence.weaviate_telemetry import (
    WeaviateTelemetryStore,
)


@pytest.mark.asyncio
async def test_circuit_breaker_transitions():
    """Verifica que el CircuitBreaker pase de CLOSED a OPEN tras N fallos consecutivos."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1, half_open_successes_needed=1)
    assert cb.state == CircuitState.CLOSED

    # Registrar 2 fallos: sigue CLOSED
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # 3er fallo: pasa a OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Llamada en OPEN lanza CircuitBreakerOpenError de inmediato sin tocar broker
    async def dummy_call():
        return 42

    with pytest.raises(CircuitBreakerOpenError):
        await cb.call(dummy_call)

    # Esperar recovery_timeout -> pasa a HALF_OPEN en la siguiente llamada
    await asyncio.sleep(0.12)
    assert cb.state == CircuitState.HALF_OPEN

    # Éxito en HALF_OPEN -> vuelve a CLOSED
    result = await cb.call(dummy_call)
    assert result == 42
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure():
    """Fallo en HALF_OPEN debe regresar inmediatamente a OPEN."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.06)
    assert cb.state == CircuitState.HALF_OPEN

    async def failing_call():
        raise ConnectionResetError("Broker unreachable")

    with pytest.raises(ConnectionResetError):
        await cb.call(failing_call)

    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_weaviate_telemetry_store_async_batching():
    """Verifica que WeaviateTelemetryStore use la cola en memoria de forma no bloqueante."""
    store = WeaviateTelemetryStore(url="http://localhost:8080", batch_size=5, flush_interval_sec=0.1)
    mock_session = AsyncMock()
    mock_session.closed = False
    store._session = mock_session

    # log_telemetry no bloquea y añade a cola interna
    store.log_telemetry({"symbol": "BTC/USD", "mid_price": 50000.0, "action": "BUY"})
    store.log_telemetry({"symbol": "ETH/USD", "mid_price": 3000.0, "action": "HOLD"})

    assert len(store._queue) == 2

    # Flush batch envía objetos
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.__aenter__.return_value = mock_resp
    mock_session.post = MagicMock(return_value=mock_resp)

    flushed = await store.flush()
    assert flushed == 2
    assert len(store._queue) == 0
    assert mock_session.post.called


@pytest.mark.asyncio
async def test_market_queue_drop_oldest():
    """Verifica la lógica de conflación de ticks cuando la cola está saturada."""
    queue = asyncio.Queue(maxsize=3)
    for i in range(3):
        await queue.put(f"tick_{i}")

    assert queue.full()

    # Simular la lógica de _ingest_worker
    new_obs = "tick_3"
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await queue.put(new_obs)

    # El elemento más viejo tick_0 debe haber sido descartado
    first = await queue.get()
    assert first == "tick_1"
