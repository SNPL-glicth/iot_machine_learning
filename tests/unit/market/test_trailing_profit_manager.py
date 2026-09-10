"""Unit tests for TrailingProfitManager."""


from iot_machine_learning.infrastructure.adapters.market.trailing_profit_manager import (
    TrailingProfitConfig,
    TrailingProfitManager,
)


def test_no_exit_below_activation():
    """No debe disparar salida si la ganancia no superó el umbral de activación."""
    config = TrailingProfitConfig(activation_pnl_usd=2.0)
    manager = TrailingProfitManager(config)

    # Ganancia sube a 1.20 y luego baja a 0.50
    exit1, _ = manager.update(1.20)
    assert not exit1
    assert manager.peak_pnl == 1.20
    assert not manager.is_active

    exit2, _ = manager.update(0.50)
    assert not exit2


def test_exit_triggered_after_peak_pullback():
    """Debe disparar salida cuando retrocede más del buffer desde el pico."""
    config = TrailingProfitConfig(
        activation_pnl_usd=2.0,
        giveback_ratio=0.25,
        min_giveback_usd=0.75,
        max_giveback_usd=2.0,
    )
    manager = TrailingProfitManager(config)

    # Sube progresivamente a 4.00
    manager.update(2.50)
    assert manager.is_active
    manager.update(4.00)
    assert manager.peak_pnl == 4.00

    # 25% de 4.00 es 1.00. Umbral de salida = 4.00 - 1.00 = 3.00
    # Si baja a 3.50 -> No sale
    exit1, _ = manager.update(3.50)
    assert not exit1

    # Si baja a 2.90 -> Sale y bloquea ganancia
    exit2, reason = manager.update(2.90)
    assert exit2
    assert "Peak was $4.00" in reason
    assert "current is $2.90" in reason


def test_infinite_ceiling_ratchet():
    """Debe permitir que el pico crezca indefinidamente si el mercado sigue a favor."""
    config = TrailingProfitConfig(
        activation_pnl_usd=2.0,
        giveback_ratio=0.25,
        min_giveback_usd=0.75,
        max_giveback_usd=2.0,
    )
    manager = TrailingProfitManager(config)

    for pnl in [2.0, 5.0, 8.0, 15.0, 25.0]:
        exit_flag, _ = manager.update(pnl)
        assert not exit_flag
        assert manager.peak_pnl == pnl

    # A 25.00, giveback está capeado en max_giveback_usd (2.00).
    # Umbral de salida = 25.00 - 2.00 = 23.00
    exit_retract, reason = manager.update(22.50)
    assert exit_retract
    assert "Peak was $25.00" in reason


def test_reset_clears_state():
    """Reset debe limpiar el pico y estado activo."""
    manager = TrailingProfitManager()
    manager.update(5.0)
    assert manager.is_active
    assert manager.peak_pnl == 5.0

    manager.reset()
    assert not manager.is_active
    assert manager.peak_pnl == 0.0


def test_can_execute_market_session_gating():
    """Verifica que can_execute bloquee nuevas órdenes si el mercado va a cerrar o si ya hay posición activa."""
    from types import SimpleNamespace

    from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
    from iot_machine_learning.infrastructure.adapters.market.live_runner_execution import (
        can_execute,
    )
    from iot_machine_learning.infrastructure.adapters.market.live_runner_models import LiveBotState

    cfg = LiveBotConfig(symbol="SPY")
    plan = SimpleNamespace(action="EXECUTE")

    # Caso normal: permitido
    state_ok = LiveBotState(last_phi_moe=0.8, last_lambda_t=0.1, current_position=0.0)
    assert can_execute(plan, cfg, state_ok, 758.0) is True

    # Bloqueo por posición existente
    state_in_pos = LiveBotState(last_phi_moe=0.8, last_lambda_t=0.1, current_position=7.0)
    assert can_execute(plan, cfg, state_in_pos, 758.0) is False

    # Bloqueo por mercado cerrando pronto
    state_closing = LiveBotState(last_phi_moe=0.8, last_lambda_t=0.1, current_position=0.0, market_closing_soon=True)
    assert can_execute(plan, cfg, state_closing, 758.0) is False

    # Bloqueo por mercado cerrado
    state_closed = LiveBotState(last_phi_moe=0.8, last_lambda_t=0.1, current_position=0.0, market_closed=True)
    assert can_execute(plan, cfg, state_closed, 758.0) is False
