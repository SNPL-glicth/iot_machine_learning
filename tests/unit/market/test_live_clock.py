"""Tests del LiveClock (FASE 6).

Verifica que LiveClock cumple el mismo contrato que ReplayClock:
- monotonía en advance_to
- validación de timestamps finitos
- Clock.current() retorna tiempo actual
"""

from __future__ import annotations

import time

import pytest

from iot_machine_learning.domain.entities.market.replay import (
    Clock,
    ClockRollbackError,
    LiveClock,
    ReplayClock,
)


class TestLiveClockContract:
    def test_current_returns_finite_now(self) -> None:
        """LiveClock.current() debe retornar un timestamp finito."""
        clock = LiveClock.current()
        assert clock.now > 0
        assert clock.now < time.time() + 1.0  # razonablemente actual

    def test_advance_to_must_be_monotonic(self) -> None:
        """LiveClock no puede retroceder (igual que ReplayClock)."""
        clock = LiveClock.current()
        past = clock.now - 1.0
        with pytest.raises(ClockRollbackError):
            clock.advance_to(past)

    def test_advance_to_accepts_future(self) -> None:
        """LiveClock puede avanzar al futuro."""
        clock = LiveClock.current()
        future = clock.now + 1.0
        new_clock = clock.advance_to(future)
        assert new_clock.now == future
        assert new_clock.now > clock.now

    def test_rejects_infinite_now(self) -> None:
        """LiveClock rechaza timestamps infinitos (igual que ReplayClock)."""
        with pytest.raises(ValueError):
            LiveClock(now=float("inf"))
        with pytest.raises(ValueError):
            LiveClock(now=float("-inf"))
        with pytest.raises(ValueError):
            LiveClock(now=float("nan"))

    def test_frozen_immutable(self) -> None:
        """LiveClock es inmutable (frozen): advance_to retorna nueva instancia."""
        clock = LiveClock.current()
        new_clock = clock.advance_to(clock.now + 1.0)
        assert clock.now != new_clock.now
        assert clock.now < new_clock.now


class TestClockProtocol:
    def test_replay_clock_implements_protocol(self) -> None:
        """ReplayClock implementa el protocol Clock."""
        clock: Clock = ReplayClock(now=100.0)
        assert clock.now == 100.0
        new_clock = clock.advance_to(200.0)
        assert new_clock.now == 200.0

    def test_live_clock_implements_protocol(self) -> None:
        """LiveClock implementa el protocol Clock."""
        clock: Clock = LiveClock.current()
        assert clock.now > 0
        new_clock = clock.advance_to(clock.now + 1.0)
        assert new_clock.now > clock.now

    def test_clock_polymorphism(self) -> None:
        """Engine puede usar cualquier implementación de Clock."""
        replay: Clock = ReplayClock(now=100.0)
        live: Clock = LiveClock.current()

        # Ambos pueden avanzar
        replay_advanced = replay.advance_to(200.0)
        live_advanced = live.advance_to(live.now + 1.0)

        assert replay_advanced.now == 200.0
        assert live_advanced.now > live.now
