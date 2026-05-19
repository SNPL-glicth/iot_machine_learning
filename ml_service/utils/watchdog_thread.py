"""Watchdog para daemon threads críticos.

FIX P1-3: Si un daemon thread muere (excepción no capturada, timeout,
OOM), el watchdog lo reinicia automáticamente con backoff.
"""

from __future__ import annotations

import logging
import random
import threading
import time
import traceback
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class WatchdogThread:
    """Wrapper que mantiene vivo un daemon thread.

    Si el thread muere, lo reinicia automáticamente con backoff.
    Después de ``max_restarts`` consecutivos sin éxito, deja de reintentar.
    """

    def __init__(
        self,
        target: Callable[[], None],
        name: str,
        max_restarts: int = 10,
        backoff_seconds: float = 5.0,
        healthy_threshold_seconds: float = 60.0,
    ) -> None:
        self._target = target
        self._name = name
        self._max_restarts = max_restarts
        self._backoff_seconds = backoff_seconds
        self._healthy_threshold = healthy_threshold_seconds

        self._watchdog_thread: Optional[threading.Thread] = None
        self._inner_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._restart_count = 0
        self._last_start_time: Optional[float] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Inicia el watchdog (daemon) que supervisa el thread interno."""
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            logger.warning("watchdog_already_running", extra={"name": self._name})
            return
        self._shutdown = False
        self._restart_count = 0
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name=f"watchdog-{self._name}",
            daemon=True,
        )
        self._watchdog_thread.start()
        logger.info("watchdog_started", extra={"name": self._name, "max_restarts": self._max_restarts})

    def stop(self) -> None:
        """Señaliza shutdown graceful. El thread interno debe respetar el flag."""
        self._shutdown = True
        logger.info("watchdog_stop_signal", extra={"name": self._name})

    def is_healthy(self) -> bool:
        """Retorna False si el thread está muerto y esperando restart (o superó max_restarts)."""
        with self._lock:
            if self._shutdown:
                return True  # Shutdown intencional no es unhealth
            if self._restart_count >= self._max_restarts:
                return False
            if self._inner_thread is None:
                return False
            return self._inner_thread.is_alive()

    def _watchdog_loop(self) -> None:
        """Loop del supervisor: arranca, vigila, reinicia."""
        while not self._shutdown:
            self._start_inner()

            # Monitorear hasta que muera o se ordene shutdown
            while not self._shutdown:
                alive = self._inner_thread is not None and self._inner_thread.is_alive()
                if alive:
                    time.sleep(1.0)
                    continue
                # El thread murió
                break

            if self._shutdown:
                break

            # Determinar si fue un fallo circunstancial (resetea contador)
            now = time.monotonic()
            if self._last_start_time and (now - self._last_start_time) >= self._healthy_threshold:
                self._restart_count = 0
                logger.info(
                    "watchdog_healthy_reset",
                    extra={"name": self._name, "uptime_seconds": round(now - self._last_start_time, 1)},
                )

            self._restart_count += 1
            if self._restart_count > self._max_restarts:
                logger.critical(
                    "watchdog_max_restarts_exceeded",
                    extra={"name": self._name, "max_restarts": self._max_restarts},
                )
                break

            # Backoff con jitter ±20%
            jitter = self._backoff_seconds * (0.8 + random.random() * 0.4)
            logger.error(
                "watchdog_restart_scheduled",
                extra={
                    "name": self._name,
                    "restart_count": self._restart_count,
                    "backoff_seconds": round(jitter, 2),
                },
            )
            time.sleep(jitter)

        logger.info("watchdog_stopped", extra={"name": self._name})

    def _start_inner(self) -> None:
        """Arranca el thread interno real y registra la hora."""
        with self._lock:
            self._inner_thread = threading.Thread(
                target=self._run_wrapped,
                name=self._name,
                daemon=True,
            )
            self._last_start_time = time.monotonic()
            self._inner_thread.start()
        logger.info("watchdog_inner_started", extra={"name": self._name})

    def _run_wrapped(self) -> None:
        """Wrapper que captura cualquier excepción no manejada del target."""
        try:
            self._target()
        except Exception:
            logger.exception(
                "watchdog_inner_crashed",
                extra={"name": self._name, "traceback": traceback.format_exc()},
            )
