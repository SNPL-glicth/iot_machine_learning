"""Independent memory layer for Zephyr Bot.

Tracks non-ML contextual state such as consecutive losses, daily PnL thresholds,
and cool-down periods. Acts as a safety layer before ML signals are executed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BotContextState:
    """State tracking for the current trading session."""
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    last_trade_time: float = 0.0
    is_in_cooldown: bool = False
    cooldown_until: float = 0.0
    total_trades_today: int = 0


class ContextualMemoryManager:
    """Manages independent bot state and provides rule-based vetos."""
    
    def __init__(self, max_consecutive_losses: int = 3, max_daily_loss: float = 50.0, cooldown_seconds: float = 900.0):
        self.state = BotContextState()
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss = abs(max_daily_loss)
        self.cooldown_seconds = cooldown_seconds

    def update_from_execution(self, pnl: float) -> None:
        """Update internal state based on the outcome of a closed trade."""
        self.state.last_trade_time = time.time()
        self.state.total_trades_today += 1
        self.state.daily_pnl += pnl
        
        if pnl < 0:
            self.state.consecutive_losses += 1
            logger.warning(
                f"[MEMORY] Loss registered. Consecutive losses: {self.state.consecutive_losses}"
            )
        else:
            self.state.consecutive_losses = 0
            logger.info("[MEMORY] Profitable trade. Consecutive loss counter reset.")
            
        self._evaluate_cooldown()

    def _evaluate_cooldown(self) -> None:
        """Determines if limits are hit and triggers cooldowns."""
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            logger.error(
                f"[MEMORY] Hit max consecutive losses ({self.max_consecutive_losses}). Initiating cooldown."
            )
            self._trigger_cooldown()
            return
            
        if self.state.daily_pnl <= -self.max_daily_loss:
            logger.error(
                f"[MEMORY] Daily loss limit exceeded (${self.state.daily_pnl:.2f}). Initiating deep cooldown."
            )
            self._trigger_cooldown(multiplier=4) # 4x cooldown for max daily loss

    def _trigger_cooldown(self, multiplier: float = 1.0) -> None:
        self.state.is_in_cooldown = True
        self.state.cooldown_until = time.time() + (self.cooldown_seconds * multiplier)

    def can_execute(self) -> tuple[bool, Optional[str]]:
        """Checks if the bot is allowed to execute trades based on non-ML context.
        
        Returns:
            (can_execute, reason_if_blocked)
        """
        if self.state.is_in_cooldown:
            if time.time() < self.state.cooldown_until:
                remaining = self.state.cooldown_until - time.time()
                return False, f"Memory VETO: In cooldown for {remaining:.1f} more seconds."
            else:
                logger.info("[MEMORY] Cooldown period expired. Resuming operations.")
                self.state.is_in_cooldown = False
                self.state.consecutive_losses = 0 # reset on cooldown expiry
                
        return True, None
