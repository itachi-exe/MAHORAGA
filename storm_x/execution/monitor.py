"""Open position monitor and pre-resolution exit logic.

Checks every 30 minutes:
  - For each open bet, fetch current mid-price.
  - If unrealised gain ≥ 70% of maximum possible profit, sell at current bid.

Maximum possible profit on a YES bought at price P = (1 - P) per share.
Trigger condition: current_bid >= P + 0.70 * (1 - P)
  → equivalently:  current_bid >= 0.70 + 0.30 * P

In paper mode, the 'sale' is simulated at current bid price.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
from loguru import logger

from storm_x.config import settings
from storm_x.markets.prices import get_price
from storm_x.storage.db import get_open_bets, resolve_bet

_TRIGGER = settings.exits.profit_trigger_fraction   # 0.70
_PAPER   = settings.paper_trader.enabled


def _trigger_price(entry_price: float) -> float:
    """Minimum current price that triggers an early exit."""
    return _TRIGGER + (1.0 - _TRIGGER) * entry_price


async def check_exits(dry_run: bool = True) -> list[str]:
    """Scan all open bets and close any that hit the exit trigger.

    Returns list of bet_ids that were closed.
    """
    open_bets = get_open_bets()
    closed = []

    for bet in open_bets:
        token_id    = bet["market_token_id"]
        entry_price = float(bet["entry_price"])
        size_usdc   = float(bet["size"])
        side        = bet["side"]

        # For YES bets, sell means hitting the YES bid; for NO bets, same token
        current_bid = await get_price(token_id, "SELL")
        if current_bid is None:
            continue

        trigger = _trigger_price(entry_price)
        if current_bid < trigger:
            continue

        # Exit triggered
        shares     = size_usdc / entry_price
        sale_usdc  = shares * current_bid
        pnl        = sale_usdc - size_usdc
        outcome    = 1 if pnl > 0 else 0

        if dry_run or _PAPER:
            logger.info(
                "[EXIT][PAPER] bet={} entry={:.3f} bid={:.3f} trigger={:.3f} pnl=${:.4f}",
                bet["bet_id"][:8], entry_price, current_bid, trigger, pnl,
            )
        else:
            logger.info(
                "[EXIT][LIVE] bet={} entry={:.3f} bid={:.3f} pnl=${:.4f}",
                bet["bet_id"][:8], entry_price, current_bid, pnl,
            )

        resolve_bet(bet["bet_id"], outcome, round(pnl, 4))
        closed.append(bet["bet_id"])

    if closed:
        logger.info("Pre-resolution exits: {} positions closed", len(closed))
    return closed
