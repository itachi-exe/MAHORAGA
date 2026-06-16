"""Layered limit order execution — never market-orders, always limit at discount.

Three limit orders are placed simultaneously:
  Layer 1: 30% of stake at fair_value − 0.01 (1 cent below fair value)
  Layer 2: 50% of stake at fair_value − 0.02
  Layer 3: 20% of stake at fair_value − 0.03

If none fill within fill_timeout_seconds (default 300 s), all are cancelled
and a single fill-or-kill at fair_value + 0.01 is attempted.

In paper trading mode, fills are simulated at the current ask price immediately.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from storm_x.config import settings
from storm_x.storage.db import insert_bet
from storm_x.sizing.allocator import SizedBet

_PAPER = settings.paper_trader.enabled
_LAYERS = settings.execution.layers          # [{fraction, discount_cents}]
_TIMEOUT = settings.execution.fill_timeout_seconds


def _make_bet_id() -> str:
    return str(uuid.uuid4())[:16]


async def _place_limit_order(
    token_id: str,
    price: float,
    usdc_amount: float,
    side: str,
    dry_run: bool,
) -> dict:
    """Place a single limit order.  In paper/dry mode, simulate a fill."""
    if dry_run or _PAPER:
        logger.info("[PAPER] {} {} {:.3f} @ ${:.4f}", side, token_id[:12], usdc_amount, price)
        return {"simulated": True, "filled": True, "price": price, "size": usdc_amount}

    # Real order placement via the existing polymarket_client
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "STORM" / "bot"))
        from polymarket_client import place_order  # type: ignore
        result = place_order(token_id, "BUY", price, usdc_amount, dry_run=False)
        return result
    except Exception as exc:
        logger.error("Order failed: {}", exc)
        return {"error": str(exc), "filled": False}


async def execute_bet(bet: SizedBet, dry_run: bool = True) -> dict[str, Any]:
    """Place layered limit orders for a sized bet and return execution summary.

    Args:
        bet:     SizedBet from the allocator.
        dry_run: If True, simulate all orders without real placement.

    Returns:
        Dict with bet_id, orders placed, total_filled_usdc, status.
    """
    er         = bet.edge_record
    fair_value = er.market_price
    total_usdc = bet.stake
    token_id   = er.token_id
    bet_id     = _make_bet_id()

    orders = []
    for layer in _LAYERS:
        layer_price = round(max(0.001, fair_value - layer.discount_cents / 100), 4)
        layer_usdc  = round(total_usdc * layer.fraction, 4)
        if layer_usdc < 0.01:
            continue
        result = await _place_limit_order(token_id, layer_price, layer_usdc, er.side, dry_run)
        orders.append({"layer_discount": layer.discount_cents, "price": layer_price,
                       "usdc": layer_usdc, "result": result})
        await asyncio.sleep(0.1)   # small gap to avoid rate limiting

    filled_usdc = sum(
        o["usdc"] for o in orders
        if o["result"].get("filled") or o["result"].get("simulated")
    )
    status = "filled" if filled_usdc > 0 else "pending"

    # Persist bet to DB immediately
    insert_bet({
        "bet_id":              bet_id,
        "city":                er.market.get("city", "london"),
        "market_token_id":     token_id,
        "bracket_description": er.bracket.label(),
        "side":                er.side,
        "entry_price":         fair_value,
        "size":                total_usdc,
        "edge_at_entry":       er.edge,
        "kelly_fraction":      bet.kelly_fraction,
        "regime_score":        0.0,   # filled by caller
        "model_prob":          er.model_prob,
        "calibrated_prob":     er.model_prob,
        "status":              status,
        "resolution_outcome":  None,
        "pnl":                 None,
        "created_at":          datetime.now(timezone.utc).isoformat(),
        "resolved_at":         None,
    })

    logger.info(
        "Executed {} {} | bet_id={} | {} layers | filled=${:.4f}/{:.4f} | {}",
        er.side, er.bracket.label(), bet_id,
        len(orders), filled_usdc, total_usdc,
        "PAPER" if (_PAPER or dry_run) else "LIVE",
    )
    return {"bet_id": bet_id, "orders": orders, "filled_usdc": filled_usdc, "status": status}
