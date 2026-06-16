"""STORM-X entry point — starts FastAPI health server + APScheduler trading loop.

Usage:
    python -m storm_x.main

The health endpoint is always available at http://0.0.0.0:8003/health.
The trading loop starts immediately and runs indefinitely.
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI
from loguru import logger

from storm_x.calibration.isotonic import calibrator
from storm_x.config import settings
from storm_x.data.climatology import load_climatology
from storm_x.scheduler.runner import build_scheduler, main_cycle, intraday_update
from storm_x.storage.db import init_db, get_open_bets, get_resolved_bets


# ── Logging setup ──────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    logger.remove()
    log_cfg = settings.logging

    # Console (stderr)
    logger.add(
        sys.stderr,
        level=log_cfg.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )

    # Rotating file
    import os
    log_dir = "/var/log/storm-x"
    try:
        os.makedirs(log_dir, exist_ok=True)
        logger.add(
            f"{log_dir}/storm-x.log",
            level=log_cfg.level,
            rotation=log_cfg.rotation,
            retention=log_cfg.retention,
            compression="gz",
            enqueue=True,   # async-safe
        )
    except PermissionError:
        # Development fallback: log to local file
        logger.add(
            "storm_x/logs/storm-x.log",
            level=log_cfg.level,
            rotation=log_cfg.rotation,
            retention=log_cfg.retention,
            compression="gz",
            enqueue=True,
        )
        logger.warning("Cannot write to /var/log/storm-x — logging to storm_x/logs/storm-x.log")


# ── Startup + scheduler lifecycle ─────────────────────────────────────────────

_scheduler = build_scheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """FastAPI lifespan — init DB, climatology, calibrator, then start scheduler."""
    _setup_logging()
    logger.info("STORM-X starting up | paper_mode={} bankroll=${}", settings.paper_trader.enabled, settings.betting.bankroll)

    init_db()

    for city_key in settings.cities:
        await load_climatology(city_key)

    calibrator.refit()
    logger.info("Calibrator active={} samples={}", calibrator.is_active, calibrator._n_samples)

    _scheduler.start()
    logger.info("Scheduler started — running main cycle immediately")

    # Kick off the first cycle immediately on startup (don't wait 15 min)
    asyncio.create_task(main_cycle())

    yield

    _scheduler.shutdown(wait=True)
    logger.info("STORM-X shutdown complete")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="STORM-X", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Health endpoint — returns system state at a glance."""
    open_bets = get_open_bets()
    resolved  = get_resolved_bets(limit=30)

    # 30-day win rate
    wins  = sum(1 for r in resolved if r["resolution_outcome"] == 1)
    total = len(resolved)
    win_rate = round(wins / total, 3) if total else None

    # Today's PNL
    today = datetime.now(timezone.utc).date().isoformat()
    today_pnl = sum(
        r["pnl"] for r in resolved
        if r["resolved_at"] and r["resolved_at"][:10] == today and r["pnl"] is not None
    )

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_mode": settings.paper_trader.enabled,
        "bankroll": settings.betting.bankroll,
        "open_positions": len(open_bets),
        "today_pnl": round(today_pnl, 4),
        "win_rate_30d": win_rate,
        "total_resolved_30d": total,
        "calibrator_active": calibrator.is_active,
        "calibrator_samples": calibrator._n_samples,
        "scheduler_running": _scheduler.running,
        "cities": list(settings.cities.keys()),
    }


@app.get("/positions")
async def positions() -> dict:
    """List all open positions."""
    open_bets = get_open_bets()
    return {
        "open_positions": [dict(b) for b in open_bets],
        "count": len(open_bets),
    }


@app.post("/reset-paper")
async def reset_paper() -> dict:
    """Safety command: close all open paper positions and reset bankroll tracking.

    Only available in paper trading mode.
    """
    if not settings.paper_trader.enabled:
        return {"error": "reset-paper is only available in paper trading mode"}

    from storm_x.storage.db import _connect
    conn = _connect()
    conn.execute(
        "UPDATE bet_history SET status='closed', resolution_outcome=0, pnl=0.0, resolved_at=? WHERE status='open'",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    logger.warning("PAPER RESET: all open positions closed")
    return {"status": "reset", "message": "All open paper positions closed with outcome=0"}


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    uvicorn.run(
        "storm_x.main:app",
        host="0.0.0.0",
        port=settings.health.port,
        log_level="warning",   # let loguru handle our logs
        reload=False,
    )


if __name__ == "__main__":
    main()
