"""APScheduler orchestration — main daily trading loop + intraday updates + nightly refit.

Schedule:
  Every 15 min   → main_cycle()        — discover markets, compute edges, size and execute bets
  Every 30 min   → intraday_update()   — re-condition members on running max, re-check exits
  02:00 UTC      → nightly_refit()     — update bias estimates, refit isotonic calibrator

All jobs run in an asyncio event loop via AsyncIOScheduler.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta, date

import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from storm_x.calibration.bias import apply_bias
from storm_x.config import settings
from storm_x.data.ensemble import fetch_ensemble
from storm_x.data.observation import get_running_max
from storm_x.edges.model_lag import check_for_new_run
from storm_x.edges.tail_bucket import effective_edge_threshold
from storm_x.edges.correlation import record_correlation, effective_n_cities
from storm_x.execution.monitor import check_exits
from storm_x.execution.orders import execute_bet
from storm_x.learning.refit import nightly_refit
from storm_x.markets.discovery import fetch_markets
from storm_x.markets.liquidity import filter_markets
from storm_x.markets.prices import enrich_market_prices
from storm_x.markets.edge import compute_edges
from storm_x.probability.intraday import should_update, condition_members, is_bracket_eliminated
from storm_x.regime.scorer import compute_score, edge_threshold, model_agreement_gate
from storm_x.sizing.allocator import allocate
from storm_x.storage.db import init_db


_PAPER = settings.paper_trader.enabled
_BANKROLL = settings.betting.bankroll


async def _run_city(city_key: str, target_date: date, dry_run: bool) -> dict:
    """Run a full trading cycle for one city. Returns a summary dict."""
    summary = {"city": city_key, "bets_placed": 0, "skipped": None}

    # 1. Fetch ensemble (checks for new model run)
    new_run, ens_result = await check_for_new_run(city_key)
    members = ens_result.members  # raw np.ndarray, shape (N,)

    target_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    lead_hours = (target_dt - datetime.now(timezone.utc)).total_seconds() / 3600

    # 2. Model agreement gate
    if not model_agreement_gate(members, ens_result.model_names):
        logger.warning("[{}] Agreement gate failed — skipping city today", city_key)
        summary["skipped"] = "agreement_gate"
        return summary

    # 3. Bias-correct members
    corrected_members = apply_bias(members, city_key, target_dt)

    # 4. Regime score → edge threshold
    regime_score = compute_score(corrected_members, lead_hours)
    base_threshold = edge_threshold(regime_score)
    if base_threshold is None:
        logger.warning("[{}] Chaotic regime ({:.2f}) — no bets today", city_key, regime_score)
        summary["skipped"] = "chaotic_regime"
        return summary

    # 5. Market discovery + liquidity filter + price enrichment
    raw_markets = await fetch_markets(city_key, target_date)
    tradeable, dead = filter_markets(raw_markets)
    if not tradeable:
        logger.info("[{}] No tradeable markets found", city_key)
        summary["skipped"] = "no_markets"
        return summary

    markets_with_prices = await enrich_market_prices(tradeable)

    # 6. Compute edges (tail threshold applied per bracket)
    n_brackets = len(markets_with_prices)
    edges = []
    for mkt in markets_with_prices:
        yes_ask = float(mkt.get("yes_price", 0.5))
        eff_thresh = effective_edge_threshold(yes_ask, base_threshold)
        if eff_thresh is None:
            continue
        city_edges = compute_edges(
            [mkt], members, city_key, target_dt, lead_hours, eff_thresh
        )
        edges.extend(city_edges)

    if not edges:
        logger.info("[{}] No edges above threshold", city_key)
        summary["skipped"] = "no_edges"
        return summary

    # 7. Sizing + allocation
    sized_bets = allocate(edges, corrected_members, n_brackets, _BANKROLL)
    if not sized_bets:
        summary["skipped"] = "zero_stakes"
        return summary

    # 8. Execute
    for bet in sized_bets:
        await execute_bet(bet, dry_run=dry_run)

    summary["bets_placed"] = len(sized_bets)
    return summary


async def main_cycle() -> None:
    """Full trading cycle for all configured cities."""
    dry_run = _PAPER
    target_date = (datetime.now(timezone.utc) + timedelta(days=1)).date()

    logger.info("=== STORM-X MAIN CYCLE {} dry_run={} ===", target_date, dry_run)

    all_members: dict[str, np.ndarray] = {}
    for city_key in settings.cities:
        try:
            summary = await _run_city(city_key, target_date, dry_run)
            if "members" in summary:
                all_members[city_key] = summary["members"]
            logger.info("City {} | bets={} skipped={}", city_key, summary["bets_placed"], summary["skipped"])
        except Exception as exc:
            logger.exception("Main cycle error for city {}: {}", city_key, exc)

    # Record cross-city correlation if both cities have members
    if "london" in all_members and "berlin" in all_members:
        record_correlation(all_members["london"], all_members["berlin"])


async def intraday_update() -> None:
    """Intraday update: re-condition members on running max, re-check exits."""
    target_date = (datetime.now(timezone.utc) + timedelta(days=0)).date()

    for city_key in settings.cities:
        if not should_update(city_key):
            continue

        try:
            obs, is_approx = await get_running_max(city_key)
            if obs is None:
                continue

            ens_result = await fetch_ensemble(city_key)
            conditioned = condition_members(ens_result.members, obs)
            logger.info(
                "Intraday update [{}] running_max={:.1f}°C → {}/{} members remain",
                city_key, obs, len(conditioned), len(ens_result.members),
            )
        except Exception as exc:
            logger.exception("Intraday update error [{}]: {}", city_key, exc)

    # Check exits for all open positions
    await check_exits(dry_run=_PAPER)


def build_scheduler() -> AsyncIOScheduler:
    """Build and configure the APScheduler instance."""
    scheduler = AsyncIOScheduler(timezone="UTC")

    # Main trading cycle every 15 minutes
    scheduler.add_job(
        main_cycle,
        trigger="interval",
        minutes=15,
        id="main_cycle",
        max_instances=1,
        coalesce=True,
    )

    # Intraday member conditioning + exit checks every 30 minutes
    scheduler.add_job(
        intraday_update,
        trigger="interval",
        minutes=30,
        id="intraday_update",
        max_instances=1,
        coalesce=True,
    )

    # Nightly refit at 02:00 UTC
    scheduler.add_job(
        nightly_refit,
        trigger="cron",
        hour=2,
        minute=0,
        id="nightly_refit",
        max_instances=1,
    )

    return scheduler
