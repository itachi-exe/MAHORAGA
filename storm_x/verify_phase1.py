"""Phase 1 verification script — run this to confirm the data foundation works.

Expected output:
  - DB initialised
  - 119+ ensemble members for London
  - generation_hash printed (changes when new model run drops)
  - Running max observation fetched (or fallback flagged)
  - Climatology loaded and a sample stat printed
"""
import asyncio
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from storm_x.storage.db import init_db
from storm_x.data.ensemble import fetch_ensemble
from storm_x.data.observation import get_running_max
from storm_x.data.climatology import load_climatology, get_climatology_stats


async def main() -> None:
    logger.info("═══ STORM-X Phase 1 Verification ═══")

    # 1. Init DB
    init_db()
    logger.success("✓ DB initialised")

    # 2. Ensemble fetch
    logger.info("Fetching ensemble members for London...")
    result = await fetch_ensemble("london")
    assert result.member_count >= 100, \
        f"Expected 100+ members, got {result.member_count}"
    logger.success("✓ Ensemble: {} members from {} | hash={}",
                   result.member_count, result.model_names, result.generation_hash)
    logger.info("  Min={:.1f}°C  Max={:.1f}°C  Mean={:.1f}°C  Std={:.1f}°C",
                float(result.members.min()), float(result.members.max()),
                float(result.members.mean()), float(result.members.std()))

    # 3. Observation
    logger.info("Fetching running daily max for London...")
    tmax, approx = await get_running_max("london")
    if tmax is not None:
        logger.success("✓ Running max: {:.1f}°C {}", tmax, "(APPROXIMATE)" if approx else "")
    else:
        logger.warning("✗ Running max fetch returned None — both Wunderground and OM fallback failed")

    # 4. Climatology
    logger.info("Loading 10-year climatology for London (may take ~30s on first run)...")
    ok = await load_climatology("london")
    if ok:
        from datetime import date
        today = date.today()
        mean, std = get_climatology_stats("london", today.month, today.day)
        logger.success("✓ Climatology loaded | Today's historical mean={:.1f}°C std={:.1f}°C",
                       mean, std)
    else:
        logger.error("✗ Climatology load failed")

    logger.info("═══ Phase 1 complete ═══")


if __name__ == "__main__":
    asyncio.run(main())
