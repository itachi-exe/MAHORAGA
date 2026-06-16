"""Model update lag detector — triggers immediate rescan when a new model run drops.

Open-Meteo generates new ECMWF/GFS/ICON runs roughly every 6-12 hours.
By comparing the generation_hash of the current fetch to the previous one,
we detect when the underlying forecast has changed.

When a new run is detected, the scheduler triggers an immediate market rescan
and edge computation — this is the highest-value window because the market
price hasn't moved yet but the forecast has.
"""
from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from storm_x.data.ensemble import fetch_ensemble, last_generation_hash, EnsembleResult

# In-memory cache of last known hash per city
_last_hash: dict[str, str] = {}


async def check_for_new_run(city: str) -> tuple[bool, EnsembleResult]:
    """Fetch ensemble and return (new_run_detected, result).

    A new model run is detected when the generation_hash changes from the
    last time we checked.  The caller should immediately rescan markets.
    """
    result = await fetch_ensemble(city)
    prev   = _last_hash.get(city)
    new_run = (prev is not None) and (result.generation_hash != prev)

    if new_run:
        logger.info(
            "NEW MODEL RUN DETECTED | city={} | hash {} → {}",
            city, prev[:8], result.generation_hash[:8],
        )
    else:
        logger.debug("No new model run | city={} hash={}", city, result.generation_hash[:8])

    _last_hash[city] = result.generation_hash
    return new_run, result
