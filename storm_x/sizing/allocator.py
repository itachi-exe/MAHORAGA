"""Portfolio allocator — enforces total daily exposure cap across all bets.

If the sum of proposed stakes exceeds 10% of bankroll, all stakes are scaled
proportionally so the total equals exactly 10%.  Bets are sorted by edge
descending so the highest-edge positions are filled first.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from storm_x.config import settings
from storm_x.markets.edge import BracketEdge
from storm_x.sizing.kelly import compute_stake
from storm_x.probability.brackets import Bracket

_DAILY_CAP = settings.betting.daily_exposure_cap   # 0.10 (10% of bankroll)


@dataclass
class SizedBet:
    edge_record: BracketEdge
    stake: float
    kelly_fraction: float
    conservative_prob: float


def allocate(
    edge_records: list[BracketEdge],
    members: np.ndarray,
    n_brackets: int,
    bankroll: float,
) -> list[SizedBet]:
    """Size and allocate capital across eligible bets.

    Args:
        edge_records: Bets that cleared the edge threshold, sorted by edge desc.
        members:      Bias-corrected ensemble member array (for bootstrap Kelly).
        n_brackets:   Total bracket count (Laplace denominator).
        bankroll:     Current account balance in USDC.

    Returns:
        List of SizedBet with scaled stakes that respect the daily exposure cap.
        Empty list if no bets survive sizing.
    """
    daily_cap_usdc = bankroll * _DAILY_CAP
    sized: list[SizedBet] = []

    for er in edge_records:
        stake, k, p_cons = compute_stake(
            members, er.bracket, n_brackets, er.market_price, bankroll
        )
        if stake <= 0.0:
            logger.debug("Zero stake for {} — skipping", er.bracket.label())
            continue
        sized.append(SizedBet(er, stake, k, p_cons))

    if not sized:
        return []

    total_proposed = sum(b.stake for b in sized)
    if total_proposed > daily_cap_usdc:
        scale = daily_cap_usdc / total_proposed
        logger.info(
            "Scaling stakes by {:.2f}x — total ${:.2f} > daily cap ${:.2f}",
            scale, total_proposed, daily_cap_usdc,
        )
        for b in sized:
            b.stake = round(b.stake * scale, 4)

    logger.info(
        "Allocated {} bets | total stake=${:.2f} | bankroll=${:.2f}",
        len(sized), sum(b.stake for b in sized), bankroll,
    )
    return sized
