"""Edge calculator — computes model_prob vs market_price for every bracket and side.

Edge is defined as:  edge = model_probability - market_ask_price

Positive edge on YES side: we think YES is more likely than the market implies.
Positive edge on NO side:  we think NO is more likely than the market implies.

Both sides are evaluated independently.  Only the better edge per bracket is
kept.  A bracket qualifies for betting only when |edge| exceeds the regime-
adjusted threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from loguru import logger

from storm_x.probability.brackets import Bracket, bracket_probability, bracket_from_market
from storm_x.calibration.bias import apply_bias


@dataclass
class BracketEdge:
    market: dict
    bracket: Bracket
    side: str            # 'YES' or 'NO'
    model_prob: float
    market_price: float  # ask price for the chosen side
    edge: float          # model_prob - market_price
    token_id: str


def compute_edges(
    markets: list[dict],
    raw_members: np.ndarray,
    city: str,
    target_date: datetime,
    lead_hours: float,
    edge_threshold: float,
) -> list[BracketEdge]:
    """Compute edge for every tradeable bracket.

    Args:
        markets:        List of tradeable market dicts with live yes_price/no_price.
        raw_members:    Raw (not yet bias-corrected) ensemble member array.
        city:           City key.
        target_date:    Resolution date.
        lead_hours:     Hours until market resolution.
        edge_threshold: Minimum |edge| to include in results.

    Returns:
        List of BracketEdge with |edge| ≥ edge_threshold, sorted by edge descending.
    """
    members   = apply_bias(raw_members, city, target_date)
    n_markets = len(markets)

    edges: list[BracketEdge] = []
    for mkt in markets:
        bracket = bracket_from_market(mkt)
        p_model = bracket_probability(
            members, bracket, n_markets,
            city, target_date.month, target_date.day, lead_hours,
        )

        yes_ask = float(mkt.get("yes_price", 0.5))
        no_ask  = 1.0 - float(mkt.get("no_price", yes_ask))

        edge_yes = p_model - yes_ask
        edge_no  = (1.0 - p_model) - no_ask

        best_edge  = edge_yes if edge_yes >= edge_no else edge_no
        best_side  = "YES"    if edge_yes >= edge_no else "NO"
        best_price = yes_ask  if best_side == "YES"  else no_ask
        best_token = mkt["token_yes"] if best_side == "YES" else mkt["token_no"]

        logger.debug(
            "{} | model={:.3f} yes_ask={:.3f} e_yes={:+.3f} e_no={:+.3f}",
            bracket.label(), p_model, yes_ask, edge_yes, edge_no,
        )

        if abs(best_edge) >= edge_threshold:
            edges.append(BracketEdge(
                market=mkt,
                bracket=bracket,
                side=best_side,
                model_prob=p_model,
                market_price=best_price,
                edge=best_edge,
                token_id=best_token,
            ))

    edges.sort(key=lambda e: e.edge, reverse=True)
    logger.info("Edge scan: {} brackets, {} cleared threshold={:.0%}",
                len(markets), len(edges), edge_threshold)
    return edges
