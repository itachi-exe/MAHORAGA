"""Bootstrap asymmetric Kelly position sizer.

Standard Kelly uses the point-estimate probability, which is over-confident.
Bootstrap Kelly:
  1. Resample the ensemble member array 1000 times (with replacement).
  2. For each resample, recompute the bracket probability.
  3. Take the 25th-percentile probability as the conservative estimate.
  4. Run Kelly on the conservative estimate, multiplied by 0.20.
  5. Cap at 3% of bankroll per market.

Using the 25th percentile instead of the mean means we size for the
plausible-but-unlucky scenario, not the central case.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from storm_x.config import settings
from storm_x.probability.brackets import Bracket, bracket_probability
from storm_x.calibration.smoothing import laplace, clamp

_N_BOOTSTRAP    = settings.sizing.bootstrap_samples         # 1000
_CONS_PCT       = settings.sizing.conservative_percentile   # 25
_KELLY_MULT     = settings.betting.kelly_multiplier         # 0.20
_PER_MARKET_CAP = settings.betting.per_market_cap           # 0.03


def _raw_count_prob(members: np.ndarray, bracket: Bracket, n_brackets: int) -> float:
    """Fast bracket probability without calibration — used in bootstrap inner loop."""
    count = int(np.sum((members >= bracket.lo) & (members < bracket.hi)))
    return laplace(count, len(members), n_brackets)


def bootstrap_conservative_prob(
    members: np.ndarray,
    bracket: Bracket,
    n_brackets: int,
) -> float:
    """Return the 25th-percentile bracket probability over 1000 bootstrap resamples.

    This is the 'conservative' probability used for Kelly sizing.
    """
    n = len(members)
    probs = np.empty(_N_BOOTSTRAP, dtype=np.float64)
    rng   = np.random.default_rng()

    for i in range(_N_BOOTSTRAP):
        resampled = rng.choice(members, size=n, replace=True)
        probs[i]  = _raw_count_prob(resampled, bracket, n_brackets)

    return float(np.percentile(probs, _CONS_PCT))


def kelly_fraction(prob: float, price: float) -> float:
    """Compute full Kelly fraction for a bet at price with probability prob.

    Returns 0.0 if the bet has negative expected value.
    """
    if price <= 0 or price >= 1:
        return 0.0
    b = (1.0 - price) / price      # net odds: win $b for every $1 risked
    f = (prob * b - (1.0 - prob)) / b
    return max(0.0, f)


def compute_stake(
    members: np.ndarray,
    bracket: Bracket,
    n_brackets: int,
    market_price: float,
    bankroll: float,
) -> tuple[float, float, float]:
    """Compute the stake for a single bracket bet.

    Returns:
        (stake_usdc, kelly_f, conservative_prob)
        stake_usdc = 0.0 if Kelly is zero or negative.
    """
    p_cons  = bootstrap_conservative_prob(members, bracket, n_brackets)
    k       = kelly_fraction(p_cons, market_price)
    k_adj   = k * _KELLY_MULT
    cap_abs = bankroll * _PER_MARKET_CAP

    stake = min(bankroll * k_adj, cap_abs)
    stake = max(0.0, stake)

    logger.debug(
        "Kelly | {} | p_cons={:.3f} price={:.3f} k={:.4f} k_adj={:.4f} stake=${:.4f}",
        bracket.label(), p_cons, market_price, k, k_adj, stake,
    )
    return round(stake, 4), round(k_adj, 6), round(p_cons, 4)
