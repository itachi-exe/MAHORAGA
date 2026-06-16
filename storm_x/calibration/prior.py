"""Climatology prior blender — reduces over-confidence at long forecast lead times.

At 5 days (120 h) lead time, the ensemble members carry little predictive skill
beyond climatology.  At 12 h lead time the ensemble is the dominant signal.
A smooth exponential decay interpolates between these extremes.

weight_clim(t) = w_short + (w_long - w_short) * exp(-λ * (t - t_short) / (t_long - t_short))
where t is clipped to [t_short, t_long].

Final probability = weight_clim * p_clim + (1 - weight_clim) * p_ensemble
"""
from __future__ import annotations

import math

from storm_x.config import settings

_W_LONG  = settings.edges.climatology_blend_long_horizon_weight   # 0.60 at 120 h
_W_SHORT = settings.edges.climatology_blend_short_horizon_weight  # 0.05 at 12 h
_T_LONG  = 120.0   # hours
_T_SHORT = 12.0    # hours
_LAMBDA  = 3.0     # decay rate — controls steepness of transition


def climatology_weight(lead_hours: float) -> float:
    """Return the weight to assign to climatology prior at this forecast lead time.

    Args:
        lead_hours: Hours until market resolution (e.g. 36 h if betting tonight
                    on tomorrow's max and resolution is at midnight local).

    Returns:
        Float in [_W_SHORT, _W_LONG].
    """
    t = max(_T_SHORT, min(_T_LONG, float(lead_hours)))
    # Linear fraction of the way from short to long horizon
    frac = (t - _T_SHORT) / (_T_LONG - _T_SHORT)
    # Exponential curve so most of the transition happens in the middle
    weight = _W_SHORT + (_W_LONG - _W_SHORT) * (1 - math.exp(-_LAMBDA * frac))
    return weight


def blend(p_ensemble: float, p_climatology: float, lead_hours: float) -> float:
    """Blend ensemble and climatology probabilities weighted by lead time.

    Args:
        p_ensemble:    Probability from empirical ensemble member counting.
        p_climatology: Probability from climatology (historical frequency in bracket).
        lead_hours:    Hours until market resolution.

    Returns:
        Blended probability in (0, 1).
    """
    w = climatology_weight(lead_hours)
    return w * p_climatology + (1.0 - w) * p_ensemble


def climatology_bracket_prob(mean: float, std: float, lo: float, hi: float) -> float:
    """Probability that a N(mean, std) variable falls in [lo, hi].

    Used to convert historical mean/std from the climatology table into
    a bracket probability comparable to the ensemble-derived probability.
    """
    import math
    def _phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    if std <= 0:
        return 1.0 if lo <= mean < hi else 0.0

    z_lo = (lo - mean) / std
    z_hi = (hi - mean) / std
    return max(0.0, _phi(z_hi) - _phi(z_lo))
