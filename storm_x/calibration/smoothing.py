"""Laplace (add-one) smoothing for empirical bracket probabilities.

Prevents zero and one probabilities that would cause Kelly sizing to blow up.
Pure stateless functions — no side effects.
"""
from __future__ import annotations


def laplace(count: int, total: int, n_brackets: int) -> float:
    """Laplace-smoothed probability for a bracket.

    Formula: (count + 1) / (total + n_brackets)

    Args:
        count:      Number of ensemble members falling in this bracket.
        total:      Total number of ensemble members.
        n_brackets: Total number of brackets being considered (the pseudo-count
                    is spread uniformly across all brackets).

    Returns:
        Smoothed probability in (0, 1).
    """
    if total <= 0:
        return 1.0 / max(n_brackets, 1)
    return (count + 1) / (total + n_brackets)


def clamp(p: float, lo: float = 0.001, hi: float = 0.999) -> float:
    """Hard-clamp a probability to [lo, hi] — last-resort guard against edge cases."""
    return max(lo, min(hi, p))
