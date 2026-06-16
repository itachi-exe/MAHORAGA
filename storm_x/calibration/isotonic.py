"""Isotonic regression calibrator — maps raw model probabilities to observed frequencies.

Refit nightly from the last 60 resolved bets stored in SQLite.
Before 60 resolved bets exist, the calibrator is a no-op identity function.

Isotonic regression is monotone-constrained, making it ideal for probability
calibration: it can only flatten or steepen the probability-to-frequency curve,
never invert it.
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.isotonic import IsotonicRegression

from storm_x.config import settings
from storm_x.storage.db import get_resolved_bets

_MIN_BETS = settings.calibration.get("min_resolved_bets", 60)


class Calibrator:
    """Isotonic regression probability calibrator with lazy refit."""

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None
        self._n_samples: int = 0
        self._fitted: bool = False

    @property
    def is_active(self) -> bool:
        return self._fitted

    def refit(self) -> None:
        """Load last N resolved bets from DB and refit the isotonic regression."""
        rows = get_resolved_bets(limit=_MIN_BETS)
        resolved = [r for r in rows if r["resolution_outcome"] is not None]

        if len(resolved) < _MIN_BETS:
            logger.info("Calibrator: only {} resolved bets (need {}) — identity mode",
                        len(resolved), _MIN_BETS)
            self._fitted = False
            return

        probs    = np.array([r["model_prob"]         for r in resolved], dtype=np.float64)
        outcomes = np.array([r["resolution_outcome"]  for r in resolved], dtype=np.float64)

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        model.fit(probs, outcomes)

        self._model    = model
        self._n_samples = len(resolved)
        self._fitted   = True
        logger.info("Calibrator refit on {} bets", self._n_samples)

    def calibrate(self, raw_prob: float) -> float:
        """Map raw model probability to calibrated probability.

        Returns raw_prob unchanged if fewer than _MIN_BETS resolved bets exist.
        """
        if not self._fitted or self._model is None:
            return raw_prob
        result = float(self._model.predict([[raw_prob]])[0])
        return max(0.001, min(0.999, result))


# Module-level singleton — import and use this everywhere
calibrator = Calibrator()
