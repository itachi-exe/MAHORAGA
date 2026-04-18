"""
igris_learner.py
────────────────
Online learning loop for IGRIS — updates an SGDClassifier after every resolved
bet so the model continuously adapts to current market regime without waiting
for batch retraining.

Features (7 total, fixed order):
  [0] momentum_score        backbone signal 1 raw score
  [1] ob_imbalance_score    backbone signal 2 raw score
  [2] funding_divergence    backbone signal 3 raw score
  [3] odds_velocity_score   backbone signal 4 raw score
  [4] mlp_confidence        upstream MLP confidence (0-1, NOT %)
  [5] polymarket_odds        current YES token mid-price (0-1)
  [6] direction_encoded      1=UP, 0=DOWN

Thread safety: all model mutations (partial_fit, save) are protected by a Lock.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# ── Module logger — uses same hierarchy as PolymarketTrader ──────────────────
log = logging.getLogger("PolymarketTrader.learner")

# ── Constants ────────────────────────────────────────────────────────────────
LEARNER_CONFIDENCE_THRESHOLD = 0.60   # min predict_proba to return a direction
CONSOLIDATION_EVERY          = 20     # full buffer refit every N updates
BUFFER_MAX                   = 200    # rolling sample buffer size
N_FEATURES                   = 7

DEFAULT_MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "igris_model.pkl")
DEFAULT_SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "igris_scaler.pkl")
DEFAULT_BUFFER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "igris_buffer.pkl")

# Path to the existing MAHORAGA model (for weight transfer on first run)
_MAHORAGA_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MAHORAGA_model.pkl")


class IGRISLearner:
    """
    Online SGD classifier that learns from every resolved Polymarket bet.
    Call update() after each resolution; the model adjusts immediately.
    """

    def __init__(
        self,
        model_path:  str = DEFAULT_MODEL_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
        buffer_path: str = DEFAULT_BUFFER_PATH,
    ) -> None:
        self.model_path  = model_path
        self.scaler_path = scaler_path
        self.buffer_path = buffer_path

        self._lock          = threading.Lock()
        self._update_count  = 0
        self._buffer: deque[tuple[np.ndarray, int]] = deque(maxlen=BUFFER_MAX)

        self.model  = SGDClassifier(
            loss="log_loss",
            learning_rate="adaptive",
            eta0=0.01,
            n_iter_no_change=5,
            warm_start=True,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._scaler_fitted = False
        self._model_fitted  = False

        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load model, scaler, and buffer from disk. Fall back to fresh state."""
        # Buffer
        if os.path.exists(self.buffer_path):
            try:
                self._buffer = joblib.load(self.buffer_path)
                log.info(f"[IGRISLearner] Loaded buffer ({len(self._buffer)} samples)")
            except Exception as exc:
                log.warning(f"[IGRISLearner] Buffer load failed: {exc}")

        # Scaler
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                self._scaler_fitted = True
                log.info("[IGRISLearner] Loaded scaler")
            except Exception as exc:
                log.warning(f"[IGRISLearner] Scaler load failed: {exc}")

        # Model
        if os.path.exists(self.model_path):
            try:
                saved = joblib.load(self.model_path)
                if isinstance(saved, SGDClassifier):
                    self.model = saved
                    self._model_fitted = hasattr(saved, "coef_")
                    log.info(
                        f"[IGRISLearner] Loaded SGDClassifier "
                        f"(fitted={self._model_fitted})"
                    )
                else:
                    # Unknown type saved — try weight transfer
                    self._try_weight_transfer(saved)
            except Exception as exc:
                log.warning(f"[IGRISLearner] Model load failed: {exc}")
        elif os.path.exists(_MAHORAGA_MODEL_PATH):
            # First run — try to bootstrap from MAHORAGA weights
            try:
                existing = joblib.load(_MAHORAGA_MODEL_PATH)
                self._try_weight_transfer(existing)
            except Exception as exc:
                log.debug(f"[IGRISLearner] MAHORAGA weight transfer skipped: {exc}")

    def _try_weight_transfer(self, existing_model) -> None:
        """
        Attempt to copy weights from an existing sklearn linear model.
        If the model has coef_ / intercept_ of compatible shape, copy them.
        """
        try:
            if hasattr(existing_model, "coef_") and hasattr(existing_model, "intercept_"):
                coef = np.asarray(existing_model.coef_)
                intercept = np.asarray(existing_model.intercept_)
                # Only transfer if shape is exactly (1, N_FEATURES) or (N_FEATURES,)
                if coef.shape[-1] == N_FEATURES:
                    # Warm-initialise with a tiny dummy fit so coef_ is writable
                    dummy_X = np.zeros((2, N_FEATURES))
                    dummy_y = [0, 1]
                    self.model.partial_fit(dummy_X, dummy_y, classes=[0, 1])
                    if coef.ndim == 1:
                        coef = coef.reshape(1, -1)
                    self.model.coef_      = coef[:1, :]    # shape (1, N_FEATURES)
                    self.model.intercept_ = intercept[:1]  # shape (1,)
                    self._model_fitted = True
                    log.info(
                        f"[IGRISLearner] Transferred weights from "
                        f"{type(existing_model).__name__} "
                        f"(coef shape={self.model.coef_.shape})"
                    )
                    return
            log.warning(
                "[IGRISLearner] WARNING: Existing model type incompatible, starting fresh"
            )
        except Exception as exc:
            log.warning(f"[IGRISLearner] Weight transfer failed: {exc} — starting fresh")

    def _save(self) -> None:
        """Persist model, scaler, and buffer. Called inside _lock."""
        try:
            joblib.dump(self.model,   self.model_path)
            joblib.dump(self.scaler,  self.scaler_path)
            joblib.dump(self._buffer, self.buffer_path)
        except Exception as exc:
            log.warning(f"[IGRISLearner] Save failed: {exc}")

    # ── Feature building ─────────────────────────────────────────────────────

    def build_features(
        self,
        backbone_scores: dict,
        confidence: float,
        odds: float,
        direction: str,
    ) -> np.ndarray:
        """
        Build the 7-feature vector and apply StandardScaler transform.

        Args:
            backbone_scores: dict with keys momentum, ob_imbalance,
                             funding_divergence, odds_velocity
            confidence:      MLP confidence in [0, 1] (divide by 100 before passing)
            odds:            current Polymarket YES price in [0, 1]
            direction:       "UP" or "DOWN"

        Returns:
            np.ndarray shape (1, 7), scaled
        """
        raw = np.array([[
            float(backbone_scores.get("momentum",            0.0)),
            float(backbone_scores.get("ob_imbalance",        0.0)),
            float(backbone_scores.get("funding_divergence",  0.0)),
            float(backbone_scores.get("odds_velocity",       0.0)),
            float(confidence),
            float(odds),
            1.0 if direction == "UP" else 0.0,
        ]], dtype=np.float64)  # shape (1, 7)

        with self._lock:
            if not self._scaler_fitted:
                self.scaler.partial_fit(raw)
                self._scaler_fitted = True
            return self.scaler.transform(raw)

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """
        Predict direction and confidence.

        Returns:
            ("UP"|"DOWN"|"NONE", probability)
            NONE when confidence < LEARNER_CONFIDENCE_THRESHOLD or model not fitted.
        """
        with self._lock:
            if not self._model_fitted:
                log.debug("[IGRISLearner] Model not yet fitted — returning NONE")
                return "NONE", 0.0
            try:
                proba = self.model.predict_proba(features)[0]  # [P(0), P(1)]
                conf  = float(proba[1])   # P(WON)
                if conf >= LEARNER_CONFIDENCE_THRESHOLD:
                    return "UP", conf
                elif (1.0 - conf) >= LEARNER_CONFIDENCE_THRESHOLD:
                    return "DOWN", 1.0 - conf
                else:
                    return "NONE", max(conf, 1.0 - conf)
            except Exception as exc:
                log.warning(f"[IGRISLearner] predict error: {exc}")
                return "NONE", 0.0

    # ── Online update ─────────────────────────────────────────────────────────

    def update(self, features: np.ndarray, actual_outcome: str) -> None:
        """
        Incorporate one resolved bet into the model.

        Args:
            features:        np.ndarray shape (1, 7) — same array from build_features()
            actual_outcome:  "WON" or "LOST"

        Side effects:
            - Calls partial_fit immediately
            - Appends to rolling buffer
            - Every CONSOLIDATION_EVERY updates: refits scaler on full buffer,
              retransforms, refits model on full buffer
            - Saves model + scaler + buffer to disk
            - Logs update details
        """
        label = 1 if actual_outcome == "WON" else 0

        with self._lock:
            try:
                self.model.partial_fit(features, [label], classes=[0, 1])
                self._model_fitted = True
            except Exception as exc:
                log.warning(f"[IGRISLearner] partial_fit error: {exc}")
                return

            self._buffer.append((features.copy(), label))
            self._update_count += 1

            # ── Consolidation step every N updates ───────────────────────────
            if self._update_count % CONSOLIDATION_EVERY == 0 and len(self._buffer) >= 10:
                try:
                    # Buffer stores features already scaled by build_features() —
                    # do NOT refit the scaler here or we would double-scale.
                    buf_X = np.vstack([x for x, _ in self._buffer])
                    buf_y = np.array([y for _, y in self._buffer])

                    # Refit model on full buffer (already-scaled features)
                    self.model = SGDClassifier(
                        loss="log_loss",
                        learning_rate="adaptive",
                        eta0=0.01,
                        n_iter_no_change=5,
                        warm_start=True,
                        random_state=42,
                    )
                    self.model.fit(buf_X, buf_y)
                    self._model_fitted = True

                    acc = self.get_accuracy(_locked=True)
                    log.info(
                        f"[IGRISLearner] Consolidation #{self._update_count // CONSOLIDATION_EVERY}"
                        f" | buffer={len(self._buffer)} samples | accuracy={acc:.3f}"
                    )
                except Exception as exc:
                    log.warning(f"[IGRISLearner] Consolidation failed: {exc}")

            # Compute accuracy on current buffer for the log line
            try:
                acc = self.get_accuracy(_locked=True)
            except Exception:
                acc = 0.0

            # Save immediately after every update
            self._save()

        ts = datetime.now(timezone.utc).isoformat()
        log.info(
            f"[IGRISLearner] {ts} | update #{self._update_count} "
            f"label={label} ({actual_outcome}) "
            f"features={features.flatten().round(4).tolist()} "
            f"buffer={len(self._buffer)} accuracy={acc:.3f}"
        )

    # ── Accuracy ─────────────────────────────────────────────────────────────

    def get_accuracy(self, _locked: bool = False) -> float:
        """
        Score the model on the current rolling buffer.
        _locked=True skips re-acquiring the lock (for internal calls already inside _lock).
        """
        def _compute() -> float:
            if not self._model_fitted or len(self._buffer) < 2:
                return 0.0
            buf_X = np.vstack([x for x, _ in self._buffer])
            buf_y = np.array([y for _, y in self._buffer])
            try:
                return float(self.model.score(buf_X, buf_y))
            except Exception:
                return 0.0

        if _locked:
            return _compute()
        with self._lock:
            return _compute()
