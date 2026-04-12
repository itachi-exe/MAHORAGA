"""
MAHORAGA core_trading_system.py

AI model: MLP (Multi-Layer Perceptron), 18 input features.
Replaces the prior LSTM architecture (seq_len=1 gave no temporal benefit).

Label scheme: forward-return based (3-candle horizon).
Eliminates the indicator-derived tautology where labels were deterministic
functions of the same features fed to the model.

All fixes applied in this revision:
  CRITICAL-1 : Forward-return labels replace ICT indicator tautology
  CRITICAL-2 : Unclosed candle dropped before feature/label computation
  CRITICAL-3 : LSTM replaced with a proper MLP
  HIGH-1     : Train/val split — accuracy evaluated on held-out set only
  HIGH-2     : VirtualTrader position sizing mirrors live formula
  MEDIUM-2   : Equal-weight CrossEntropyLoss (weighted experiments collapsed HOLD recall)
  MEDIUM-5   : Scaler always refit from scratch on each full retrain
  NEW        : fetch_orderbook_imbalance, fetch_funding_rate
  NEW        : Signal confidence overlays (OB imbalance + funding rate)
  NEW        : MAHORAGA.get_prediction_snapshot() (async, read-only)
  NEW        : Adaptive risk engine integrated into MAHORAGA
               (compute_adaptive_params, compute_atr_baseline)
               Modulates SL/TP/cooloff ±40% based on ATR ratio, session,
               win/loss streaks, and model confidence.
"""

import os
import sys
import json
import asyncio
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import ta

from datetime import datetime, timezone

# ── PyTorch — optional. Falls back to technical-indicator predict if absent. ─
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

    class _FakeModule:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return None
        def eval(self): return self
        def train(self): return self
        def parameters(self): return iter([])
        def state_dict(self): return {}
        def load_state_dict(self, *a, **kw): pass
        def to(self, *a, **kw): return self

    class _FakeOptimizer:
        def __init__(self, *a, **kw): pass
        def zero_grad(self): pass
        def step(self): pass

    class nn:
        Module           = _FakeModule
        Sequential       = _FakeModule
        Linear           = _FakeModule
        BatchNorm1d      = _FakeModule
        ReLU             = _FakeModule
        Dropout          = _FakeModule
        CrossEntropyLoss = _FakeModule

    class optim:
        Adam = _FakeOptimizer

    class _FakeTensor:
        def __init__(self, *a, **kw): pass
        def to(self, *a, **kw): return self
        def mean(self, *a, **kw): return self
        def cpu(self): return self
        def numpy(self): return np.array([])
        def backward(self): pass
        def __mul__(self, other): return self

    class torch:
        float32 = None
        long    = None

        @staticmethod
        def tensor(*a, **kw): return _FakeTensor()
        @staticmethod
        def ones(*a, **kw): return _FakeTensor()
        @staticmethod
        def zeros(*a, **kw): return _FakeTensor()
        @staticmethod
        def argmax(*a, **kw): return _FakeTensor()
        @staticmethod
        def save(*a, **kw): pass
        @staticmethod
        def load(*a, **kw): return {}

        class device:
            def __init__(self, *a): pass

        class no_grad:
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class cuda:
            @staticmethod
            def is_available(): return False

    class F:
        @staticmethod
        def softmax(x, dim=None): return x


# ── Alpaca (optional — used outside of live Bybit trading) ──────────────────
try:
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.historical import (StockHistoricalDataClient,
                                        CryptoHistoricalDataClient)
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# ── Bybit (live trading) ─────────────────────────────────────────────────────
from pybit.unified_trading import HTTP
from dotenv import load_dotenv


# ── Runtime path helpers ─────────────────────────────────────────────────────
def _app_dir() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def _bundle_dir() -> str:
    return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))


load_dotenv(dotenv_path=os.path.join(_bundle_dir(), '.env'), override=False)
load_dotenv(dotenv_path=os.path.join(_app_dir(),    '.env'), override=True)


# ── Ghost-in-the-Market risk constants ──────────────────────────────────────
# STOP_LOSS_PCT, TAKE_PROFIT_PCT, COOLOFF_HOURS are intentionally absent here.
# Base values (1.0 / 3.0 / 5.0) live inside MAHORAGA.BASE_* and are modulated
# per-trade by compute_adaptive_params().  server.py AutoTrader should import
# AdaptiveRisk via the MAHORAGA instance rather than using hardcoded constants.
TRADE_JOURNAL_FILE          = 'trade_journal.json'
RETRADING_CYCLE_LENGTH      = 20
WINNING_TRADE_WEIGHT        = 2.0
LOSING_TRADE_WEIGHT         = 0.5
RISK_PER_TRADE              = 0.01
MAX_DAILY_LOSS              = 0.03
MAX_DAILY_PROFIT            = 0.10
MAX_TRADES_PER_DAY          = 10
MAX_CONSECUTIVE_LOSSES      = 3
CONFIDENCE_THRESHOLD_GHOST  = 0.65
MAX_POSITIONS_PER_DIRECTION = 3

# ── Forward-return labeling constants (CRITICAL-1) ───────────────────────────
# Labels encode actual future price movement, not current indicator state.
FORWARD_BARS   = 3       # number of candles ahead to measure return
BUY_THRESHOLD  =  0.003  # +0.3% forward return → BUY label  (0.5% → 81% HOLD → 9% acc)
SELL_THRESHOLD = -0.003  # -0.3% forward return → SELL label

# ── Feature columns — exactly 18, fixed order ────────────────────────────────
# Order is frozen: any change invalidates saved model + scaler.
MODEL_FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'RSI', 'MACD', 'BB_upper', 'BB_lower', 'MA_5', 'MA_10',
    'ATR', 'close_lag1', 'close_lag2', 'RSI_lag1', 'MACD_lag1',
    'ob_imbalance', 'funding_rate',
]
MODEL_INPUT_SIZE = len(MODEL_FEATURE_COLS)  # 18


# ─────────────────────────────────────────────────────────────────────────────
# Bybit client — defined early so live-data helpers can call it
# ─────────────────────────────────────────────────────────────────────────────

def get_bybit_client(api_key: str = None, api_secret: str = None,
                     testnet: bool = False) -> HTTP:
    key    = api_key    or os.getenv('BYBIT_API_KEY',    '')
    secret = api_secret or os.getenv('BYBIT_API_SECRET', '')
    return HTTP(testnet=testnet, api_key=key, api_secret=secret,
                recv_window=10000)


# ─────────────────────────────────────────────────────────────────────────────
# Live-data fetch helpers (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_orderbook_imbalance(symbol: str = 'BTCUSDT',
                               api_key: str = None,
                               api_secret: str = None) -> float:
    """
    Bid/ask volume imbalance from the top-20 order book levels.
    Returns (bid_vol - ask_vol) / (bid_vol + ask_vol) in [-1.0, +1.0].
    +1 = fully bid-heavy, -1 = fully ask-heavy. Returns 0.0 on any error.
    """
    try:
        session = get_bybit_client(api_key, api_secret)
        resp = session.get_orderbook(category='linear', symbol=symbol, limit=20)
        if resp.get('retCode') != 0:
            return 0.0
        bids = resp['result'].get('b', [])
        asks = resp['result'].get('a', [])
        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total   = bid_vol + ask_vol
        if total == 0.0:
            return 0.0
        return round((bid_vol - ask_vol) / total, 6)
    except Exception:
        return 0.0


def fetch_funding_rate(symbol: str = 'BTCUSDT',
                        api_key: str = None,
                        api_secret: str = None) -> float:
    """
    Current perpetual funding rate from the tickers endpoint.
    Typical range: -0.001 to +0.001 per 8-hour period.
    Returns 0.0 on any error.
    """
    try:
        session = get_bybit_client(api_key, api_secret)
        resp = session.get_tickers(category='linear', symbol=symbol)
        if resp.get('retCode') != 0:
            return 0.0
        tickers = resp['result'].get('list', [])
        if not tickers:
            return 0.0
        return float(tickers[0].get('fundingRate', 0.0))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Signal confidence overlays (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_signal_overlays(signal: str, confidence: float,
                            ob_imbalance: float, funding_rate: float) -> float:
    """
    Adjust model confidence using live order-book imbalance and funding rate.
    All adjustments are additive percentage-point changes to the 0–1 confidence
    score. Final result is clamped to [0.01, 0.99].

    Order-book logic:
      Bid-heavy book (>+0.35) confirms a BUY, contradicts a SELL.
      Ask-heavy book (<-0.35) confirms a SELL, contradicts a BUY.

    Funding-rate logic:
      High positive funding (>0.0008): market is over-long.
        → BUY becomes riskier (mean-reversion), SELL gets a squeeze tailwind.
      High negative funding (<-0.0004): market is over-short.
        → SELL becomes riskier, BUY gets a squeeze tailwind.
    """
    c = confidence

    # ── Order-book imbalance overlays ────────────────────────────────────────
    if signal == 'BUY':
        if ob_imbalance > 0.35:
            c += 0.10   # bid-heavy: confluence
        elif ob_imbalance < -0.35:
            c -= 0.15   # ask-heavy: contradiction
    elif signal == 'SELL':
        if ob_imbalance < -0.35:
            c += 0.10   # ask-heavy: confluence
        elif ob_imbalance > 0.35:
            c -= 0.15   # bid-heavy: contradiction

    # ── Funding-rate overlays ────────────────────────────────────────────────
    if signal == 'BUY':
        if funding_rate > 0.0008:
            c -= 0.10   # over-long market → mean-reversion risk
        elif funding_rate < -0.0004:
            c += 0.10   # over-short squeeze → long tailwind
    elif signal == 'SELL':
        if funding_rate > 0.0008:
            c += 0.10   # over-long squeeze → short tailwind
        elif funding_rate < -0.0004:
            c -= 0.10   # over-short market → mean-reversion risk

    return round(max(0.01, min(0.99, c)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# AI_trading_bot: MLP classifier (CRITICAL-3 — replaces broken LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class AI_trading_bot(nn.Module):
    """
    Feed-forward MLP: 18 features → BUY (2) / HOLD (1) / SELL (0).

    Architecture:
        Linear(18 → 128) → ReLU → Dropout(0.2)
        Linear(128 → 64)  → ReLU → Dropout(0.1)
        Linear(64  → 3)   → Softmax

    BatchNorm is intentionally excluded: its running statistics (mean/variance
    accumulated from training batches) diverge from the validation distribution
    when the val period is a different market regime, causing systematic
    misclassification that drops val accuracy below random chance.

    The previous LSTM used seq_len=1, which gives zero temporal benefit
    (hidden state reset to zeros on every call). This MLP is simpler,
    faster, and equivalent in expressiveness for single-timestep inputs.
    """

    def __init__(self, input_size: int = MODEL_INPUT_SIZE,
                 hidden: int = 128,
                 output_size: int = 3,
                 model_path: str  = 'MAHORAGA_model.pkl',
                 scaler_path: str = 'MAHORAGA_scaler.pkl'):
        super(AI_trading_bot, self).__init__()
        self.input_size   = input_size
        self.hidden       = hidden
        self.output_size  = output_size
        self.model_path   = model_path
        self.scaler_path  = scaler_path
        self.scaler       = StandardScaler()
        self.current_accuracy = 0.0   # validation accuracy, set after each fit()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, output_size),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, features_df: pd.DataFrame, labels: np.ndarray,
            epochs: int = 20, lr: float = 0.001, batch_size: int = 32,
            sample_weights: np.ndarray = None):
        """
        Train the MLP.

        HIGH-1  : Data is split into train (85%) / val (15%) before any fitting.
                  shuffle=False is mandatory — this is ordered time-series data.
                  Accuracy is reported on the held-out val set only.

        MEDIUM-2: Equal-weight CrossEntropyLoss.
                  Weighted experiments ('balanced', capped, sqrt) all suppressed
                  HOLD recall to the point where val accuracy fell below 26% on a
                  val period where HOLD = 72.5%.  Equal weights allow the model to
                  learn the natural distribution; signal overlays + RL retraining
                  handle BUY/SELL refinement at inference time.

        MEDIUM-5: Scaler is always refit from scratch on X_train.
                  No partial_fit — eliminates accumulated drift across retrains.
        """
        if not _TORCH_AVAILABLE:
            print("PyTorch not installed — cannot train MLP.")
            return
        if features_df.empty or len(labels) == 0:
            print("Empty dataset — skipping fit.")
            return

        # 2.4 — Data freshness gate: require at least 30 unique trading days
        try:
            unique_days = len(features_df.index.normalize().unique())
            if unique_days < 30:
                print(f"[Retrain] Skipped — only {unique_days} days of data available, need 30")
                return
        except Exception:
            pass   # non-datetime index (e.g. RangeIndex from concat) — proceed

        labels = np.asarray(labels)

        # HIGH-1: temporal split — first 85% train, last 15% val
        X_train, X_val, y_train, y_val = train_test_split(
            features_df, labels, test_size=0.15, shuffle=False
        )
        y_train = np.asarray(y_train)
        y_val   = np.asarray(y_val)
        n_train = len(X_train)

        # MEDIUM-5: always refit scaler on training data — never partial_fit
        scaled_train = self.scaler.fit_transform(X_train)
        joblib.dump(self.scaler, self.scaler_path)
        scaled_val   = self.scaler.transform(X_val)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        features_t = torch.tensor(scaled_train.astype(np.float32))
        labels_t   = torch.tensor(y_train, dtype=torch.long)

        weight_tensor = torch.ones(3, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, reduction='none')

        # Sample weights for RL-weighted retraining — sliced to train split
        if sample_weights is not None:
            if len(sample_weights) != len(features_df):
                raise ValueError(
                    "sample_weights length must match features_df length."
                )
            # train_test_split with shuffle=False → first n_train rows are train
            sw_train = torch.tensor(
                sample_weights[:n_train].astype(np.float32)
            ).to(device)
        else:
            sw_train = torch.ones(n_train, dtype=torch.float32).to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        self.train()

        for _epoch in range(epochs):
            for i in range(0, len(features_t), batch_size):
                bf  = features_t[i:i + batch_size].to(device)
                bl  = labels_t[i:i + batch_size].to(device)
                bw  = sw_train[i:i + batch_size]
                out  = self(bf)
                loss = (criterion(out, bl) * bw).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # HIGH-1: accuracy on val set only
        self.eval()
        val_t = torch.tensor(scaled_val.astype(np.float32)).to(device)
        with torch.no_grad():
            val_out   = self(val_t)
            val_preds = torch.argmax(val_out, dim=1).cpu().numpy()
        self.current_accuracy = float(accuracy_score(y_val, val_preds))
        print(f"Training complete — val_accuracy={self.current_accuracy:.4f} "
              f"(train={n_train}, val={len(y_val)})")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: str = None, scaler_path: str = None):
        path        = path        or self.model_path
        scaler_path = scaler_path or self.scaler_path
        state = {
            'architecture':     'MLP',       # sentinel — rejects old LSTM files
            'state_dict':       self.state_dict(),
            'current_accuracy': self.current_accuracy,
            'input_size':       self.input_size,
            'hidden':           self.hidden,
            'output_size':      self.output_size,
        }
        torch.save(state, path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved → {path}  (val_accuracy={self.current_accuracy:.4f})")

    def load_model(self, path: str = None, scaler_path: str = None) -> bool:
        if not _TORCH_AVAILABLE:
            try:
                self.scaler = joblib.load(scaler_path or self.scaler_path)
            except Exception:
                pass
            print("PyTorch not installed — LSTM/MLP disabled, using technical fallback.")
            return False

        path        = path        or self.model_path
        scaler_path = scaler_path or self.scaler_path
        try:
            raw = torch.load(path, map_location=torch.device('cpu'))

            # Reject any checkpoint that is not from this MLP revision
            if not isinstance(raw, dict) or 'state_dict' not in raw:
                print("Incompatible checkpoint format. Retraining required.")
                return False
            if raw.get('architecture') != 'MLP':
                print("Old LSTM checkpoint detected — incompatible with MLP "
                      "architecture. Retraining required.")
                return False

            saved_input  = raw.get('input_size',  self.input_size)
            saved_hidden = raw.get('hidden',       self.hidden)
            saved_output = raw.get('output_size',  self.output_size)
            self.current_accuracy = raw.get('current_accuracy', 0.0)

            # Re-initialise weights if architecture hyperparams changed
            if (self.input_size  != saved_input  or
                    self.hidden  != saved_hidden  or
                    self.output_size != saved_output):
                self.__init__(
                    input_size=saved_input,  hidden=saved_hidden,
                    output_size=saved_output,
                    model_path=self.model_path, scaler_path=self.scaler_path,
                )

            self.load_state_dict(raw['state_dict'])
            self.scaler = joblib.load(scaler_path)
            self.eval()
            print(f"MLP loaded (input={saved_input}, hidden={saved_hidden}, "
                  f"val_accuracy={self.current_accuracy:.4f})")
            return True

        except FileNotFoundError:
            print(f"Model/scaler not found at {path} / {scaler_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame):
        """
        Returns (labels ndarray, probabilities ndarray).
        labels: 0=SELL, 1=HOLD, 2=BUY.
        """
        if not _TORCH_AVAILABLE:
            return np.array([]), np.array([])
        if not hasattr(self.scaler, 'mean_'):
            return np.array([]), np.array([])
        scaled = self.scaler.transform(features_df)
        t = torch.tensor(scaled.astype(np.float32))
        self.eval()
        with torch.no_grad():
            out = self(t)
            return torch.argmax(out, dim=1).cpu().numpy(), out.cpu().numpy()

    def evaluate_model(self, features_df: pd.DataFrame,
                        labels: np.ndarray) -> float:
        """Evaluate on unscaled features (applies scaler.transform internally)."""
        if (hasattr(features_df, 'empty') and features_df.empty) or len(labels) == 0:
            return 0.0
        if not _TORCH_AVAILABLE or not hasattr(self.scaler, 'mean_'):
            return 0.0
        scaled = self.scaler.transform(features_df)
        t = torch.tensor(scaled.astype(np.float32))
        self.eval()
        with torch.no_grad():
            out   = self(t)
            preds = torch.argmax(out, dim=1).cpu().numpy()
        return float(accuracy_score(labels, preds))


# ─────────────────────────────────────────────────────────────────────────────
# Trade journal (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def load_trade_journal() -> list:
    if os.path.exists(TRADE_JOURNAL_FILE):
        with open(TRADE_JOURNAL_FILE, 'r') as f:
            return json.load(f)
    return []


def save_trade_journal(journal: list):
    with open(TRADE_JOURNAL_FILE, 'w') as f:
        json.dump(journal, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# Reinforcement-learning weighted retraining
# ─────────────────────────────────────────────────────────────────────────────

def perform_weighted_retraining(ai_model: 'AI_trading_bot',
                                 preprocess_func,        # accepted but unused
                                 current_journal_data: list,
                                 original_features: pd.DataFrame,
                                 original_labels: np.ndarray):
    """
    Retrain the model using historical base data combined with trade-journal
    outcomes. Real trades are weighted 2× (win) / 0.5× (loss). Virtual/paper
    trades receive reduced influence (1.1× / 0.8×) to avoid paper-trading
    noise overwriting live battle-tested signal.

    After retraining, if validation accuracy drops below the pre-retrain value
    the backup checkpoint is restored automatically.
    """
    print("\n--- Performing Weighted Retraining ---")
    completed = [t for t in current_journal_data if 'outcome' in t]
    if not completed:
        print("No completed trades for retraining. Skipping.")
        return

    feats, lbls, weights = [], [], []

    if not original_features.empty and len(original_labels) > 0:
        feats.append(original_features)
        lbls.append(original_labels)
        weights.append(np.ones(len(original_labels), dtype=np.float32))

    for trade in completed:
        entry_features = trade.get('features_at_entry')
        entry_label    = trade.get('label_at_entry')
        if not isinstance(entry_features, dict) or not isinstance(entry_label, int):
            continue

        # Build a single-row DataFrame aligned to current feature columns.
        # Old journal entries (pre-18-feature update) will be missing
        # ob_imbalance / funding_rate → fillna(0.0) makes them neutral.
        row = pd.DataFrame(
            [entry_features], columns=original_features.columns
        ).fillna(0.0)

        feats.append(row)
        lbls.append(np.array([entry_label]))

        is_virtual = trade.get('virtual', False)
        if is_virtual:
            w = 1.1 if trade['outcome'] == 'win' else 0.8
        else:
            w = WINNING_TRADE_WEIGHT if trade['outcome'] == 'win' else LOSING_TRADE_WEIGHT
        weights.append(np.array([w], dtype=np.float32))

    if not feats:
        print("No valid trades for retraining. Skipping.")
        return

    X  = pd.concat(feats, ignore_index=True)
    y  = np.concatenate(lbls)
    sw = np.concatenate(weights)

    backup_m = ai_model.model_path.replace('.pkl', '_backup.pkl')
    backup_s = ai_model.scaler_path.replace('.pkl', '_backup.pkl')
    ai_model.save_model(path=backup_m, scaler_path=backup_s)
    old_acc = ai_model.current_accuracy

    try:
        ai_model.fit(X, y, epochs=10, lr=0.001, batch_size=32,
                     sample_weights=sw)
        if ai_model.current_accuracy < old_acc:
            print(f"Val accuracy dropped ({old_acc:.4f} → "
                  f"{ai_model.current_accuracy:.4f}) — rolling back.")
            ai_model.load_model(path=backup_m, scaler_path=backup_s)
        else:
            ai_model.save_model()
    except Exception as e:
        print(f"Retraining error: {e}. Rolling back.")
        ai_model.load_model(path=backup_m, scaler_path=backup_s)
    finally:
        for p in (backup_m, backup_s):
            if os.path.exists(p):
                os.remove(p)
    print("--- Weighted Retraining Complete ---")


# ─────────────────────────────────────────────────────────────────────────────
# Bybit OHLCV fetch (with 15-second in-process cache)
# ─────────────────────────────────────────────────────────────────────────────

_kline_cache: dict = {}


def fetch_bybit_data(symbol: str = 'BTCUSDT', interval: str = '60',
                     limit: int = 500, api_key: str = None,
                     api_secret: str = None) -> pd.DataFrame:
    """Fetch historical OHLCV candles from Bybit (linear perpetuals)."""
    cache_key = f"{symbol}_{interval}_{limit}"
    now       = time.time()
    if cache_key in _kline_cache:
        cached_df, ts = _kline_cache[cache_key]
        if now - ts < 15:
            return cached_df.copy()

    session = get_bybit_client(api_key, api_secret)
    resp    = session.get_kline(category='linear', symbol=symbol,
                                interval=interval, limit=limit)
    if resp['retCode'] != 0:
        raise ValueError(f"Bybit API error: {resp['retMsg']}")

    raw = resp['result']['list']
    df  = pd.DataFrame(raw, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df = df.astype({
        'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
        'low': 'float64',     'close': 'float64', 'volume': 'float64',
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    result = df[['open', 'high', 'low', 'close', 'volume']]
    _kline_cache[cache_key] = (result.copy(), now)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation — internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 18 technical features from a closed-candle OHLCV DataFrame.

    Caller is responsible for ensuring df contains ONLY closed candles
    (i.e. the unclosed current candle has been stripped before calling).

    ob_imbalance and funding_rate are initialised to 0.0 here.
    For live inference, the caller (preprocess_single_bar) overwrites these
    with freshly fetched values before returning the row.

    Returns a DataFrame with MODEL_FEATURE_COLS columns; NaN rows dropped.
    """
    if df.empty:
        return pd.DataFrame()

    dc = df.copy()

    dc['RSI']      = ta.momentum.RSIIndicator(dc['close']).rsi()
    dc['MACD']     = ta.trend.MACD(dc['close']).macd()
    bb             = ta.volatility.BollingerBands(dc['close'])
    dc['BB_upper'] = bb.bollinger_hband()
    dc['BB_lower'] = bb.bollinger_lband()
    dc['MA_5']     = ta.trend.sma_indicator(dc['close'], window=5)
    dc['MA_10']    = ta.trend.sma_indicator(dc['close'], window=10)
    dc['ATR']      = ta.volatility.AverageTrueRange(
                         dc['high'], dc['low'], dc['close']
                     ).average_true_range()

    dc['close_lag1'] = dc['close'].shift(1)
    dc['close_lag2'] = dc['close'].shift(2)
    dc['RSI_lag1']   = dc['RSI'].shift(1)
    dc['MACD_lag1']  = dc['MACD'].shift(1)

    # Placeholders — overwritten with live values during inference
    dc['ob_imbalance'] = 0.0
    dc['funding_rate']  = 0.0

    dc.dropna(inplace=True)
    if dc.empty:
        return pd.DataFrame()

    return dc[MODEL_FEATURE_COLS].copy()


# ─────────────────────────────────────────────────────────────────────────────
# preprocess_data — training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame,
                    threshold: float = 0.0025,
                    **kwargs) -> tuple:
    """
    Full training pre-processing pipeline.

    Steps:
      1. Drop the last (unclosed/forming) candle.           [CRITICAL-2]
      2. Compute all 18 technical features.
      3. Generate labels via forward-return thresholding.   [CRITICAL-1]
         - Uses dc['close'].shift(-FORWARD_BARS) so the final FORWARD_BARS
           rows have NaN labels and are dropped automatically by dropna().
         - No label is derived from current-candle indicator values.
      4. Return (features DataFrame, labels ndarray).

    threshold kwarg is retained for API compatibility but is not used;
    see BUY_THRESHOLD / SELL_THRESHOLD module constants.
    """
    if df.empty:
        return pd.DataFrame(), np.array([])

    # CRITICAL-2: strip the unclosed candle before any computation
    df = df.iloc[:-1]

    features = _build_features_df(df)
    if features.empty:
        return pd.DataFrame(), np.array([])

    # CRITICAL-1: forward-return labels (3-candle horizon)
    features = features.copy()
    features['future_return'] = (
        features['close'].shift(-FORWARD_BARS) / features['close'] - 1
    )
    features['label'] = 1                                          # HOLD default
    features.loc[features['future_return'] >  BUY_THRESHOLD,  'label'] = 2  # BUY
    features.loc[features['future_return'] < SELL_THRESHOLD, 'label'] = 0  # SELL

    # dropna removes the last FORWARD_BARS rows (no known future return)
    features.dropna(inplace=True)
    if features.empty:
        return pd.DataFrame(), np.array([])

    labels   = features['label'].values.astype(int)
    features = features[MODEL_FEATURE_COLS]

    return features, labels



# ─────────────────────────────────────────────────────────────────────────────
# MAHORAGA — server.py-compatible wrapper around AI_trading_bot
#
# Adaptive risk engine is integrated directly into this class (no subclass).
# BASE_STOP_LOSS / BASE_TAKE_PROFIT / BASE_COOLOFF_HOURS are class-level
# constants; compute_adaptive_params() modulates them per-trade context.
#
# AutoTrader integration (server.py — do not add hardcoded constants):
#
#   # In AutoTrader.__init__():
#   self.cycle_count       = 0
#   self.atr_baseline      = 0.015
#   self.consecutive_wins  = 0        # already tracked
#   self.consecutive_losses = 0       # already tracked
#
#   # At the top of the run loop:
#   self.cycle_count += 1
#   if self.cycle_count % 100 == 0:
#       self.atr_baseline = bot.compute_atr_baseline(df)
#
#   # Before every trade entry (replace hardcoded SL/TP/cooloff refs):
#   atr_val = float(features.iloc[-1].get('ATR', 0.0))
#   close_val = float(features.iloc[-1].get('close', 1.0)) or 1.0
#   context = {
#       'atr_pct':            atr_val / close_val,
#       'atr_baseline':       self.atr_baseline,
#       'consecutive_wins':   self.consecutive_wins,
#       'consecutive_losses': self.consecutive_losses,
#       'confidence':         confidence,
#       'hour_utc':           datetime.utcnow().hour,
#   }
#   risk_params = bot.compute_adaptive_params(context)
#   bot.last_risk_params = risk_params
#   log.info(
#       f"[AdaptiveRisk] SL={risk_params['stop_loss_pct']:.3f}% "
#       f"TP={risk_params['take_profit_pct']:.3f}% "
#       f"Cooloff={risk_params['cooloff_hours']:.1f}h "
#       f"R:R={risk_params['r_ratio']} "
#       f"VolRatio={risk_params['vol_ratio']}"
#   )
#   # Use risk_params['stop_loss_pct'], ['take_profit_pct'], ['cooloff_hours']
#   # instead of STOP_LOSS_PCT, TAKE_PROFIT_PCT, COOLOFF_HOURS.
# ─────────────────────────────────────────────────────────────────────────────

class MAHORAGA:
    """
    Public interface used by server.py. Wraps AI_trading_bot (MLP) with the
    same method signatures as the previous RandomForest / LSTM wrapper so
    server.py requires no changes.

    Includes the adaptive risk engine directly:
      compute_atr_baseline(df)       → float (ATR% over last 30 days of data)
      compute_adaptive_params(ctx)   → dict  (sl_pct, tp_pct, cooloff_h, r_ratio)

    last_risk_params is updated by get_prediction_snapshot() on every poll
    and by AutoTrader before each trade entry (see integration comment above).
    """

    # ── Adaptive risk base values (Ghost-in-the-Market spec) ─────────────────
    BASE_STOP_LOSS     = 1.0   # %
    BASE_TAKE_PROFIT   = 3.0   # %
    BASE_COOLOFF_HOURS = 5.0   # hours

    def __init__(self, model_path: str = None, scaler_path: str = None):
        base = _app_dir()
        mp = model_path  or os.path.join(base, 'MAHORAGA_model.pkl')
        sp = scaler_path or os.path.join(base, 'MAHORAGA_scaler.pkl')
        self._bot    = AI_trading_bot(input_size=MODEL_INPUT_SIZE,
                                      model_path=mp, scaler_path=sp)
        self._loaded = self._bot.load_model()

        # Adaptive risk state
        self._atr_baseline = 0.015   # refreshed by compute_atr_baseline()
        self.last_risk_params = {
            'stop_loss_pct':   self.BASE_STOP_LOSS,
            'take_profit_pct': self.BASE_TAKE_PROFIT,
            'cooloff_hours':   self.BASE_COOLOFF_HOURS,
            'r_ratio':         round(self.BASE_TAKE_PROFIT / self.BASE_STOP_LOSS, 2),
            'vol_ratio':       1.0,
            'adapted':         True,
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model(self):
        """Truthy when a trained model is loaded; None otherwise."""
        return self._bot if self._loaded else None

    @property
    def scaler(self):
        return (self._bot.scaler
                if self._loaded and hasattr(self._bot.scaler, 'mean_')
                else None)

    # ── Persistence ───────────────────────────────────────────────────────────

    def load_model(self):
        self._loaded = self._bot.load_model()

    def save_model(self):
        if self._loaded:
            self._bot.save_model()

    # ── Training ──────────────────────────────────────────────────────────────

    def train_model(self, X: pd.DataFrame, y: np.ndarray, **kwargs):
        """Train MLP on features X and labels y. Saves model + scaler."""
        self._bot.fit(X, y, epochs=20, lr=0.001, batch_size=32)
        self._bot.save_model()
        self._loaded = True

    def retrain_with_journal(self, X: pd.DataFrame, y: np.ndarray,
                              journal_data: list):
        """RL-weighted retrain using historical base data + trade journal."""
        if not self._loaded:
            self.train_model(X, y)
            return
        perform_weighted_retraining(self._bot, None, journal_data, X, y)
        self._loaded = True

    # ── Feature computation ───────────────────────────────────────────────────

    def preprocess_single_bar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features on live data and return a single-row DataFrame
        representing the last CONFIRMED closed candle.

        CRITICAL-2 fix — two-step protection against unclosed candles:
          Step 1: df.iloc[:-1] strips the currently-forming candle before
                  indicator computation so no unclosed OHLCV enters the math.
          Step 2: features.iloc[-2] takes the second-to-last closed row as
                  an additional safety margin (per spec).

        ob_imbalance and funding_rate are fetched live and injected into the
        row, overwriting the 0.0 placeholders set by _build_features_df.
        """
        if df.empty:
            return pd.DataFrame()

        df_closed = df.iloc[:-1]          # CRITICAL-2 step 1: drop unclosed
        features  = _build_features_df(df_closed)

        if len(features) < 2:
            return pd.DataFrame()

        row = features.iloc[[-2]].copy()  # CRITICAL-2 step 2: per spec

        # Inject live order-book and funding-rate values
        row['ob_imbalance'] = fetch_orderbook_imbalance()
        row['funding_rate']  = fetch_funding_rate()

        return row

    # ── Signal prediction ─────────────────────────────────────────────────────

    def predict(self, features: pd.DataFrame,
                confidence_threshold: float = 0.65) -> tuple:
        """
        Returns (signal_str, confidence_float).
          signal_str      : 'BUY' | 'SELL' | 'HOLD'
          confidence_float: 0.01 – 0.99 (after overlays + clamp)

        Strategy:
          1. Try MLP output. If confidence ≥ threshold and not saturated,
             apply overlays and return.
          2. If MLP is unconfident (near-uniform ≈ 0.333) or saturated
             (> 0.97, likely out-of-distribution inputs), blend with the
             technical-indicator fallback.
          3. Apply ob_imbalance + funding_rate overlays to the final signal.
        """
        # Extract overlay inputs from feature row (0.0 if missing)
        ob_imbalance = 0.0
        funding_rate  = 0.0
        if features is not None and not (hasattr(features, 'empty') and
                                          features.empty):
            try:
                ob_imbalance = float(features.iloc[-1].get('ob_imbalance', 0.0))
                funding_rate  = float(features.iloc[-1].get('funding_rate',  0.0))
            except Exception:
                pass

        if not self._loaded:
            sig, conf = self._technical_predict(features)
            return sig, _apply_signal_overlays(sig, conf, ob_imbalance, funding_rate)

        try:
            labels, probs = self._bot.predict(features)
            if len(labels) == 0:
                sig, conf = self._technical_predict(features)
                return sig, _apply_signal_overlays(sig, conf, ob_imbalance, funding_rate)

            label      = int(labels[0])
            raw_conf   = float(probs[0][label])
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            mlp_signal = signal_map[label]

            # Saturated softmax (> 0.97): out-of-distribution input → treat uncertain
            is_saturated = raw_conf > 0.97
            mlp_conf     = min(raw_conf, 0.92)

            if mlp_conf >= confidence_threshold and not is_saturated:
                final_conf = _apply_signal_overlays(
                    mlp_signal, mlp_conf, ob_imbalance, funding_rate
                )
                return mlp_signal, final_conf

            # MLP is unconfident — blend with technical-indicator score
            tech_signal, tech_conf = self._technical_predict(features)

            # Weight MLP by how far its confidence exceeds the uniform floor (0.333)
            mlp_weight  = max(0.0, (mlp_conf - 0.333) / (1.0 - 0.333))
            tech_weight = 1.0 - mlp_weight

            if tech_signal != 'HOLD':
                blended_conf  = mlp_conf * mlp_weight + tech_conf * tech_weight
                final_signal  = (mlp_signal if mlp_signal == tech_signal
                                 else tech_signal)
                final_conf    = _apply_signal_overlays(
                    final_signal, blended_conf, ob_imbalance, funding_rate
                )
                return final_signal, final_conf

            # Both uncertain — return MLP lean with overlays
            final_conf = _apply_signal_overlays(
                mlp_signal, mlp_conf, ob_imbalance, funding_rate
            )
            return mlp_signal, final_conf

        except Exception as e:
            print(f"Predict error: {e}")
            sig, conf = self._technical_predict(features)
            return sig, _apply_signal_overlays(sig, conf, ob_imbalance, funding_rate)

    def _technical_predict(self, features: pd.DataFrame) -> tuple:
        """
        Pure RSI/MACD/MA technical signal — always produces a live confidence
        score regardless of whether the MLP has been trained. Used as fallback
        when the MLP is unconfident, saturated, or not yet loaded.
        """
        try:
            if features is None or (hasattr(features, 'empty') and
                                     features.empty):
                return 'HOLD', 0.0
            row = features.iloc[-1]

            rsi       = float(row.get('RSI',      50.0))
            macd      = float(row.get('MACD',      0.0))
            close     = float(row.get('close',     0.0))
            ma5       = float(row.get('MA_5',     close))
            rsi_lag1  = float(row.get('RSI_lag1',  rsi))
            macd_lag1 = float(row.get('MACD_lag1', macd))

            rsi_rising    = rsi  > rsi_lag1
            macd_positive = macd > 0
            macd_rising   = macd > macd_lag1
            above_ma5     = close > ma5

            buy_score  = sum([rsi > 50, rsi_rising, macd_positive,
                               macd_rising, above_ma5])
            sell_score = sum([rsi < 50, not rsi_rising, not macd_positive,
                               not macd_rising, not above_ma5])

            if buy_score >= 3:
                conf = round(min(0.85, 0.55 + (buy_score - 3) * 0.12), 4)
                return 'BUY', conf
            elif sell_score >= 3:
                conf = round(min(0.85, 0.55 + (sell_score - 3) * 0.12), 4)
                return 'SELL', conf
            else:
                return 'HOLD', 0.34
        except Exception:
            return 'HOLD', 0.0

    # ── Bybit pass-throughs ───────────────────────────────────────────────────

    def get_balance(self, api_key: str = None,
                    api_secret: str = None) -> float:
        try:
            session = get_bybit_client(api_key, api_secret)
            resp    = session.get_wallet_balance(
                accountType='UNIFIED', coin='USDT'
            )
            return float(
                resp['result']['list'][0]['coin'][0]['walletBalance']
            )
        except Exception:
            return 0.0

    def place_order(self, symbol: str, side: str, qty: float,
                    api_key: str = None, api_secret: str = None) -> dict:
        session = get_bybit_client(api_key, api_secret)
        return session.place_order(
            category='linear', symbol=symbol,
            side=side, orderType='Market',
            qty=str(qty), timeInForce='IOC',
        )

    # ── Adaptive risk engine ──────────────────────────────────────────────────

    def compute_atr_baseline(self, df: pd.DataFrame) -> float:
        """
        Compute the 30-day rolling ATR baseline as a fraction of price.

        Uses the last 2880 rows of the supplied DataFrame (30 days at 15-min
        resolution) and a 14-period ATR.  Returns the mean ATR normalised by
        mean close price so the result is dimensionless (e.g. 0.015 = 1.5%).

        Returns 0.015 (1.5% default) when the DataFrame is too short or any
        computation fails.
        """
        try:
            tail = df.tail(2880)
            if len(tail) < 14:
                return 0.015
            atr_series = ta.volatility.AverageTrueRange(
                tail['high'], tail['low'], tail['close'], window=14
            ).average_true_range()
            atr_mean   = float(atr_series.dropna().mean())
            close_mean = float(tail['close'].mean())
            if close_mean <= 0 or atr_mean <= 0:
                return 0.015
            return round(atr_mean / close_mean, 6)
        except Exception:
            return 0.015

    def compute_adaptive_params(self, context: dict) -> dict:
        """
        Compute per-trade SL%, TP%, and cooloff hours from market context.

        All three parameters start at their BASE value and are multiplied by
        a composite factor assembled from four independent signals:
          • ATR ratio     (high vol → wider SL/TP, low vol → tighter)
          • Session clock (Asian session vs NY+London overlap)
          • Win/loss streak (confidence feedback loop)
          • Model confidence (threshold-based TP widening/tightening)

        Bounds enforce ±40% of base:
          SL:      [0.60 %, 1.40 %]
          TP:      [1.80 %, 4.20 %]
          Cooloff: [3.0 h,  7.0 h]

        Minimum R:R of 1.5 is enforced after bounding TP.

        Parameters
        ----------
        context : dict with keys
            atr_pct            float  — current ATR / close price (e.g. 0.015)
            atr_baseline       float  — rolling 30-day mean ATR %
            consecutive_wins   int
            consecutive_losses int
            confidence         float  — model confidence 0.0–1.0
            hour_utc           int    — current UTC hour 0–23

        Returns
        -------
        dict with keys: stop_loss_pct, take_profit_pct, cooloff_hours,
                        r_ratio, vol_ratio, adapted
        """
        atr_pct            = float(context.get('atr_pct',            0.015))
        atr_baseline       = float(context.get('atr_baseline',        0.015))
        consecutive_wins   = int(context.get('consecutive_wins',       0))
        consecutive_losses = int(context.get('consecutive_losses',     0))
        confidence         = float(context.get('confidence',           0.65))
        hour_utc           = int(context.get('hour_utc',               12))

        if atr_baseline <= 0:
            atr_baseline = 0.015

        # vol_ratio: current ATR relative to 30-day baseline, clamped [0.5, 2.0]
        vol_ratio = max(0.5, min(2.0, atr_pct / atr_baseline))

        # ── Stop-loss adaptation ──────────────────────────────────────────────
        sl_atr_adjustment = (vol_ratio - 1.0) * 0.4   # high vol → wider SL

        if hour_utc in range(0, 6):       # Asian session — compressed volatility
            sl_session = -0.15
        elif hour_utc in range(13, 21):   # NY + London overlap — elevated vol
            sl_session = +0.15
        else:
            sl_session = 0.0

        final_sl = self.BASE_STOP_LOSS * (1.0 + sl_atr_adjustment + sl_session)
        final_sl = max(0.60, min(1.40, final_sl))

        # ── Take-profit adaptation ────────────────────────────────────────────
        tp_atr_adjustment = (vol_ratio - 1.0) * 0.4   # mirrors SL ATR scaling

        # Win streak: wider TP, momentum reward
        if consecutive_wins >= 3:
            tp_streak = +0.20
        elif consecutive_wins >= 1:
            tp_streak = +0.10
        else:
            tp_streak = 0.0

        # Loss streak overrides: tighten TP to protect remaining capital
        if consecutive_losses >= 3:
            tp_streak -= 0.20
        elif consecutive_losses >= 1:
            tp_streak -= 0.10

        # Model confidence: high-confidence signals earn wider TP runway
        if confidence >= 0.85:
            tp_conf = +0.15
        elif confidence >= 0.75:
            tp_conf = +0.08
        elif confidence < 0.68:
            tp_conf = -0.10
        else:
            tp_conf = 0.0

        final_tp = self.BASE_TAKE_PROFIT * (
            1.0 + tp_atr_adjustment + tp_streak + tp_conf
        )
        final_tp = max(1.80, min(4.20, final_tp))

        # Enforce minimum R:R ≥ 1.5 — never let TP compress below 1.5× SL
        if final_tp < final_sl * 1.5:
            final_tp = final_sl * 1.5

        # ── Cooloff adaptation ────────────────────────────────────────────────
        if consecutive_losses >= 3:
            cooloff_loss = +0.40   # enforced rest after 3-loss streak
        elif consecutive_losses >= 2:
            cooloff_loss = +0.25
        elif consecutive_losses >= 1:
            cooloff_loss = +0.10
        else:
            cooloff_loss = 0.0

        # Win momentum: shorten cooloff to stay in flow
        if consecutive_wins >= 3:
            cooloff_win = -0.20
        elif consecutive_wins >= 1:
            cooloff_win = -0.10
        else:
            cooloff_win = 0.0

        # Session: fewer setups in Asia → shorter pause; more in NY/London → shorter too
        if hour_utc in range(0, 6):
            cooloff_session = -0.15
        elif hour_utc in range(13, 21):
            cooloff_session = -0.20
        else:
            cooloff_session = 0.0

        final_cooloff = self.BASE_COOLOFF_HOURS * (
            1.0 + cooloff_loss + cooloff_win + cooloff_session
        )
        final_cooloff = max(3.0, min(7.0, final_cooloff))

        return {
            'stop_loss_pct':   round(final_sl,      4),
            'take_profit_pct': round(final_tp,      4),
            'cooloff_hours':   round(final_cooloff, 4),
            'r_ratio':         round(final_tp / final_sl, 2),
            'vol_ratio':       round(vol_ratio, 3),
            'adapted':         True,
        }

    # ── Market regime detector ────────────────────────────────────────────────

    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Classify the current market regime using the last 96 candles (24 h of
        15-min data).

        Metrics:
          directional_move — absolute close-to-close move over 96 bars as a
                             fraction of the opening price of that window.
          realized_vol     — rolling std of bar-over-bar pct returns over 96 bars.

        Returns
        -------
        "trending"  — large directional move, low realised vol (clean trend)
        "volatile"  — high realised vol regardless of direction (choppy/news)
        "ranging"   — small directional move, low realised vol (sideways)
        """
        try:
            if len(df) < 96:
                return 'ranging'
            tail             = df.tail(96)
            open_price       = float(tail['close'].iloc[0])
            if open_price == 0:
                return 'ranging'
            directional_move = abs(float(tail['close'].iloc[-1]) - open_price) / open_price
            realized_vol     = float(tail['close'].pct_change().dropna().std())
            if directional_move > 0.03 and realized_vol < 0.012:
                return 'trending'
            elif realized_vol >= 0.012:
                return 'volatile'
            else:
                return 'ranging'
        except Exception:
            return 'ranging'

    # ── Read-only prediction snapshot (NEW) ──────────────────────────────────

    async def get_prediction_snapshot(self, symbol: str = 'BTCUSDT',
                                       interval: str = '60') -> dict:
        """
        Async, read-only: returns a summary of the current model signal plus
        live market context and adaptive risk parameters. Never places or
        modifies any order.

        Adaptive risk is computed with consecutive_wins/losses = 0 because
        those counters live in AutoTrader (server.py).  AutoTrader should call
        compute_adaptive_params() directly before each trade entry and store
        the result in bot.last_risk_params so the dashboard reflects real state.

        Returns
        -------
        {
          "direction"               : "UP" | "DOWN" | "NEUTRAL",
          "confidence"              : float  (0–100, percentage),
          "ob_imbalance"            : float  (-1.0 to +1.0),
          "funding_rate"            : float  (e.g. 0.0001),
          "signal_raw"              : "BUY" | "SELL" | "HOLD",
          "candle_close_countdown_s": int    (seconds to next 15-min candle close),
          "timestamp"               : str    (ISO 8601 UTC),
          "adaptive_sl"             : float  (stop-loss %, e.g. 1.05),
          "adaptive_tp"             : float  (take-profit %, e.g. 3.15),
          "adaptive_cooloff"        : float  (hours, e.g. 4.5),
          "r_ratio"                 : float  (TP / SL, e.g. 3.0)
        }
        """
        _direction_map = {'BUY': 'UP', 'SELL': 'DOWN', 'HOLD': 'NEUTRAL'}

        try:
            df = await asyncio.to_thread(
                fetch_bybit_data,
                symbol=symbol, interval=interval, limit=300,
                api_key=os.getenv('BYBIT_API_KEY',    ''),
                api_secret=os.getenv('BYBIT_API_SECRET', ''),
            )
            features        = await asyncio.to_thread(
                self.preprocess_single_bar, df.copy()
            )
            signal, confidence = await asyncio.to_thread(
                self.predict, features, CONFIDENCE_THRESHOLD_GHOST
            )

            ob_imbalance = 0.0
            funding_rate  = 0.0
            if not features.empty:
                try:
                    ob_imbalance = float(
                        features.iloc[-1].get('ob_imbalance', 0.0)
                    )
                    funding_rate  = float(
                        features.iloc[-1].get('funding_rate',  0.0)
                    )
                except Exception:
                    pass

            # Derive current-bar ATR% for adaptive risk context
            atr_pct = 0.015
            if not features.empty:
                try:
                    atr_val   = float(features.iloc[-1].get('ATR',   0.0))
                    close_val = float(features.iloc[-1].get('close',  1.0)) or 1.0
                    if atr_val > 0:
                        atr_pct = atr_val / close_val
                except Exception:
                    pass

            # Lazily initialise the ATR baseline on first snapshot call
            if self._atr_baseline == 0.015 and not df.empty:
                self._atr_baseline = await asyncio.to_thread(
                    self.compute_atr_baseline, df
                )

            context = {
                'atr_pct':            atr_pct,
                'atr_baseline':       self._atr_baseline,
                'consecutive_wins':   0,   # not tracked here; AutoTrader provides live
                'consecutive_losses': 0,
                'confidence':         float(confidence),
                'hour_utc':           datetime.utcnow().hour,
            }
            risk_params = self.compute_adaptive_params(context)
            self.last_risk_params = risk_params

            # 2.2/2.3 — Regime detection + regime-aware confidence adjustment
            regime = await asyncio.to_thread(self.detect_regime, df)
            regime_multipliers = {'trending': 1.10, 'volatile': 0.85, 'ranging': 1.0}
            adj_confidence = max(0.01, min(0.99,
                float(confidence) * regime_multipliers.get(regime, 1.0)
            ))

            return {
                'direction':                _direction_map.get(signal, 'NEUTRAL'),
                'confidence':               round(adj_confidence * 100, 2),
                'ob_imbalance':             round(ob_imbalance, 4),
                'funding_rate':             round(funding_rate,  6),
                'signal_raw':               signal,
                'candle_close_countdown_s': 900 - (int(time.time()) % 900),
                'timestamp':                datetime.now(timezone.utc).isoformat(),
                'adaptive_sl':              risk_params['stop_loss_pct'],
                'adaptive_tp':              risk_params['take_profit_pct'],
                'adaptive_cooloff':         risk_params['cooloff_hours'],
                'r_ratio':                  risk_params['r_ratio'],
                'regime':                   regime,
            }

        except Exception as e:
            rp = self.last_risk_params
            return {
                'direction':                'NEUTRAL',
                'confidence':               0.0,
                'ob_imbalance':             0.0,
                'funding_rate':             0.0,
                'signal_raw':               'HOLD',
                'candle_close_countdown_s': 900 - (int(time.time()) % 900),
                'timestamp':                datetime.now(timezone.utc).isoformat(),
                'error':                    str(e),
                'adaptive_sl':              rp['stop_loss_pct'],
                'adaptive_tp':              rp['take_profit_pct'],
                'adaptive_cooloff':         rp['cooloff_hours'],
                'r_ratio':                  rp['r_ratio'],
                'regime':                   'unknown',
            }


# ─────────────────────────────────────────────────────────────────────────────
# VirtualTrader position-sizing helper (HIGH-2)
# ─────────────────────────────────────────────────────────────────────────────

def virtual_trader_qty(balance: float, current_price: float,
                        stop_loss_pct: float = 1.0) -> float:
    """
    Mirror of AutoTrader._get_qty() for use in server.py's VirtualTrader.

    Formula:
        risk_usdt    = balance × RISK_PER_TRADE          (1%)
        position_val = risk_usdt / (stop_loss_pct / 100) (= balance at 1% SL)
        qty          = position_val / current_price

    Produces the same notional exposure as live trading.
    Replace the old paper sizing formula with:
        qty = virtual_trader_qty(self.state['balance'], current_price)
    """
    if current_price <= 0:
        return 0.001
    risk_usdt    = balance * RISK_PER_TRADE
    position_val = risk_usdt / (stop_loss_pct / 100.0)
    qty          = round(position_val / current_price, 3)
    return max(0.001, qty)


# ─────────────────────────────────────────────────────────────────────────────
# Updated MODEL_FEATURE_COLS — 18 features in exact training order
# ─────────────────────────────────────────────────────────────────────────────
#
#  Index  Feature          Source
#  ─────  ───────────────  ────────────────────────────────────────────────
#   0     open             Raw OHLCV
#   1     high             Raw OHLCV
#   2     low              Raw OHLCV
#   3     close            Raw OHLCV
#   4     volume           Raw OHLCV
#   5     RSI              ta.momentum.RSIIndicator (period 14)
#   6     MACD             ta.trend.MACD line
#   7     BB_upper         Bollinger upper band
#   8     BB_lower         Bollinger lower band
#   9     MA_5             5-period SMA
#  10     MA_10            10-period SMA
#  11     ATR              Average True Range (14)
#  12     close_lag1       close.shift(1)
#  13     close_lag2       close.shift(2)
#  14     RSI_lag1         RSI.shift(1)
#  15     MACD_lag1        MACD.shift(1)
#  16     ob_imbalance     Live: (bid_vol - ask_vol) / total_vol   [NEW]
#  17     funding_rate     Live: current 8-h perpetual funding rate [NEW]
#
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# RETRAIN REQUIRED: Run python retrain.py after replacing this file.
# The old MAHORAGA_model.pkl and MAHORAGA_scaler.pkl are now INVALID.
#
# Reasons:
#   1. Architecture changed: LSTM → MLP (incompatible weight format).
#   2. Input size changed: 16 → 18 features (ob_imbalance, funding_rate added).
#   3. Label scheme changed: ICT indicator tautology → 3-bar forward return.
#   4. Scaler was fit on full data; new scaler is fit on train split only.
#
# Do NOT run the autotrader with old model weights after this update.
# The load_model() call will detect the old format and refuse to load it,
# leaving the system in technical-fallback mode until retrained.
