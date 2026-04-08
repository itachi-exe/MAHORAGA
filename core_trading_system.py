"""
MAHORAGA core_trading_system.py
New model: PyTorch LSTM (AI_trading_bot) replacing old RandomForest (MAHORAGA).
MAHORAGA class is kept as a wrapper for server.py compatibility.
Bybit helpers are preserved for live trading data & order execution.
"""

import os
import sys
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import ta

# ── PyTorch — optional. If not installed the LSTM model is disabled and the
#    system falls back to the technical-indicator predict automatically. ─────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Provide dummy nn.Module base so the class definition below doesn't crash
    class _FakeModule:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return None
        def eval(self): pass
        def parameters(self): return iter([])
        def state_dict(self): return {}
        def load_state_dict(self, *a, **kw): pass
        def to(self, *a, **kw): return self
    class nn:
        Module = _FakeModule
        class LSTM:
            def __init__(self, *a, **kw): pass
        class Linear:
            def __init__(self, *a, **kw): pass
        CrossEntropyLoss = _FakeModule
    class optim:
        Adam = _FakeModule
    class F:
        @staticmethod
        def softmax(x, dim=None): return x

# ── Optional Alpaca imports (not required for live trading via Bybit) ──────────
try:
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# ── Bybit (live trading) ────────────────────────────────────────────────────────
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from datetime import datetime

def _app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def _bundle_dir():
    return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

load_dotenv(dotenv_path=os.path.join(_bundle_dir(), '.env'), override=False)
load_dotenv(dotenv_path=os.path.join(_app_dir(), '.env'), override=True)

# ── GHOSTINTHEMARKET ALGORITHM Constants ────────────────────────────────────────
TRADE_JOURNAL_FILE         = 'trade_journal.json'
RETRADING_CYCLE_LENGTH     = 20        # retrain after every 20 completed trades
WINNING_TRADE_WEIGHT       = 2.0
LOSING_TRADE_WEIGHT        = 0.5
RISK_PER_TRADE             = 0.01      # 1%
MAX_DAILY_LOSS             = 0.03      # 3%
MAX_DAILY_PROFIT           = 0.10      # 10%
MAX_TRADES_PER_DAY         = 10
MAX_CONSECUTIVE_LOSSES     = 3
CONFIDENCE_THRESHOLD_GHOST = 0.65
TRAILING_STOP_PERCENTAGE   = 0.01      # 1%
TAKE_PROFIT_PERCENTAGE     = 0.03      # 3%
MAX_POSITIONS_PER_DIRECTION = 3

# ── Feature columns used by model ──────────────────────────────────────────────
MODEL_FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'RSI', 'MACD', 'BB_upper', 'BB_lower', 'MA_5', 'MA_10',
    'ATR', 'close_lag1', 'close_lag2', 'RSI_lag1', 'MACD_lag1'
]
MODEL_INPUT_SIZE = len(MODEL_FEATURE_COLS)  # 16


# ── AI_trading_bot: PyTorch LSTM model ─────────────────────────────────────────
class AI_trading_bot(nn.Module):
    """
    LSTM-based trading bot predicting BUY (2), HOLD (1), SELL (0).
    Architecture: LSTM → FC → Softmax
    """

    def __init__(self, input_size=MODEL_INPUT_SIZE, hidden_size=64, num_layers=2,
                 output_size=3,
                 model_path='MAHORAGA_model.pkl',
                 scaler_path='MAHORAGA_scaler.pkl'):
        super(AI_trading_bot, self).__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.output_size  = output_size
        self.lstm         = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc           = nn.Linear(hidden_size, output_size)
        self.scaler       = StandardScaler()
        self.model_path   = model_path
        self.scaler_path  = scaler_path
        self.current_accuracy = 0.0

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return F.softmax(out, dim=1)

    def fit(self, features_df, labels, epochs=20, lr=0.001, batch_size=32, sample_weights=None):
        if not _TORCH_AVAILABLE:
            print("torch not installed — cannot train LSTM. Install torch first.")
            return
        if features_df.empty or len(labels) == 0:
            print("Features DataFrame or labels are empty, cannot fit the model.")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if sample_weights is not None:
            if len(sample_weights) != len(features_df):
                raise ValueError("Length of sample_weights must match length of features_df.")
            sw = torch.tensor(sample_weights, dtype=torch.float32).to(device)
        else:
            sw = torch.ones(len(features_df), dtype=torch.float32).to(device)

        if hasattr(self.scaler, 'mean_'):
            # Continual Learning: update variance shift without permanently erasing old memories
            self.scaler.partial_fit(features_df)
            scaled = self.scaler.transform(features_df)
        else:
            scaled = self.scaler.fit_transform(features_df)
        joblib.dump(self.scaler, self.scaler_path)
        features_t = torch.tensor(scaled, dtype=torch.float32)
        labels_t   = torch.tensor(labels,  dtype=torch.long)
        criterion  = nn.CrossEntropyLoss(reduction='none')
        optimizer  = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        for epoch in range(epochs):
            for i in range(0, len(features_t), batch_size):
                bf  = features_t[i:i+batch_size].to(device)
                bl  = labels_t[i:i+batch_size].to(device)
                bw  = sw[i:i+batch_size]
                out = self(bf)
                loss = (criterion(out, bl) * bw).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.current_accuracy = self.evaluate_model(features_df, labels)

    def save_model(self, path=None, scaler_path=None):
        path       = path       or self.model_path
        scaler_path = scaler_path or self.scaler_path
        state = {
            'state_dict':      self.state_dict(),
            'current_accuracy': self.current_accuracy,
            'input_size':      self.input_size,
            'hidden_size':     self.hidden_size,
            'num_layers':      self.num_layers,
            'output_size':     self.output_size,
        }
        torch.save(state, path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {path} (accuracy={self.current_accuracy:.4f})")

    def load_model(self, path=None, scaler_path=None):
        if not _TORCH_AVAILABLE:
            print("torch not installed — LSTM disabled, using technical fallback.")
            # Load scaler only so feature names are available
            try:
                self.scaler = joblib.load(scaler_path or self.scaler_path)
            except Exception:
                pass
            return False
        path        = path        or self.model_path
        scaler_path = scaler_path or self.scaler_path
        try:
            raw = torch.load(path, map_location=torch.device('cpu'))

            # Detect format: new format has metadata keys; old format is raw state_dict
            if isinstance(raw, dict) and 'state_dict' in raw:
                # ── New format: {state_dict, input_size, hidden_size, ...} ──────
                saved_input  = raw['input_size']
                saved_hidden = raw['hidden_size']
                saved_layers = raw['num_layers']
                saved_output = raw['output_size']
                state_dict   = raw['state_dict']
                self.current_accuracy = raw.get('current_accuracy', 0.0)
            else:
                # ── Raw state_dict format (weights only) ────────────────────────
                # Infer architecture from weight tensor shapes:
                #   lstm.weight_ih_l0  shape: (4*hidden, input)
                #   fc.weight          shape: (output, hidden)
                state_dict   = raw
                w_ih         = raw['lstm.weight_ih_l0']
                saved_hidden = w_ih.shape[0] // 4
                saved_input  = w_ih.shape[1]
                saved_output = raw['fc.weight'].shape[0]
                # Count LSTM layers by key presence
                saved_layers = sum(1 for k in raw if k.startswith('lstm.weight_ih_l'))
                self.current_accuracy = 0.0

            # Reinitialise if architecture differs from current instance
            if (self.input_size  != saved_input  or
                self.hidden_size != saved_hidden  or
                self.num_layers  != saved_layers  or
                self.output_size != saved_output):
                self.__init__(
                    input_size=saved_input,
                    hidden_size=saved_hidden,
                    num_layers=saved_layers,
                    output_size=saved_output,
                    model_path=self.model_path,
                    scaler_path=self.scaler_path,
                )

            self.load_state_dict(state_dict)
            self.scaler = joblib.load(scaler_path)
            self.eval()
            print(f"LSTM loaded (input={saved_input}, hidden={saved_hidden}, "
                  f"layers={saved_layers}, accuracy={self.current_accuracy:.4f})")
            return True
        except FileNotFoundError:
            print(f"Model or scaler not found at {path} / {scaler_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, features_df):
        """Returns (np.ndarray labels, np.ndarray probabilities). Labels: 0=SELL, 1=HOLD, 2=BUY."""
        if not _TORCH_AVAILABLE:
            return np.array([]), np.array([])
        if self.scaler is None or not hasattr(self.scaler, 'mean_'):
            return np.array([]), np.array([])
        scaled = self.scaler.transform(features_df)
        t = torch.tensor(scaled, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            out = self(t)
            return torch.argmax(out, dim=1).cpu().numpy(), out.cpu().numpy()

    def evaluate_model(self, features_df, labels):
        if features_df.empty or len(labels) == 0:
            return 0.0
        scaled = self.scaler.transform(features_df)
        t = torch.tensor(scaled, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            out = self(t)
            preds = torch.argmax(out, dim=1).cpu().numpy()
        return accuracy_score(labels, preds)


# ── Trade Journal ───────────────────────────────────────────────────────────────
def load_trade_journal():
    if os.path.exists(TRADE_JOURNAL_FILE):
        with open(TRADE_JOURNAL_FILE, 'r') as f:
            return json.load(f)
    return []

def save_trade_journal(journal):
    with open(TRADE_JOURNAL_FILE, 'w') as f:
        json.dump(journal, f, indent=4)


def perform_weighted_retraining(ai_model, preprocess_func, current_journal_data,
                                original_features, original_labels):
    print("\n--- Performing Weighted Retraining ---")
    completed = [t for t in current_journal_data if 'outcome' in t]
    if not completed:
        print("No completed trades for retraining. Skipping.")
        return

    feats, lbls, weights = [], [], []
    if not original_features.empty and len(original_labels) > 0:
        feats.append(original_features)
        lbls.append(original_labels)
        weights.append(np.ones(len(original_labels)))

    for trade in completed:
        if isinstance(trade.get('features_at_entry'), dict) and isinstance(trade.get('label_at_entry'), int):
            row = pd.DataFrame([trade['features_at_entry']], columns=original_features.columns)
            feats.append(row)
            lbls.append(np.array([trade['label_at_entry']]))
            
            is_virtual = trade.get('virtual', False)
            if is_virtual:
                # Simulated paper trades get reduced network impact to avoid overwriting real battle data.
                w = 1.1 if trade['outcome'] == 'win' else 0.8
            else:
                # Real trades command the highest authority
                w = WINNING_TRADE_WEIGHT if trade['outcome'] == 'win' else LOSING_TRADE_WEIGHT
                
            weights.append(np.array([w]))

    if not feats:
        print("No valid trades for retraining. Skipping.")
        return

    X = pd.concat(feats, ignore_index=True)
    y = np.concatenate(lbls)
    sw = np.concatenate(weights)

    backup_m = ai_model.model_path.replace('.pkl', '_backup.pkl')
    backup_s = ai_model.scaler_path.replace('.pkl', '_backup.pkl')
    ai_model.save_model(path=backup_m, scaler_path=backup_s)
    old_acc = ai_model.current_accuracy

    try:
        ai_model.fit(X, y, epochs=10, lr=0.001, batch_size=32, sample_weights=sw)
        if ai_model.current_accuracy < old_acc:
            print("New accuracy lower — rolling back.")
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


# ── Bybit helpers (used by server.py for live data & orders) ───────────────────
def get_bybit_client(api_key=None, api_secret=None, testnet=False):
    key    = api_key    or os.getenv('BYBIT_API_KEY')
    secret = api_secret or os.getenv('BYBIT_API_SECRET')
    return HTTP(testnet=testnet, api_key=key, api_secret=secret, recv_window=10000)


import time
_kline_cache = {}

def fetch_bybit_data(symbol='BTCUSDT', interval='60', limit=500,
                     api_key=None, api_secret=None):
    """Fetch historical OHLCV candles from Bybit."""
    cache_key = f"{symbol}_{interval}_{limit}"
    now = time.time()
    if cache_key in _kline_cache:
        cached_df, timestamp = _kline_cache[cache_key]
        if now - timestamp < 15:
            return cached_df.copy()

    session = get_bybit_client(api_key, api_secret)
    resp = session.get_kline(category='linear', symbol=symbol,
                             interval=interval, limit=limit)
    if resp['retCode'] != 0:
        raise ValueError(f"Bybit API error: {resp['retMsg']}")
    raw = resp['result']['list']
    df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype({'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
                    'low': 'float64', 'close': 'float64', 'volume': 'float64'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    res = df[['open', 'high', 'low', 'close', 'volume']]
    _kline_cache[cache_key] = (res.copy(), now)
    return res


# ── preprocess_data: builds 16-feature dataset for the LSTM model ──────────────
def preprocess_data(df, threshold=0.0025, **kwargs):
    """
    Compute technical indicators and generate BUY/HOLD/SELL labels.

    Args:
        df        : DataFrame with open/high/low/close/volume columns.
        threshold : (legacy kwarg, accepted but not used — labelling uses ICT conditions)

    Returns:
        (features DataFrame, labels ndarray)  — same API as old version.
    """
    if df.empty:
        return pd.DataFrame(), np.array([])

    dc = df.copy()

    dc['RSI']      = ta.momentum.RSIIndicator(dc['close']).rsi()
    dc['MACD']     = ta.trend.MACD(dc['close']).macd()
    bb             = ta.volatility.BollingerBands(dc['close'])
    dc['BB_upper'] = bb.bollinger_hband()
    dc['BB_lower'] = bb.bollinger_lband()
    dc['MA_5']     = ta.trend.sma_indicator(dc['close'], window=5)
    dc['MA_10']    = ta.trend.sma_indicator(dc['close'], window=10)
    dc['ATR']      = ta.volatility.AverageTrueRange(dc['high'], dc['low'], dc['close']).average_true_range()

    dc['close_lag1'] = dc['close'].shift(1)
    dc['close_lag2'] = dc['close'].shift(2)
    dc['RSI_lag1']   = dc['RSI'].shift(1)
    dc['MACD_lag1']  = dc['MACD'].shift(1)

    dc.dropna(inplace=True)

    # ICT-style labelling
    dc['label'] = 1  # HOLD
    buy_cond  = (dc['RSI'].diff() > 0) & (dc['RSI'] > 40) & (dc['MACD'] > 0) & (dc['close'] > dc['MA_5'])
    sell_cond = (dc['RSI'].diff() < 0) & (dc['RSI'] < 60) & (dc['MACD'] < 0) & (dc['close'] < dc['MA_5'])
    dc.loc[buy_cond,  'label'] = 2
    dc.loc[sell_cond, 'label'] = 0
    dc.dropna(inplace=True)

    features = dc[MODEL_FEATURE_COLS]
    labels   = dc['label'].values
    return features, labels


# ── Alpaca data fetch (optional — only used outside of live Bybit trading) ──────
def fetch_alpaca_data(symbol, timeframe, start_date, end_date,
                      asset_class='crypto', api_key_id=None, api_secret_key=None):
    if not _ALPACA_AVAILABLE:
        print("alpaca-py not installed. Cannot fetch Alpaca data.")
        return pd.DataFrame()
    if not api_key_id or not api_secret_key:
        print("Alpaca API keys not provided.")
        return pd.DataFrame()

    stock_client  = StockHistoricalDataClient(api_key_id, api_secret_key)
    crypto_client = CryptoHistoricalDataClient(api_key_id, api_secret_key)

    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt   = datetime.strptime(end_date,   '%Y-%m-%d')
    except ValueError:
        print("Dates must be YYYY-MM-DD.")
        return pd.DataFrame()

    tf_map = {'1m': (1, 'Minute'), '5m': (5, 'Minute'), '15m': (15, 'Minute'),
              '1H': (1, 'Hour'),   '1D': (1, 'Day')}
    amt, unit_name = tf_map.get(timeframe, (15, 'Minute'))
    alpaca_tf = TimeFrame(amount=amt, unit=getattr(TimeFrameUnit, unit_name))

    try:
        if asset_class == 'crypto':
            req  = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=alpaca_tf,
                                     start=start_dt, end=end_dt)
            bars = crypto_client.get_crypto_bars(req)
        else:
            req  = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=alpaca_tf,
                                    start=start_dt, end=end_dt)
            bars = stock_client.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Alpaca fetch error: {e}")
        return pd.DataFrame()


# ── MAHORAGA: backward-compatible wrapper around AI_trading_bot ─────────────────
class MAHORAGA:
    """
    Drop-in replacement for the old RandomForest-based MAHORAGA class.
    Wraps AI_trading_bot (PyTorch LSTM) with the same public interface so
    server.py requires zero changes to its own logic.
    """

    def __init__(self, model_path=None, scaler_path=None):
        base = _app_dir()
        mp = model_path  or os.path.join(base, 'MAHORAGA_model.pkl')
        sp = scaler_path or os.path.join(base, 'MAHORAGA_scaler.pkl')
        self._bot     = AI_trading_bot(input_size=MODEL_INPUT_SIZE,
                                       model_path=mp, scaler_path=sp)
        self._loaded  = self._bot.load_model()

    # ── Properties expected by server.py ──────────────────────────────────────
    @property
    def model(self):
        """None when model is not loaded; truthy when loaded. Checked as `if not bot.model`."""
        return self._bot if self._loaded else None

    @property
    def scaler(self):
        return self._bot.scaler if self._loaded and hasattr(self._bot.scaler, 'mean_') else None

    # ── load / save ────────────────────────────────────────────────────────────
    def load_model(self):
        self._loaded = self._bot.load_model()

    def save_model(self):
        if self._loaded:
            self._bot.save_model()

    # ── train ──────────────────────────────────────────────────────────────────
    def train_model(self, X, y, **kwargs):
        """Train the LSTM on features X and labels y. Saves model & scaler automatically."""
        self._bot.fit(X, y, epochs=20, lr=0.001, batch_size=32)
        self._bot.save_model()
        self._loaded = True

    def retrain_with_journal(self, X, y, journal_data):
        """Perform reinforcement learning using historical 1000 candles + actual trade journal."""
        if not self._loaded:
            # Fallback to normal train if starting from scratch
            self.train_model(X, y)
            return
        # Use weighted retraining
        perform_weighted_retraining(self._bot, None, journal_data, X, y)
        self._loaded = True

    # ── preprocess ────────────────────────────────────────────────────────────
    def preprocess_single_bar(self, df):
        """Compute indicators and return the **latest row** as a single-row DataFrame."""
        features, _ = preprocess_data(df)
        if features.empty:
            return features
        return features.iloc[[-1]]

    # ── predict ───────────────────────────────────────────────────────────────
    def predict(self, features, confidence_threshold=0.65):
        """
        Returns (signal_str, confidence_float).
        signal_str: 'BUY' | 'SELL' | 'HOLD'
        confidence_float: 0.0 – 1.0

        Strategy:
        - Use LSTM output when its confidence is high (model well-trained).
        - When LSTM is uncertain (near-uniform ~0.33, e.g. model not yet trained
          on current market data), fall back to a technical-indicator score built
          from the same 16 features so the dashboard always shows live values.
        """
        if not self._loaded:
            return self._technical_predict(features)
        try:
            labels, probs = self._bot.predict(features)
            if len(labels) == 0:
                return self._technical_predict(features)

            label         = int(labels[0])
            raw_conf      = float(probs[0][label])
            signal_map    = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            lstm_signal   = signal_map[label]

            # Detect saturated softmax (> 0.97 on any single class) — sign that
            # the model is receiving out-of-distribution scaled inputs.
            # In that case treat as uncertain and blend with technical score.
            is_saturated = raw_conf > 0.97
            # Cap at 0.92 so risk system isn't fooled by a spuriously perfect score
            lstm_conf = min(raw_conf, 0.92)

            # LSTM is well-calibrated (not saturated) — use its output directly
            if lstm_conf >= confidence_threshold and not is_saturated:
                return lstm_signal, lstm_conf

            # LSTM is uncertain (near-uniform).
            # Blend with technical score so confidence is always meaningful.
            tech_signal, tech_conf = self._technical_predict(features)

            # Weight LSTM by how far its confidence is above the uniform floor (0.33)
            lstm_weight = max(0.0, (lstm_conf - 0.333) / (1.0 - 0.333))
            tech_weight = 1.0 - lstm_weight

            if tech_signal != 'HOLD':
                blended_conf = lstm_conf * lstm_weight + tech_conf * tech_weight
                # Prefer agreement; otherwise trust the stronger signal
                final_signal = lstm_signal if lstm_signal == tech_signal else tech_signal
                return final_signal, round(blended_conf, 4)

            # Both uncertain — return whatever the LSTM leaned toward
            return lstm_signal, lstm_conf

        except Exception as e:
            print(f"Predict error: {e}")
            return self._technical_predict(features)

    def _technical_predict(self, features):
        """
        Pure RSI/MACD/MA technical signal — always produces a live confidence
        score regardless of whether the LSTM has been trained on current data.
        Used as fallback when the LSTM is unconfident or not yet loaded.
        """
        try:
            if features is None or (hasattr(features, 'empty') and features.empty):
                return 'HOLD', 0.0
            row = features.iloc[-1]

            rsi      = float(row.get('RSI',      50.0))
            macd     = float(row.get('MACD',      0.0))
            close    = float(row.get('close',     0.0))
            ma5      = float(row.get('MA_5',     close))
            rsi_lag1 = float(row.get('RSI_lag1',  rsi))
            macd_lag1 = float(row.get('MACD_lag1', macd))

            rsi_rising    = rsi  > rsi_lag1
            macd_positive = macd > 0
            macd_rising   = macd > macd_lag1
            above_ma5     = close > ma5

            buy_score  = sum([rsi > 50, rsi_rising, macd_positive, macd_rising, above_ma5])
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

    # ── Bybit helpers (pass-through) ──────────────────────────────────────────
    def get_balance(self, api_key=None, api_secret=None):
        try:
            session = get_bybit_client(api_key, api_secret)
            resp = session.get_wallet_balance(accountType='UNIFIED', coin='USDT')
            return float(resp['result']['list'][0]['coin'][0]['walletBalance'])
        except Exception:
            return 0.0

    def place_order(self, symbol, side, qty, api_key=None, api_secret=None):
        session = get_bybit_client(api_key, api_secret)
        return session.place_order(
            category='linear', symbol=symbol,
            side=side, orderType='Market',
            qty=str(qty), timeInForce='IOC',
        )
