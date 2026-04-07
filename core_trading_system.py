"""
MAHORAGA core_trading_system.py
New model: PyTorch LSTM (AI_trading_bot) replacing old RandomForest (MAHORAGA).
MAHORAGA class is kept as a wrapper for server.py compatibility.
Bybit helpers are preserved for live trading data & order execution.
"""

import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import ta

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
        path        = path        or self.model_path
        scaler_path = scaler_path or self.scaler_path
        try:
            state = torch.load(path, map_location=torch.device('cpu'))
            # Reinitialise if architecture differs
            if (self.input_size  != state['input_size']  or
                self.hidden_size != state['hidden_size']  or
                self.num_layers  != state['num_layers']   or
                self.output_size != state['output_size']):
                self.__init__(
                    input_size=state['input_size'],
                    hidden_size=state['hidden_size'],
                    num_layers=state['num_layers'],
                    output_size=state['output_size'],
                    model_path=self.model_path,
                    scaler_path=self.scaler_path,
                )
            self.load_state_dict(state['state_dict'])
            self.current_accuracy = state.get('current_accuracy', 0.0)
            self.scaler = joblib.load(scaler_path)
            self.eval()
            print(f"Model loaded from {path} (accuracy={self.current_accuracy:.4f})")
            return True
        except FileNotFoundError:
            print(f"Model or scaler not found at {path} / {scaler_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, features_df):
        """Returns (np.ndarray labels, np.ndarray probabilities). Labels: 0=SELL, 1=HOLD, 2=BUY."""
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
        if isinstance(trade.get('features_at_entry'), list) and isinstance(trade.get('label_at_entry'), int):
            row = pd.DataFrame([trade['features_at_entry']], columns=original_features.columns)
            feats.append(row)
            lbls.append(np.array([trade['label_at_entry']]))
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


def fetch_bybit_data(symbol='BTCUSDT', interval='60', limit=500,
                     api_key=None, api_secret=None):
    """Fetch historical OHLCV candles from Bybit."""
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
    return df[['open', 'high', 'low', 'close', 'volume']]


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
        """
        if not self._loaded:
            return 'HOLD', 0.0
        try:
            labels, probs = self._bot.predict(features)
            if len(labels) == 0:
                return 'HOLD', 0.0
            label      = int(labels[0])
            confidence = float(probs[0][label])

            if confidence < confidence_threshold:
                return 'HOLD', confidence

            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            return signal_map[label], confidence
        except Exception as e:
            print(f"Predict error: {e}")
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
