import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

def get_bybit_client(api_key=None, api_secret=None, testnet=False):
    key = api_key or os.getenv('BYBIT_API_KEY')
    secret = api_secret or os.getenv('BYBIT_API_SECRET')
    return HTTP(testnet=testnet, api_key=key, api_secret=secret, recv_window=10000)

def fetch_bybit_data(symbol='BTCUSDT', interval='60', limit=500, api_key=None, api_secret=None):
    """
    Fetches historical OHLCV data from Bybit.
    interval: 1 3 5 15 30 60 120 240 360 720 D W M
    """
    session = get_bybit_client(api_key, api_secret)
    resp = session.get_kline(
        category='linear',
        symbol=symbol,
        interval=interval,
        limit=limit
    )
    if resp['retCode'] != 0:
        raise ValueError(f"Bybit API error: {resp['retMsg']}")

    raw = resp['result']['list']
    df = pd.DataFrame(raw, columns=['timestamp','open','high','low','close','volume','turnover'])
    df = df.astype({'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
                    'low': 'float64', 'close': 'float64', 'volume': 'float64'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df

def preprocess_data(df, lookback=60, prediction_horizon=1, threshold=0.005):
    df = dropna(df)
    df = add_all_ta_features(df, open="open", high="high", low="low",
                              close="close", volume="volume", fillna=True)
    df['future_close'] = df['close'].shift(-prediction_horizon)
    df['price_change'] = (df['future_close'] - df['close']) / df['close']
    df['label'] = np.where(df['price_change'] > threshold, 2,
                  np.where(df['price_change'] < -threshold, 0, 1))
    df.dropna(inplace=True)
    feature_cols = [col for col in df.columns if col not in ['future_close', 'price_change', 'label']]
    X = df[feature_cols]
    y = df['label']
    return X, y

class MAHORAGA:
    def __init__(self, model_path=None, scaler_path=None):
        base = os.path.dirname(__file__)
        self.model_path = model_path or os.path.join(base, 'MAHORAGA_model.pkl')
        self.scaler_path = scaler_path or os.path.join(base, 'MAHORAGA_scaler.pkl')
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("MAHORAGA model loaded successfully.")
        except Exception as e:
            print(f"Model load failed ({e}). Train a new model from the dashboard.")
            self.model = None
            self.scaler = None

    def save_model(self):
        if self.model and self.scaler:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print("MAHORAGA model saved.")

    def train_model(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def predict(self, features, confidence_threshold=0.6):
        if not self.model or not self.scaler:
            return "HOLD", 0.0
        features_scaled = self.scaler.transform([features])
        probabilities = self.model.predict_proba(features_scaled)[0]
        max_prob = np.max(probabilities)
        prediction = np.argmax(probabilities)
        if max_prob < confidence_threshold:
            return "HOLD", max_prob
        signal = ["SELL", "HOLD", "BUY"][prediction]
        return signal, max_prob

    def preprocess_single_bar(self, df):
        df = add_all_ta_features(df, open="open", high="high", low="low",
                                  close="close", volume="volume", fillna=True)
        feature_cols = [col for col in df.columns if col not in ['future_close', 'price_change', 'label']]
        return df[feature_cols].iloc[-1].values

    def place_order(self, symbol, side, qty, api_key=None, api_secret=None):
        """Place a real order on Bybit."""
        session = get_bybit_client(api_key, api_secret)
        resp = session.place_order(
            category='linear',
            symbol=symbol,
            side=side,
            orderType='Market',
            qty=str(qty),
            timeInForce='IOC'
        )
        return resp

    def get_balance(self, api_key=None, api_secret=None):
        """Get USDT wallet balance from Bybit."""
        session = get_bybit_client(api_key, api_secret)
        resp = session.get_wallet_balance(accountType='UNIFIED', coin='USDT')
        try:
            balance = resp['result']['list'][0]['coin'][0]['walletBalance']
            return float(balance)
        except Exception:
            return 0.0

# backwards-compat alias
AI_trading_bot = MAHORAGA
