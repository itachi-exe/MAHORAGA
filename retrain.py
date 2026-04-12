"""
retrain.py — MAHORAGA model retraining script.

Fetches 5000 × 15-min BTC candles from Bybit, runs the full
preprocess_data() pipeline (forward-return labels, 18 features),
trains the MLP, evaluates on a held-out validation split, and
saves MAHORAGA_model.pkl + MAHORAGA_scaler.pkl.

Usage:
    python retrain.py
"""

import os
import sys
import time

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ── Env ──────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

API_KEY     = os.getenv('BYBIT_API_KEY',    '').strip()
API_SECRET  = os.getenv('BYBIT_API_SECRET', '').strip()
SYMBOL      = os.getenv('BYBIT_SYMBOL',     'BTCUSDT').strip()
INTERVAL    = '15'          # 15-min candles
TARGET_ROWS = 5000

if not API_KEY or not API_SECRET:
    sys.exit('[retrain] ERROR: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env')

# ── Imports from core ─────────────────────────────────────────────────────────
from core_trading_system import (
    AI_trading_bot,
    preprocess_data,
    MODEL_FEATURE_COLS,
    _app_dir,
)
from pybit.unified_trading import HTTP


# ─────────────────────────────────────────────────────────────────────────────
# Paginated candle fetch (Bybit max 1000 per request → 5 pages for 5000 rows)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_candles_paginated(symbol: str, interval: str, total: int,
                             api_key: str, api_secret: str) -> pd.DataFrame:
    """
    Fetch `total` candles by paging backward through Bybit's kline endpoint.
    Each request returns at most 1000 candles (newest-first); we re-anchor
    using `end = oldest_timestamp - 1 ms` for every subsequent page.
    """
    session      = HTTP(api_key=api_key, api_secret=api_secret, recv_window=10000)
    batch_size   = 1000
    all_rows: list = []
    end_time_ms  = None
    page         = 0

    while len(all_rows) < total:
        page += 1
        kwargs = dict(category='linear', symbol=symbol,
                      interval=interval, limit=batch_size)
        if end_time_ms is not None:
            kwargs['end'] = str(end_time_ms)

        resp = session.get_kline(**kwargs)
        if resp.get('retCode') != 0:
            raise RuntimeError(f"Bybit API error: {resp.get('retMsg')}")

        batch = resp['result']['list']
        if not batch:
            print(f'  [page {page}] No more candles returned — stopping early.')
            break

        all_rows.extend(batch)
        oldest_ts = int(batch[-1][0])
        end_time_ms = oldest_ts - 1

        remaining = total - len(all_rows)
        print(f'  [page {page}] fetched {len(batch):>4} candles '
              f'(total so far: {len(all_rows):>5} / {total})')

        if len(batch) < batch_size:
            print(f'  [page {page}] Bybit returned fewer than {batch_size} — end of history.')
            break

    # Trim, convert, sort
    all_rows = all_rows[:total]
    df = pd.DataFrame(all_rows,
                      columns=['timestamp', 'open', 'high', 'low',
                                'close', 'volume', 'turnover'])
    df = df.astype({
        'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
        'low':       'float64', 'close': 'float64', 'volume': 'float64',
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    sep = '─' * 60

    print(sep)
    print('  MAHORAGA — Model Retraining')
    print(sep)
    print(f'  Symbol   : {SYMBOL}')
    print(f'  Interval : {INTERVAL}-min candles')
    print(f'  Target   : {TARGET_ROWS} candles')
    print(f'  Features : {len(MODEL_FEATURE_COLS)} ({", ".join(MODEL_FEATURE_COLS)})')
    print()

    # ── Step 1: Fetch candles ─────────────────────────────────────────────────
    print('[1/5] Fetching candles from Bybit …')
    df = fetch_candles_paginated(SYMBOL, INTERVAL, TARGET_ROWS, API_KEY, API_SECRET)
    elapsed_fetch = time.time() - t0
    print(f'      Received {len(df)} candles  '
          f'({df.index[0].strftime("%Y-%m-%d %H:%M")} → '
          f'{df.index[-1].strftime("%Y-%m-%d %H:%M")})')
    print(f'      Fetch time: {elapsed_fetch:.1f}s')
    print()

    if len(df) < 200:
        sys.exit('[retrain] ERROR: Too few candles to train — check API keys / symbol.')

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    print('[2/5] Running preprocess_data() …')
    t_prep = time.time()
    features, labels = preprocess_data(df)
    elapsed_prep = time.time() - t_prep

    if features.empty or len(labels) == 0:
        sys.exit('[retrain] ERROR: preprocess_data() returned empty dataset. '
                 'Check that the candles contain valid OHLCV data.')

    n_total = len(labels)
    n_buy   = int((labels == 2).sum())
    n_hold  = int((labels == 1).sum())
    n_sell  = int((labels == 0).sum())
    pct_buy  = n_buy  / n_total * 100
    pct_hold = n_hold / n_total * 100
    pct_sell = n_sell / n_total * 100

    print(f'      Prep time  : {elapsed_prep:.1f}s')
    print(f'      Raw candles: {len(df)}  →  labeled samples: {n_total}')
    print()
    print('      CLASS DISTRIBUTION')
    print(f'      BUY  (2): {n_buy:>5}  ({pct_buy:5.1f}%)')
    print(f'      HOLD (1): {n_hold:>5}  ({pct_hold:5.1f}%)')
    print(f'      SELL (0): {n_sell:>5}  ({pct_sell:5.1f}%)')
    print()

    # Warn on extreme imbalance
    WARN_THRESHOLD = 5.0  # %
    for name, pct in [('BUY', pct_buy), ('HOLD', pct_hold), ('SELL', pct_sell)]:
        if pct < WARN_THRESHOLD:
            print(f'  ⚠  WARNING: {name} class is only {pct:.1f}% of samples '
                  f'(below {WARN_THRESHOLD:.0f}% threshold). '
                  'Model may struggle to learn this class. '
                  'Consider adjusting BUY_THRESHOLD / SELL_THRESHOLD.')
    print()

    # ── Step 3: Instantiate model ─────────────────────────────────────────────
    print('[3/5] Instantiating AI_trading_bot (MLP, input_size=18) …')
    base_dir    = _app_dir()
    model_path  = os.path.join(base_dir, 'MAHORAGA_model.pkl')
    scaler_path = os.path.join(base_dir, 'MAHORAGA_scaler.pkl')

    ai_model = AI_trading_bot(
        input_size=18,
        hidden=128,
        output_size=3,
        model_path=model_path,
        scaler_path=scaler_path,
    )
    print(f'      Architecture : Linear(18→128)→ReLU→Drop(0.2)'
          f'→Linear(128→64)→ReLU→Drop(0.1)→Linear(64→3)')
    print(f'      Train split  : 85%  ({int(n_total * 0.85)} samples)')
    print(f'      Val split    : 15%  ({int(n_total * 0.15)} samples, shuffle=False)')
    print(f'      Epochs       : 20   lr=0.001  batch=32')
    print()

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    print('[4/5] Training …')
    t_train = time.time()
    ai_model.fit(features, labels, epochs=20, lr=0.001, batch_size=32)
    elapsed_train = time.time() - t_train

    val_acc = ai_model.current_accuracy
    print()
    print(f'      Training time    : {elapsed_train:.1f}s')
    print(f'      Val accuracy     : {val_acc * 100:.2f}%')

    if val_acc < 0.52:
        print(f'  ⚠  WARNING: Validation accuracy {val_acc*100:.2f}% is below 52%. '
              'The model is performing near or below random chance. '
              'Possible causes: insufficient data, extreme class imbalance, '
              'or features not predictive at this horizon/threshold. '
              'Consider adjusting BUY_THRESHOLD / SELL_THRESHOLD or fetching more data.')
    else:
        print(f'      Status           : OK — accuracy above 52% baseline')
    print()

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    print('[5/5] Saving model …')
    ai_model.save_model(path=model_path, scaler_path=scaler_path)
    print(f'      Model  → {model_path}')
    print(f'      Scaler → {scaler_path}')
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    print(sep)
    print('  RETRAIN COMPLETE')
    print(sep)
    print(f'  Samples (total)  : {n_total}')
    print(f'  BUY / HOLD / SELL: {n_buy} / {n_hold} / {n_sell}  '
          f'({pct_buy:.1f}% / {pct_hold:.1f}% / {pct_sell:.1f}%)')
    print(f'  Val accuracy     : {val_acc * 100:.2f}%')
    print(f'  Total runtime    : {elapsed_total:.1f}s')
    print(sep)

    if val_acc >= 0.52:
        print()
        print('  Model is ready. Start the autotrader from the dashboard.')
    else:
        print()
        print('  Model saved but accuracy is low. Review warnings above before live trading.')


if __name__ == '__main__':
    main()
